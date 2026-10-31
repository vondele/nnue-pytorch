"""Pull-style tiled forward of the double feature transformer (default forward path; see docs/TILED_FT.md).

The 16-list tiled gather (``tiled_ft_forward.tiled_gather``) keeps its accumulators in shared memory:
every (entry, column) costs a shared-memory read and write, which made it shared-memory bound. This
kernel turns the loop around, like the tiled backward:

- tiles are the backward's tiles of TL = 256 lists (same list order, same row-sorted keys);
- the tile's distinct rows (ascending) are staged chunk by chunk (R rows x SLICE columns, cp.async,
  double-buffered) into shared memory, so every distinct row of a 256-list tile is read once per slice;
- every warp owns LPW lists and keeps their accumulators in registers (lane = columns); per chunk it
  adds, for each of its lists, the staged rows the list uses, in ascending row order.

The sum of every output element is bias + rows in ascending row order, exactly as in the 16-list
tiled gather, so the two forwards are bit-identical.

Preparation (index-only, prefetch stream, from the backward's preparation):
  ``tile_rows``  distinct rows of every tile, ascending (int32), ``drow_start`` (n_tiles + 1);
  ``chunk_start`` (n_tiles + 1) global chunk index of every tile's first chunk;
  ``cnt``        (n_chunks, TL) uint8: entries of list l in chunk c;
  ``wstart``     (n_chunks, NW) int32: first entry of warp w in chunk c (entries are chunk-major,
                 then list, then row);
  ``ent``        (E,) uint8: row of the entry inside its chunk.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from . import tiled_ft_backward as _tb

import os
TL = int(os.environ.get("NNUE_PULL_TL", "64"))  # lists per tile (divides the backward's tile)
NS = int(os.environ.get("NNUE_PULL_NS", "2"))   # cp.async pipeline stages
R = int(os.environ.get("NNUE_PULL_R", "32"))            # staged rows per chunk (entries store the row inside the chunk as uint8)
LPW = int(os.environ.get("NNUE_PULL_LPW", "4"))  # lists per warp
NW = TL // LPW    # warps per block
VEC = int(os.environ.get("NNUE_PULL_VEC", "4"))  # columns per lane
SLICE = 32 * VEC  # columns per block
MAX_CHUNKS = 16384 // TL   # shared counters of the list kernel (max distinct rows per tile = R * MAX_CHUNKS)

_HDR = "typedef int int32_t; typedef long long int64_t; typedef unsigned int uint32_t; typedef unsigned char uint8_t; typedef unsigned short uint16_t;\n"

_LISTS_SRC = _HDR + r"""
// One block per tile. Entries arrive row-sorted (the backward's keys); seg_incl[e] - 1 - drow_start[t]
// is the entry's rank among the tile's distinct rows. Counting sort by (chunk, list), then every
// (chunk, list) bucket is sorted by row (deterministic; buckets hold ~1-2 entries).
extern "C" __global__ void pull_lists(
    const uint32_t* __restrict__ e_key, const int32_t* __restrict__ seg_incl, const int32_t* __restrict__ tile_start,
    const int32_t* __restrict__ drow_start, const int32_t* __restrict__ chunk_start,
    uint8_t* __restrict__ cnt_g, int32_t* __restrict__ wstart_g, uint8_t* __restrict__ ent_g)
{
    const int32_t TL = @TL@, R = @R@, LPW = @LPW@, NW = @NW@, MAXC = @MAXC@, LOCAL_BITS = @LOCAL_BITS@;
    __shared__ __align__(16) uint16_t sc[MAXC * TL];
    __shared__ int32_t tsum[TL];
    const int32_t t = blockIdx.x, l = threadIdx.x;
    const int32_t e0 = tile_start[t], e1 = tile_start[t + 1];
    const int32_t dr = drow_start[t] + 1;
    const int32_t c0 = chunk_start[t], nch = chunk_start[t + 1] - c0;
    for (int i = l; i < nch * TL; i += TL) sc[i] = 0;
    __syncthreads();
    for (int32_t e = e0 + l; e < e1; e += TL) {
        const int32_t r = seg_incl[e] - dr;
        const int32_t ll = (int32_t)(e_key[e] & ((1u << LOCAL_BITS) - 1));
        atomicAdd((unsigned int*)&sc[((r / R) * TL + ll) & ~1], ((r / R) * TL + ll) & 1 ? 65536u : 1u);
    }
    __syncthreads();
    // exclusive prefix over the nch * TL counters (chunk-major, then list): thread l scans [l*nch, (l+1)*nch)
    int32_t s = 0;
    for (int i = l * nch; i < (l + 1) * nch; ++i) s += sc[i];
    tsum[l] = s;
    __syncthreads();
    if (l == 0) { int32_t a = 0; for (int i = 0; i < TL; ++i) { const int32_t v = tsum[i]; tsum[i] = a; a += v; } }
    __syncthreads();
    {
        int32_t a = tsum[l];
        for (int i = l * nch; i < (l + 1) * nch; ++i) {
            const int32_t v = sc[i];
            const int32_t c = i / TL, ll = i - c * TL;
            cnt_g[(int64_t)(c0 + c) * TL + ll] = (uint8_t)v;
            if (ll % LPW == 0) wstart_g[(int64_t)(c0 + c) * NW + ll / LPW] = e0 + a;
            sc[i] = (uint16_t)a;
            a += v;
        }
    }
    __syncthreads();
    for (int32_t e = e0 + l; e < e1; e += TL) {
        const int32_t r = seg_incl[e] - dr;
        const int32_t ll = (int32_t)(e_key[e] & ((1u << LOCAL_BITS) - 1));
        const int32_t c = r / R, i = c * TL + ll;
        const unsigned old = atomicAdd((unsigned int*)&sc[i & ~1], i & 1 ? 65536u : 1u);
        const int32_t pos = (i & 1) ? (int32_t)(old >> 16) : (int32_t)(old & 0xFFFFu);
        ent_g[e0 + pos] = (uint8_t)(r - c * R);
    }
    __syncthreads();
    // sort list l's bucket of every chunk (start = cursor - count)
    for (int c = 0; c < nch; ++c) {
        const int32_t n = cnt_g[(int64_t)(c0 + c) * TL + l];
        if (n < 2) continue;
        uint8_t* bk = ent_g + e0 + sc[c * TL + l] - n;
        for (int i = 1; i < n; ++i) {
            const uint8_t v = bk[i]; int j = i;
            while (j > 0 && bk[j - 1] > v) { bk[j] = bk[j - 1]; --j; }
            bk[j] = v;
        }
    }
}
"""

_PULL_SRC = _HDR + r"""
__device__ __forceinline__ void cp_async16(void* smem, const void* gmem, int src_bytes) {
    const unsigned s = (unsigned)__cvta_generic_to_shared(smem);
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" :: "r"(s), "l"(gmem), "r"(src_bytes));
}
__device__ __forceinline__ void cp_async_commit() { asm volatile("cp.async.commit_group;\n" ::); }
template <int N> __device__ __forceinline__ void cp_async_wait() { asm volatile("cp.async.wait_group %0;\n" :: "n"(N)); }

#define VT @VT@
__device__ __forceinline__ float2 vadd(float2 a, float2 b) { return make_float2(a.x + b.x, a.y + b.y); }
__device__ __forceinline__ float4 vadd(float4 a, float4 b) { return make_float4(a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w); }
extern "C" __global__ void __launch_bounds__(@NW@ * 32) pull_gather(
    const float* __restrict__ weight, const float* __restrict__ bias, const int32_t* __restrict__ order,
    const int32_t* __restrict__ tile_rows, const int32_t* __restrict__ drow_start, const int32_t* __restrict__ chunk_start,
    const uint8_t* __restrict__ cnt_g, const int32_t* __restrict__ wstart_g, const uint8_t* __restrict__ ent_g,
    float* __restrict__ out_w, float* __restrict__ out_b, const int32_t n_lists, const int32_t batch_size)
{
    const int32_t TL = @TL@, R = @R@, LPW = @LPW@, NW = @NW@, SLICE = @SLICE@, OUT = @OUT@, VEC = SLICE / 32;
    const int32_t NS = @NS@;
    extern __shared__ __align__(16) float wsh_dyn[];   // NS stages of R * SLICE floats
    const int32_t t = blockIdx.x;
    const int32_t col0 = blockIdx.y * SLICE;
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    const int32_t d0 = drow_start[t], nd = drow_start[t + 1] - d0;
    const int32_t c0 = chunk_start[t], nch = chunk_start[t + 1] - c0;
    const VT bv = *(const VT*)(bias + col0 + VEC * lane);
    VT acc[LPW];
#pragma unroll
    for (int j = 0; j < LPW; ++j) acc[j] = bv;
    // staging: R rows x SLICE floats = R * SLICE / 4 pieces of 16 B
    const int NT = NW * 32, PPR = SLICE / 4, PPT = R * SLICE / 4 / NT;   // 16-byte pieces per thread per chunk
    int32_t rowreg[PPT];
    auto load_rows = [&](int c) {
#pragma unroll
        for (int i = 0; i < PPT; ++i) {
            const int32_t ri = c * R + (tid + i * NT) / PPR;
            rowreg[i] = (c < nch && ri < nd) ? __ldg(&tile_rows[d0 + ri]) : -1;
        }
    };
    auto stage = [&](int buf) {               // stages the chunk whose rows are in rowreg
#pragma unroll
        for (int i = 0; i < PPT; ++i) {
            const int p = tid + i * NT, rr = p / PPR, q = p - rr * PPR;
            const int32_t row = rowreg[i];
            cp_async16(wsh_dyn + buf * (R * SLICE) + rr * SLICE + q * 4,
                       weight + (int64_t)(row < 0 ? 0 : row) * OUT + col0 + q * 4, row < 0 ? 0 : 16);
        }
        cp_async_commit();
    };
#pragma unroll
    for (int s = 0; s < NS - 1; ++s) { load_rows(s); stage(s); }
    load_rows(NS - 1);
    const int32_t clast = c0 + (nch > 0 ? nch - 1 : 0);
    const int32_t mlane = warp * LPW + (lane % LPW);
    int32_t ws = __ldg(&wstart_g[(int64_t)c0 * NW + warp]);
    int32_t ws_n = __ldg(&wstart_g[(int64_t)min(c0 + 1, clast) * NW + warp]);
    int32_t cn = (int32_t)__ldg(&cnt_g[(int64_t)c0 * TL + mlane]);
    int32_t ent = (int32_t)__ldg(&ent_g[ws + lane]);
    for (int32_t c = 0; c < nch; ++c) {
        const int buf = c % NS;
        cp_async_wait<NS - 2>();
        __syncthreads();                       // chunk c staged; every warp is done with chunk c - 1
        stage((c + NS - 1) % NS);              // chunk c + NS - 1 (empty pieces past the tile)
        load_rows(c + NS);
        // metadata of chunk c + 1 (and the warp start of c + 2) while chunk c is processed
        const int32_t gn = min(c0 + c + 1, clast);
        const int32_t cn_n = (int32_t)__ldg(&cnt_g[(int64_t)gn * TL + mlane]);
        const int32_t ent_n = (int32_t)__ldg(&ent_g[ws_n + lane]);
        const int32_t ws_nn = __ldg(&wstart_g[(int64_t)min(c0 + c + 2, clast) * NW + warp]);
        // exclusive prefix of the counts of this warp's lists (lanes 0..LPW-1)
        int32_t pre = (lane < LPW) ? cn : 0;
#pragma unroll
        for (int o = 1; o < LPW; o <<= 1) { const int32_t v = __shfl_up_sync(0xFFFFFFFFu, pre, o); if (lane >= o) pre += v; }
        pre -= (lane < LPW) ? cn : 0;
        const float* wb = wsh_dyn + buf * (R * SLICE) + VEC * lane;
#pragma unroll
        for (int j = 0; j < LPW; ++j) {
            const int32_t nj = __shfl_sync(0xFFFFFFFFu, cn, j);
            const int32_t oj = __shfl_sync(0xFFFFFFFFu, pre, j);
            int32_t k = 0;
            for (; k + 1 < nj; k += 2) {
                const int32_t i0 = oj + k, i1 = i0 + 1;
                int32_t r0 = __shfl_sync(0xFFFFFFFFu, ent, i0 & 31);
                int32_t r1 = __shfl_sync(0xFFFFFFFFu, ent, i1 & 31);
                if (i1 >= 32) { if (i0 >= 32) r0 = (int32_t)__ldg(&ent_g[ws + i0]); r1 = (int32_t)__ldg(&ent_g[ws + i1]); }
                const VT v0 = *(const VT*)(wb + r0 * SLICE);
                const VT v1 = *(const VT*)(wb + r1 * SLICE);
                acc[j] = vadd(acc[j], v0);
                acc[j] = vadd(acc[j], v1);
            }
            if (k < nj) {
                const int32_t i0 = oj + k;
                int32_t r0 = __shfl_sync(0xFFFFFFFFu, ent, i0 & 31);
                if (i0 >= 32) r0 = (int32_t)__ldg(&ent_g[ws + i0]);
                const VT v0 = *(const VT*)(wb + r0 * SLICE);
                acc[j] = vadd(acc[j], v0);
            }
        }
        cn = cn_n; ent = ent_n; ws = ws_n; ws_n = ws_nn;
    }
#pragma unroll
    for (int j = 0; j < LPW; ++j) {
        const int32_t li = t * TL + warp * LPW + j;
        if (li < n_lists) {
            const int32_t lid = __ldg(&order[li]);
            float* dst = lid < batch_size ? out_w + (int64_t)lid * OUT : out_b + (int64_t)(lid - batch_size) * OUT;
            *(VT*)(dst + col0 + VEC * lane) = acc[j];
        }
    }
}
"""


def _lists_kernel():
    src = _tb._render(_LISTS_SRC, TL=TL, R=R, LPW=LPW, NW=NW, MAXC=MAX_CHUNKS, LOCAL_BITS=_tb.LOCAL_BITS)
    return _tb._get_kernel(("pull_lists", TL, R, LPW), src, "pull_lists")


def _pull_kernel(out_size: int):
    src = _tb._render(_PULL_SRC, TL=TL, R=R, LPW=LPW, NW=NW, SLICE=SLICE, OUT=out_size, NS=NS, VT={2: "float2", 4: "float4"}[VEC])
    shm = NS * R * SLICE * 4
    assert (R * SLICE // 4) % (NW * 32) == 0, "R * SLICE / 4 must be a multiple of the block size"
    return _tb._get_kernel(("pull_gather", TL, R, LPW, SLICE, out_size, NS, VEC), src, "pull_gather", dynamic_shared=max(shm, 1))


@dataclass
class PullPrep:
    n_lists: int
    n_tiles: int
    order: torch.Tensor
    tile_rows: torch.Tensor
    drow_start: torch.Tensor
    chunk_start: torch.Tensor
    cnt: torch.Tensor
    wstart: torch.Tensor
    ent: torch.Tensor

    def record_stream(self, stream):
        for t in (self.order, self.tile_rows, self.drow_start, self.chunk_start, self.cnt, self.wstart, self.ent):
            t.record_stream(stream)


def prepare(white_indices: torch.Tensor, black_indices: torch.Tensor, bp) -> PullPrep | None:
    """From the backward's preparation ``bp`` (tiles of TL lists, row-sorted keys). None when a tile
    has more than R * MAX_CHUNKS distinct rows (the caller then uses the 16-list gather)."""
    dev = white_indices.device
    if bp.T != TL:
        if bp.T % TL:
            return None
        from . import tiled_ft_forward as _tf
        bp = _tf._derive_from_backward(bp, TL, dev, None)
    if getattr(bp, "max_tile", 65536) >= 65536:   # 16-bit prefix counters of the list kernel
        return None
    B, maxa = white_indices.shape
    E = int(bp.e_key.shape[0])
    if bp.max_tile > R * MAX_CHUNKS:      # distinct rows of a tile <= its entries (max_tile: no extra host sync)
        return None
    # no host syncs below: sizes use upper bounds, masks are replaced by scatters
    rows = bp.e_key >> _tb.LOCAL_BITS
    starts = torch.zeros(E + 1, dtype=torch.bool, device=dev)
    starts[1:E] = rows[1:] != rows[:-1]
    starts[bp.tile_start[:-1].long()] = True                       # index E (empty last tiles) lands in the pad slot
    starts = starts[:E]
    seg_before = torch.cumsum(starts, 0, dtype=torch.int32)          # inclusive
    tile_rows = torch.empty(max(E, 1), dtype=torch.int32, device=dev)
    tile_rows[(seg_before - 1).long()] = rows                          # every entry of a segment writes the same row
    drow_start = torch.zeros(bp.n_tiles + 1, dtype=torch.int32, device=dev)
    ends = bp.tile_start[1:].long()
    drow_start[1:] = torch.where(ends > 0, seg_before[(ends - 1).clamp(min=0)], torch.zeros_like(ends, dtype=torch.int32))
    nd = drow_start[1:] - drow_start[:-1]
    nch = (nd + R - 1) // R
    chunk_start = torch.zeros(bp.n_tiles + 1, dtype=torch.int32, device=dev)
    chunk_start[1:] = torch.cumsum(nch, 0)
    n_chunks = E // R + bp.n_tiles                                    # upper bound of sum(ceil(nd / R))
    cnt = torch.empty(max(n_chunks, 1) * TL, dtype=torch.uint8, device=dev)
    wstart = torch.empty(max(n_chunks, 1) * NW, dtype=torch.int32, device=dev)
    ent = torch.zeros(E + 64, dtype=torch.uint8, device=dev)          # padded: the gather reads 32 entries past a warp start
    with _tb._cupy_stream(dev):
        _lists_kernel()(grid=(bp.n_tiles,), block=(TL,), args=(
            bp.e_key.data_ptr(), seg_before.data_ptr(), bp.tile_start.data_ptr(), drow_start.data_ptr(),
            chunk_start.data_ptr(), cnt.data_ptr(), wstart.data_ptr(), ent.data_ptr()))
    return PullPrep(bp.n_lists, bp.n_tiles, bp.order, tile_rows, drow_start, chunk_start, cnt, wstart, ent)


def gather(weight, bias, l1_size, prep: PullPrep, out_w, out_b):
    """Columns 0..l1_size of the two outputs (B, OUT) in original batch order."""
    OUT = bias.shape[0]
    B = out_w.shape[0]
    with _tb._cupy_stream(weight.device):
        _pull_kernel(OUT)(grid=(prep.n_tiles, l1_size // SLICE), block=(NW * 32,), shared_mem=NS * R * SLICE * 4, args=(
            weight.data_ptr(), bias.data_ptr(), prep.order.data_ptr(), prep.tile_rows.data_ptr(), prep.drow_start.data_ptr(),
            prep.chunk_start.data_ptr(), prep.cnt.data_ptr(), prep.wstart.data_ptr(), prep.ent.data_ptr(),
            out_w.data_ptr(), out_b.data_ptr(), np.int32(prep.n_lists), np.int32(B)))
