"""H100 master-net FT backward with tile-local feature aggregation.

Pack the union of features in eight positions with a position/perspective mask.
Each column block sums those contributions before issuing global atomics.
The experimental compact path uses scaled FP16 pairs and recomputes in FP32
on overflow, entirely on the current CUDA stream.
Feature indices must be unique within each position/perspective, as guaranteed
by the feature extractors. Repeated indices across rows are aggregated exactly
apart from floating-point rounding.
"""

from functools import cache

import cupy as cp
import numpy as np
import torch

from .fused_ft_kernel import _num_threads


@cache
def _kernels(active: int, width: int, compact: bool = False, fallback: bool = False):
    A = active
    half = width // 2
    threads = _num_threads(half, 128)
    tile = 8
    maxids = 2 * tile * A
    h = 1 << (maxids - 1).bit_length()
    prefix = f"#define COMPACT {int(compact)}\n#define FALLBACK {int(fallback)}\n"
    prefix += r"""
#include <cuda_fp16.h>
#if COMPACT
typedef __half grad_t;
__device__ __forceinline__ void add_grad(grad_t* p, float v) {
    // Whole warps participate; adjacent lanes address aligned column pairs.
    float other = __shfl_down_sync(0xffffffffu, v, 1);
    if ((threadIdx.x & 1) == 0 && (v != 0 || other != 0)) {
        __half2 h = __floats2half2_rn(v * 65536.0f, other * 65536.0f);
        unsigned packed = *reinterpret_cast<unsigned*>(&h);
        asm volatile("red.relaxed.gpu.global.add.noftz.f16x2 [%0], %1;"
                     :: "l"(p), "r"(packed) : "memory");
    }
}
#else
typedef float grad_t;
__device__ __forceinline__ void add_grad(grad_t* p, float v) { atomicAdd(p, v); }
#endif
"""
    code = (
        prefix +
        f"#define T {tile}\n#define A {A}\n#define H {h}\n#define M {maxids}\n"
        f"#define K {width}\n#define HALF {half}\n#define B {threads}\n"
        + r"""
extern "C" __global__ void ft_pack_features(const int *w, const int *b, int *ids, unsigned *masks,
                                            int *counts, int batch_size) {
    extern __shared__ unsigned mem[];
    int *keys = (int *)mem;
    unsigned *bits = mem + H;
    __shared__ int n;
    if (threadIdx.x == 0) {
        n = 0;
    }
    for (int i = threadIdx.x; i < H; i += blockDim.x) {
        keys[i] = -1;
        bits[i] = 0;
    }
    __syncthreads();
    for (int i = threadIdx.x; i < M; i += blockDim.x) {
        int row = i / A, k = i % A;
        int pos = blockIdx.x * T + row / 2;
        if (pos >= batch_size)
            continue;
        int id = (row & 1) ? b[pos * A + k] : w[pos * A + k];
        if (id < 0)
            continue;
        unsigned slot = ((unsigned)id * 2654435761u) & (H - 1);
        while (true) {
            int old = atomicCAS(keys + slot, -1, id);
            if (old == -1 || old == id) {
                if (old == -1) {
                    // Only the thread that claims a new hash slot appends it.
                    int offset = atomicAdd(&n, 1);
                    ids[blockIdx.x * M + offset] = slot;
                }
                atomicOr(bits + slot, 1u << row);
                break;
            }
            slot = (slot + 1) & (H - 1);
        }
    }
    __syncthreads();
    // Resolve the occupied-slot list in place after all masks are complete.
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        int slot = ids[blockIdx.x * M + i];
        ids[blockIdx.x * M + i] = keys[slot];
        masks[blockIdx.x * M + i] = bits[slot];
    }
    __syncthreads();
    if (threadIdx.x == 0)
        counts[blockIdx.x] = n;
}
extern "C" __global__ void ft_aggregate_backward(const float *us, const float *them,
                                                 const float *gl, const float *cl, float maxact,
                                                 const int *ids,
                                                 const unsigned *masks, const int *counts,
                                                 grad_t *gw, float *gb, int batch_size, const int *overflow) {
    if (FALLBACK && !*overflow) return;
    __shared__ float g[2 * T][(2*B)];
    unsigned tid = threadIdx.x, col = tid + B * blockIdx.y;
    float bias0 = 0, bias1 = 0;
    for (int t = 0; t < T; ++t) {
        int row = blockIdx.x * T + t;
        if (row >= batch_size)
            break;
        float w0 = cl[row * (2*K) + col], w1 = cl[row * (2*K) + HALF + col];
        float b0 = cl[row * (2*K) + K + col], b1 = cl[row * (2*K) + (3*HALF) + col];
        float d0 = gl[row * K + col], d1 = gl[row * K + HALF + col];
        float dw0 = (w0 == 0 || w0 == maxact) ? 0 : d0 * w1;
        float dw1 = (w1 == 0 || w1 == maxact) ? 0 : d0 * w0;
        float db0 = (b0 == 0 || b0 == maxact) ? 0 : d1 * b1;
        float db1 = (b1 == 0 || b1 == maxact) ? 0 : d1 * b0;
        float u = us[row], v = them[row];
        float gw0 = u * dw0 + v * db0, gw1 = u * dw1 + v * db1;
        float gb0 = v * dw0 + u * db0, gb1 = v * dw1 + u * db1;
        g[2 * t][tid] = gw0;
        g[2 * t][tid + B] = gw1;
        g[2 * t + 1][tid] = gb0;
        g[2 * t + 1][tid + B] = gb1;
        bias0 += gw0 + gb0;
        bias1 += gw1 + gb1;
    }
    __syncthreads();
    int n = counts[blockIdx.x];
    for (int i = 0; i < n; ++i) {
        // Form the row offset in pointer width before adding column offsets.
        size_t id = (unsigned)ids[blockIdx.x * M + i];
        unsigned mask = masks[blockIdx.x * M + i];
        // Packed masks are nonempty; avoid an add-to-zero and a loop
        // iteration for the common case of a single contributing row.
        int r = __ffs(mask) - 1;
        float v0 = g[r][tid], v1 = g[r][tid + B];
        while ((mask &= mask - 1)) {
            r = __ffs(mask) - 1;
            v0 += g[r][tid];
            v1 += g[r][tid + B];
        }
        if (COMPACT || v0 != 0)
            add_grad(gw + id * K + col, v0);
        if (COMPACT || v1 != 0)
            add_grad(gw + id * K + HALF + col, v1);
    }
    if (!FALLBACK && bias0 != 0)
        atomicAdd(gb + col, bias0);
    if (!FALLBACK && bias1 != 0)
        atomicAdd(gb + HALF + col, bias1);
}
"""
    )
    pack = cp.RawKernel(code, "ft_pack_features")
    pack.compile()
    pack.max_dynamic_shared_size_bytes = h * 8
    aggregate = cp.RawKernel(code, "ft_aggregate_backward")
    aggregate.compile()
    return pack, aggregate, maxids, h * 8, threads


@torch.compiler.disable
def aggregated_ft_backward(
    us, them, white, black, grad, clamped, grad_weight, grad_bias, maxact, compact=False
):
    """Accumulate on the current stream; compact mode overwrites grad_weight.

    Biases always accumulate in FP32. Compact weight gradients are scaled by
    65536 to reduce underflow, then unpacked into FP32 for autograd/DDP/Adam.
    Nonfinite half accumulations trigger a conditional FP32 recomputation;
    there is no host read or synchronization in the normal path.
    Nonnegative feature indices must be unique within each white/black row.
    """
    batch_size, active = white.shape
    tiles = (batch_size + 7) // 8
    with cp.cuda.Device(us.device.index):
        pack, aggregate, capacity, shared_bytes, threads = _kernels(active, grad.shape[1], compact)
        ids = torch.empty((tiles, capacity), device=us.device, dtype=torch.int32)
        masks = torch.empty_like(ids)
        counts = torch.empty(tiles, device=us.device, dtype=torch.int32)
        pack_args = (
            white.data_ptr(),
            black.data_ptr(),
            ids.data_ptr(),
            masks.data_ptr(),
            counts.data_ptr(),
            np.int32(batch_size),
        )
        work_weight = torch.zeros_like(grad_weight, dtype=torch.float16) if compact else grad_weight
        overflow = torch.zeros(1, device=us.device, dtype=torch.int32) if compact else None
        backward_args = (
            us.data_ptr(),
            them.data_ptr(),
            grad.data_ptr(),
            clamped.data_ptr(),
            np.float32(maxact),
            ids.data_ptr(),
            masks.data_ptr(),
            counts.data_ptr(),
            work_weight.data_ptr(),
            grad_bias.data_ptr(),
            np.int32(batch_size),
            overflow.data_ptr() if compact else 0,
        )
        with cp.cuda.ExternalStream(torch.cuda.current_stream(us.device).cuda_stream):
            pack((tiles,), (512,), pack_args, shared_mem=shared_bytes)
            aggregate((tiles, grad.shape[1] // (2 * threads)), (threads,), backward_args)

            if compact:
                unpack, clear = _conversion_kernels()
                pairs = grad_weight.numel() // 2
                grid = ((pairs + 255) // 256,)
                unpack(grid, (256,), (work_weight.data_ptr(), grad_weight.data_ptr(),
                                     overflow.data_ptr(), np.int32(pairs)))
                # A separate launch makes overflow visible to every block.
                clear((256,), (256,), (grad_weight.data_ptr(), overflow.data_ptr(), np.int32(pairs)))
                _, recover, _, _, _ = _kernels(active, grad.shape[1], fallback=True)
                recovery_args = (*backward_args[:8], grad_weight.data_ptr(), *backward_args[9:])
                recover((tiles, grad.shape[1] // (2 * threads)), (threads,), recovery_args)


@cache
def _conversion_kernels():
    code = r"""
#include <cuda_fp16.h>
extern "C" __global__ void unpack_scaled_gradient(const __half2* x, float2* y, int* overflow, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float2 f = __half22float2(x[i]);
        if (!isfinite(f.x) || !isfinite(f.y)) atomicExch(overflow, 1);
        f.x *= 1.0f / 65536.0f;
        f.y *= 1.0f / 65536.0f;
        y[i] = f;
    }
}
extern "C" __global__ void clear_overflow_gradient(float2* y, const int* overflow, int n) {
    if (!*overflow) return;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    for (; i < n; i += blockDim.x * gridDim.x) y[i] = make_float2(0.0f, 0.0f);
}
"""
    return cp.RawKernel(code, "unpack_scaled_gradient"), cp.RawKernel(code, "clear_overflow_gradient")
