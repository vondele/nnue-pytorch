# Tiled feature-transformer forward + backward (default on CUDA)

`model/modules/feature_transformer/tiled_ft_forward.py` and `tiled_ft_backward.py` replace the per-position
gather and the atomic scatter of the fused double feature transformer (`FusedDoubleFtFunction`,
`fused_ft_functions.py`) when CUDA + CuPy are available. `tiled_ft_fused.py` connects them: the tiled
forward produces the two raw accumulators (white, black), and a per-position epilogue applies the
perspective mix, clamp and pairwise product, writing the same `l0_` and `clamped_out` as
`fused_ft_kernel.py`. The backward computes the accumulator gradients from `grad_l0` and `clamped_out`
(same formulas as the fused backward) and hands them to the tiled scatter. Same math, fp32 sums in a
different order.

- `NNUE_TILED_FORWARD=0` selects upstream's fused forward, `NNUE_TILED_BACKWARD=0` upstream's backward
  (direct scatter, or the tile-local aggregation where `_use_aggregated_backward` enables it). Both are
  read once at import; either combination works.
- Requirements: forward `L1 % 512 == 0`; backward `L1 % 64 == 0`. Other shapes use upstream's kernels.

## Idea

Positions in a batch share most of their active feature rows. The 2B perspective lists are ordered by
[king bucket][4 min-hash values] so that consecutive lists are similar, then grouped into tiles.

- **Backward** (tiles of 256 lists): each tile's (row, list) entries are sorted by row, and one block
  per (tile, 64-column slice) holds the tile's output gradients in shared memory. Each row segment is
  summed in registers and written with ONE atomicAdd per (row, column) per tile, instead of one per
  (position, row, column). The bias gradient comes from the tile column sums. The PSQT columns use
  plain atomics, skipping zero gradients.
- **Forward** (tiles of 16 lists, derived from the backward's tiles by a stable partition): one
  block per (tile, 512-column slice) keeps the 16 accumulator rows in shared memory and reads every
  distinct weight row of the tile once. A warp-per-list kernel handles the PSQT columns.

## Index preparation

The preparation reads only the indices, so the loader's prefetch worker computes it on its own CUDA
stream (`data_loader.dataset._register_tiled_prep`). It stores the result in a side table keyed by
the `white_indices` tensor (weak reference + data_ptr + shape check); the batch tuple is not changed.
Without a registered preparation, the forward uses the plain kernel (`tiled_ft_forward.STATS["fallback"]`)
and the backward prepares itself (`tiled_ft_backward.STATS["fallback"]`).
`NNUE_TILED_PREFETCH=0` disables the prefetch-side preparation.

Tuning variables: `NNUE_TILED_T` / `NNUE_TILED_SLICE` / `NNUE_TILED_WARPS` / `NNUE_TILED_ORDER` /
`NNUE_TILED_SORT` / `NNUE_TILED_GRID` (backward); `NNUE_TILED_FWD_T` / `NNUE_TILED_FWD_SLICE` /
`NNUE_TILED_FWD_VEC` / `NNUE_TILED_FWD_P` (forward); `NNUE_TILED_HK_OFF` is the first king-bucketed row
of the composed feature set (default 64368 = Full_Threats + PP_3Wide, i.e. the start of K32H3), used
only by the ordering heuristic.

## Accuracy

Real batch (B = 65536, Full_Threats+PP_3Wide+K32H3^, L1 = 1024) against an fp64 reference that uses
each method's own fp32 clamp decisions, as max |error| / max |reference|:

| | upstream fused (direct scatter) | tiled |
|---|---|---|
| `l0_` | 5.3e-7 | 4.6-5.2e-7 |
| weight gradient | 4.2-4.5e-6 | 4.3-5.0e-7 |
| bias gradient | 3.9-4.9e-6 | 5.0-7.1e-7 |

About 3 of the 2.7e8 clamp decisions per batch differ between the two forwards (fp32 sum order), which
is why their gradients differ by more than either differs from its reference. `tests/test_tiled_ft.py`
checks a 1e-6 relative bound against fp64 and compares with upstream's kernels.

## Speed

RTX 3090 (sm_86, SM clock pinned at 1500 MHz), default model, plain training loop (real loader with 16
workers, model + loss + backward + fused Adam, no Lightning/torch.compile), arms interleaved.
`upstream (aggregated)` forces upstream's tile-local aggregation backward, which is off by default on sm_86.

| batch | upstream | upstream (aggregated) | tiled | tiled vs upstream |
|---|---|---|---|---|
| 65536 | 10.6 it/s | 13.3 it/s | 18.3 it/s | 1.73x |
| 131072 | 1.80 it/s* | 2.35 it/s* | 3.24-3.33 it/s* | 1.83x |

*Measured while another job shared the GPU, so only the ratios are meaningful. Under the same sharing,
65536 gave 3.24 / 4.05 / 5.5 it/s (1.70x).
