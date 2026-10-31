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
- **Forward** (default: the pull gather, `tiled_ft_pull.py`, tiles of 64 lists): one block per
  (tile, 128-column slice) stages the tile's distinct rows (ascending) chunk by chunk (32 rows,
  `cp.async`, double-buffered) into shared memory; each of its 16 warps keeps the accumulators of 4
  lists in registers (lane = 4 columns) and adds, per chunk, the staged rows each list uses. Every
  output is bias + rows in ascending row order, exactly as in the 16-list gather below, so the two are
  bit-identical. `NNUE_TILED_PULL=0` selects the 16-list gather (tiles of 16 lists, accumulators in
  shared memory; it was bound by the shared-memory read + write per (entry, column)). A warp-per-list
  kernel handles the PSQT columns and stops at the end of each (left-packed) list.

## Index preparation

The preparation reads only the indices, so the loader's prefetch worker computes it on its own CUDA
stream (`data_loader.dataset._register_tiled_prep`). It stores the result in a side table keyed by
the `white_indices` tensor (weak reference + data_ptr + shape check); the batch tuple is not changed.
Without a registered preparation, the forward uses the plain kernel (`tiled_ft_forward.STATS["fallback"]`)
and the backward prepares itself (`tiled_ft_backward.STATS["fallback"]`).
`NNUE_TILED_PREFETCH=0` disables the prefetch-side preparation.

The 256-list tiles (~23k entries) are too large for one shared-memory sort; their 64-list sub-tiles
are sorted in shared memory and the four sorted runs of a tile are merged by rank (the keys are unique
across runs), which gives the same keys as a global sort. The pull forward uses the sorted sub-tiles
directly. The pull preparation (`pull_lists`: per (32-row chunk, list) counts and the chunk-major
entry stream) adds no host sync. The prefetch stream runs at high priority so the preparation keeps
ahead of the training step.

Tuning variables: `NNUE_TILED_T` / `NNUE_TILED_SLICE` / `NNUE_TILED_WARPS` / `NNUE_TILED_ORDER` /
`NNUE_TILED_SORT` / `NNUE_TILED_GRID` (backward); `NNUE_TILED_FWD_T` / `NNUE_TILED_FWD_SLICE` /
`NNUE_TILED_FWD_VEC` / `NNUE_TILED_FWD_P` (16-list forward); `NNUE_PULL_TL` / `NNUE_PULL_R` /
`NNUE_PULL_LPW` / `NNUE_PULL_VEC` / `NNUE_PULL_NS` (pull forward: lists per tile, rows per chunk,
lists per warp, columns per lane, pipeline stages); `NNUE_TILED_HK_OFF` is the first king-bucketed row
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

## Fused weight and activation fake quantization

`fused_weight_quant.py` replaces two elementwise chains around the FT (`NNUE_FUSED_WEIGHT_QUANT=0`
restores the reference chains). Both are bit-identical in forward and backward to the reference.

- The merged FT weight (cat of the feature weights, virtual weight repeated over the buckets, then the
  STE fake quantization) is written in one pass; the backward hands each weight its slice of the merged
  gradient and sums the virtual slice over the buckets. For the default model this is a ~570 MB tensor
  that the reference chain passed over about eight times per step.
- `fake_quantize_ft_act(l0_) * l0_correction_factor` is one kernel (fp32 roundings as PyTorch's scalar
  ops, no FMA contraction); its gradient is `grad * c`. When the tiled forward runs, the same
  roundings are applied in the FT epilogue (`act_quant` argument of `double_feature_transform`), so
  the unquantized `l0_` is never written.

RTX 3090 at 1500 MHz, default model, B = 65536, same plain loop as above: 17.5 -> 23.2 it/s (1.33x).
Per step (42.8 ms of GPU time): tiled FT forward 10.6 ms, tiled FT backward 9.7 ms, fused Adam 5.0 ms,
index preparation 4.8 ms (on the loader's stream, overlapped), layer stacks ~7 ms, weight merge 1.7 ms.
The loader alone delivers ~34 batches/s with 16 workers, so training stays GPU bound.

## tiled-ft-4

RTX 3090 at 1500 MHz, real `train.py` + torch.compile, B = 65536, 16 loader workers, 3 epochs x 400
steps, last epoch's it/s, arms interleaved: tiled-ft-3 24.4-24.9 it/s; pull forward + high-priority
prefetch stream 25.4; + sub-tile sort and lane-major scatter loads 26.9; + fused activation
quantization and the faster tile sort 27.2-27.9. Per step (nsys, after the pull forward + sort changes): main stream 98.9% busy; tiled scatter
7.6 ms, pull gather 7.3 ms (the 16-list gather took 9.3 ms in isolation on a real batch, the pull
gather 7.0 ms), fused optimizer 6.1 ms.

Exactness: the forward (FT output, PSQT columns, fused activation quantization, and the whole model
output on real batches) is bit-identical to tiled-ft-3. The backward's per-tile sums are unchanged;
weight gradients differ from tiled-ft-3 by no more than tiled-ft-3 differs from itself between runs
(atomic order, ~9e-7 relative).
