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

## tiled-ft-5

Same setup as tiled-ft-4 (RTX 3090 at 1500 MHz, real `train.py` + torch.compile, B = 65536, 16 loader
workers, 3 epochs x 400 steps, last epoch's it/s, arms interleaved): tiled-ft-4 27.85-27.96 it/s (8 runs);
tiled-ft-5 29.83-29.95 it/s (7 runs, plus one outlier at 29.02), 1.07x. With `NNUE_L1_DENSE_FORWARD=0` (the old L1 forward) tiled-ft-5
gives 29.07.

- **Tiled scatter**: the entry walk issues the shared-memory loads of 8 entries before their adds and
  fetches the next 32 keys one batch ahead. Same entry order, per-row sums and flushes; 8.3 -> 7.7 ms
  in isolation on a real batch (random output gradients). Single-tile batches (where the result is
  deterministic) give bit-identical L1-column gradients to tiled-ft-4.
- **Grouped L1 forward** (`grouped_linear._bucketed_forward`): a per-bucket fp32 (IEEE) Triton matmul
  over the routed rows replaces the NNZ-compacted kernel, whose summation order depended on
  shared-memory atomics (so it was not deterministic). Each output is the sum of 16-wide dot chunks
  in order (an identity `mov` keeps Triton from fusing them into one 1024-long chain), then the bias.
  1.79 -> 0.72 ms in isolation (K = 1024, N = 32, 32 buckets, 60% zero activations). Against fp64:
  old 2.8e-7, new 3.4e-7 max relative error; new vs old 3.8e-7 = old vs old (run to run) 3.8e-7.
  `NNUE_L1_DENSE_FORWARD=0` restores the old kernel.
- **Grouped L1 input gradient**: 128-row tiles instead of 64 (0.69 -> 0.55 ms); bit-identical (the
  reduction is the same 32-long dot per output).
- **FT weight merge** (`merge_quant4`): float4 accesses, and each thread walks the 32 buckets of one
  virtual-weight element so the virtual weight is read once. 1.73 -> 1.39 ms (~830 GB/s, DRAM
  bound); bit-identical to the scalar kernel and to the torch chain (`test_merge_quant_vectorized_bit_identical`).

Per step after these changes (nsys, ms): pull gather 7.4, tiled scatter 6.7, fused optimizer 6.3
(~840 GB/s, DRAM bound), prologue 1.65, weight merge 1.47, L1 input gradient ~1.0, epilogue 0.97,
L1 forward 0.84, gradient norm 0.76, layer-stack GEMMs (TF32) ~1.6.

Tried without gain: scatter blocks walking several column slices with the next slice prefetched into
registers (spills at 32 warps, slower at 16), warp-batched scatter tile loads (7.71 -> 7.69 ms), 192-list
tiles with 128-column slices, slice-fastest grid; a 4-way unrolled pull loop (slower), pull R = 16/64,
NS = 3, LPW = 2/8; a float4 optimizer kernel (+0.7%, and not bit-identical). Writing the merged weight
from the optimizer was not done: the FT weights can also be changed through `.data` (input weight
clipping, the optimizer's lookahead path, checkpoint loading), which does not bump the version counter,
so a merge cached by the optimizer could go stale silently.

## tiled-ft-6

Same setup (RTX 3090 at 1500 MHz, real `train.py` + torch.compile, B = 65536, 16 loader workers,
3 epochs x 400 steps, last epoch's it/s, arms interleaved): tiled-ft-5 29.92-29.96 it/s (7 runs);
+ fused lookahead 30.31-30.71; + FT gradient row flags 31.15-31.17; + flagged clip norm 31.44-31.48 (7 runs, 1.05x).

- **Lookahead blend in the update kernel** (`ranger_lite.py`): every 5th step `lookahead_process_step`
  made three passes over every parameter and its lookahead copy (mul_, add_, copy_; ~0.9 ms per step on
  average). On a merge step the fused RangerLite kernels now apply `p * alpha`, then the FMA PyTorch's
  `add_(la, alpha=1 - alpha)` compiles to, to the new parameter and write the lookahead copy.
  Bit-identical (torch.equal on parameters and state against the unfused path, alpha 0.5/0.3/0.8).
  `NNUE_FUSE_LOOKAHEAD=0` disables.
- **FT gradient row flags** (`model/optimizers/grad_row_flags.py`): about half of the FT rows get no
  gradient in a batch, and ~25% (103.4k of 138.1k rows seen in 2000 steps) never do. The tiled scatter
  marks the rows it adds to; the merged-weight backward registers each weight's row slice with its flags;
  the fused update uses +0 for an unflagged row instead of reading the gradient (the gradient there is
  exactly +0), and an element with g == 0 and all-zero optimizer state skips the parameter read and
  all writes (its update rewrites the same bits; the host checks that the scalars keep that true).
  A lookup matches data pointer, shape and version counter (under torch.compile the gradient reaches
  the parameter as a regenerated alias, so object identity does not work) and holds the storage, so an
  in-place change or a copy makes the optimizer read the gradient again. Bit-identical
  (tests/test_fused_optimizer.py). `NNUE_GRAD_ROW_FLAGS=0` disables.
- **Clip norm over the flagged rows** (`grad_row_flags.total_norm`): the gradient-clipping norm read
  every gradient again (0.76 ms); a kernel now sums the squares of the flagged FT rows only (0.36 ms),
  the other gradients use `torch._foreach_norm`. Same value up to fp32 summation order (< 1e-6
  relative in the test). The norms in these runs are 0.006-0.012 against the clip value 2.0, so the
  clip scale is exactly 1.0 either way. `NNUE_FLAGGED_NORM=0` restores `get_total_norm`.

The training loop is no longer purely GPU bound: under nsys the main stream was 89.8% busy, and ~7 of
35 steps waited ~16 ms each for their batch before the first forward kernel.

Tried without gain: the merged FT weight written by the optimizer (a persistent merge buffer whose
validity key is (invalidation epoch, data_ptr and version of every source weight); the fused update of
each weight wrote `rint((p + v) * s) * inv` after the virtual weights were updated; `.data` writers
call `invalidate()`). It was bit-identical (118 of 118 reused merges equal to a fresh merge in real
training) and removed the 1.44 ms merge kernel, but measured 31.34-31.39 it/s against 31.44-31.45
without it (the pull gather took ~1 ms longer in the profile, and the loop now waits on batches), so it
was not kept.
