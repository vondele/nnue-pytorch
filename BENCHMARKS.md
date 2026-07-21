# GPU Performance Benchmarks — Feature-Transform Backends

Focus configs:
- **RTX 2070 SUPER**: L1 = 1024, batch = 65536, epoch-size = 10M
- **4×GH100/GH200**: global batch = 131072 (per-GPU batch = 32768)

## Summary

Three atomic-handling strategies were explored on top of the `perf-gpu-ft-memory`
baseline (merged L1/PSQT passes + cached clamped activations). Results are
**hardware-dependent**:

| strategy | branch | RTX 2070 fwd+bwd | RTX 2070 epoch | 4×GH100 128k it/s |
|---|---|---|---|---|
| Baseline (memory branch) | `perf-gpu-ft-memory` | 181.0 ms | 42 s | ~61.3 |
| A. Coalesce atomics per index | `perf-gpu-ft-atomic-coalesce` | 172.5 ms | 41 s | ~61.4 |
| B. Shared-memory bias tile | `perf-gpu-ft-atomic-privatize` | 176.6 ms | 42 s | ~61.7 |
| C. Feature-centric accumulation | `perf-gpu-ft-atomic-feature-centric` | **138.0 ms** | **37 s** | **~38.6** |
| D. Triton coalesced atomic | `perf-gpu-ft-triton` | *pending* | *pending* | *pending* |

**Key insight:** Option C wins on the bandwidth-starved RTX 2070 but is a large
regression on GH100, where fast atomics and a large L2 make the coalesced/
shared-memory strategies better. The Triton port therefore targets the
**coalesced-atomic strategy** (Option A) for production.

### Validation status

- `perf-gpu-ft-triton` was implemented on a box whose GPU later dropped offline
(`nvidia-smi: Failed to initialize NVML`). CPU tests pass; **GPU correctness and
performance are pending** on the production GH100/GH200 environment.

## RTX 2070 benchmarks

Hardware: NVIDIA GeForce RTX 2070 SUPER 8 GB.

### Microbenchmark (`tests/ft_kernel_microbench.py`)

Isolated feature-transform forward + backward, synthetic data, padding ratio 0.8.
A fix was applied to bucket the random piece-count variable to the valid PSQT
range `[0, num_psqt_buckets)` so all variants are compared on the same valid
inputs.

| variant | forward | fwd + backward | effective BW |
|---|---|---|---|
| baseline (memory branch) | 47.9 ms | 181.0 ms | 388 GB/s |
| A. coalesce per index | 47.8 ms | 172.5 ms | 407 GB/s |
| B. shared-memory bias tile | 47.8 ms | 176.6 ms | 398 GB/s |
| C. feature-centric | 47.7 ms | **138.0 ms** | **509 GB/s** |

### End-to-end training

Command (L1=1024, batch=64k, 1 epoch):

```bash
python train.py /workspace/data/official-stockfish/master-binpacks/fishpack32.binpack \
  --features=Full_Threats+HalfKAv2_hm^ --l1=1024 --l2=32 \
  --batch-size=65536 --epoch-size=10000000 --max_epochs=1 \
  --num-workers=8 --threads=1 --gpus=0 --accelerator=cuda \
  --lr=0.9e-3 --one-cycle-steps=320 \
  --early_fen_skipping=-1 --random_fen_skipping=0 --simple_eval_skipping=-1 \
  --soft_early_fen_skipping=55 --no-wld_filtered \
  --start_lambda=1.0 --end_lambda=1.0 \
  --in-offset=199.6164751319224 --in-scaling=288.6467157482683 \
  --out-offset=208.55835423042265 --out-scaling=289.58605293609645 \
  --pc-y0=-0.021286239983835456 --pc-y1=0.4630376436225087 \
  --pc-y2=0.8119190849177433 --pc-y3=0.8432669372128858 --pc-y4=0.8307417849554233 \
  --ply_x1=15 --ply_y1=0.582440530168931 \
  --ply_x2=25 --ply_y2=0.5027907706483704 \
  --ply_x3=35 --ply_y3=0.5456020825505818 \
  --ply_x4=45 --ply_y4=0.4839664887891037 \
  --pow-exp=2.2458593774318936 --qp-asymmetry=0.16123193074123665 \
  --w1=3.4992759013258348 --w2=0.24990297747029686 \
  --validation-size=0 --default_root_dir=/workspace/nnue-pytorch/tmp/bench_ft_$VARIANT
```

| variant | epoch 1 | final loss |
|---|---|---|
| baseline (memory branch) | 42 s | 0.02651 |
| A. coalesce per index | 41 s | 0.02638 |
| B. shared-memory bias tile | 42 s | 0.02628 |
| C. feature-centric | **37 s** | 0.02620 |

## GH100/GH200 benchmarks

Production run on 4×GH100/GH200, 1 node, NVLink, global batch 131072:

| branch | it/s rep 1 | it/s rep 2 |
|---|---|---|
| master | 56.78 | 56.68 |
| perf-gpu-ft-memory | 61.30 | 61.64 |
| perf-gpu-ft-atomic-coalesce | 61.40 | 61.74 |
| perf-gpu-ft-atomic-privatize | 61.99 | 61.68 |
| **perf-gpu-ft-atomic-feature-centric** | **38.64** | **38.68** |

The feature-centric strategy is **~37% slower than master** here. The likely
causes are the per-batch inverted-index build (`torch.argsort` +
`torch.unique_consecutive` on ~10M occurrences per GPU) and poor gather locality
in the reduction kernel, which outweigh the avoided atomics on a card with fast
atomic throughput and a large L2.

### GH100/GH200 benchmark command

```bash
python train.py <dataset> \
  --features=Full_Threats+HalfKAv2_hm^ --l1=1024 --l2=32 \
  --batch-size=131072 --epoch-size=... --max_epochs=... \
  --num-workers=8 --threads=1 --gpus=0,1,2,3 --accelerator=cuda \
  --strategy=ddp \
  ...
```

## Implementation notes

### Option A — `perf-gpu-ft-atomic-coalesce`
The backward kernel computes the full per-column gradient contribution for an
active index first, then issues the two `atomicAdd`s for that index's row
coalesced across the warp.

### Option B — `perf-gpu-ft-atomic-privatize`
Each CUDA block processes a tile of 4 positions and accumulates `grad_bias` in
shared memory before flushing once. Marginal on RTX 2070, roughly matches
coalesce on GH100.

### Option C — `perf-gpu-ft-atomic-feature-centric`
- Compute dense per-position gradients `g_w0`, `g_w1`, `g_b0`, `g_b1` from the
cached clamped activations and `grad_l0`.
- Build a combined inverted index of all active white/black features
(`torch.argsort` + `torch.unique_consecutive`).
- Launch a CuPy RawKernel with one block per unique active feature. Each block
iterates over its occurrences and writes the summed row directly to
`grad_weight` — no global `atomicAdd` for the L1 part.
- PSQT gradients are accumulated into `num_psqt_buckets` per-block bins and
written once per feature.
- `grad_bias` is computed with vectorized PyTorch reductions.

### Option D — `perf-gpu-ft-triton`
A Triton port targeting the coalesced-atomic strategy:

- `model/modules/feature_transformer/triton_ft_kernel.py` — Triton forward and
backward kernels.
- Forward: one program per `(position, column_tile)`. Gathers weight rows for
active white/black indices, computes the double transform + clamp, stores
`l0_out` and `clamped_out`. PSQT output is written by the first column of tile 0.
- Backward: one program per `(position, column_tile)`. Reads cached clamped
activations and `grad_l0`, recomputes the clamped derivative, then issues
coalesced `atomicAdd`s to `grad_weight` and `grad_bias` for each active index.
- Column block size is `max(64, min(512, next_power_of_2(L1/2)))`, so typical
L1 ≤ 1024 uses a single tile per position.
- The legacy CuPy kernels (`fused_ft_kernel.py`, `sparse_linear_kernel.py`) were
removed. `SparseLinearFunction` now uses the PyTorch `embedding_bag` fallback.

### Files changed (branches)

- `model/modules/feature_transformer/triton_ft_kernel.py` — Triton kernels.
- `model/modules/feature_transformer/fused_ft_functions.py` — autograd wrapper
for Triton.
- `model/modules/feature_transformer/double_ft_functions.py` — backend
selection.
- `model/modules/feature_transformer/sparse_linear_functions.py` — torch-only
fallback.
- `scripts/easy_train.py` — validate Triton instead of CuPy.
- `requirements.txt` — add `triton>=3.0`.
- `tests/test_fused_double_ft.py` — update skip condition.
- `tests/ft_kernel_microbench.py` — PSQT bucketing fix.

## Continuation checklist (for a fresh box)

1. Install dependencies: `pip install -r requirements.txt` (Triton is now
required for the fused backend).
2. Build the native data loader: `./setup_script.sh` from repo root.
3. Run correctness tests:
   ```bash
   export MPLCONFIGDIR=/workspace/nnue-pytorch/.mplconfig
   mkdir -p "$MPLCONFIGDIR"
   pytest tests/test_fused_double_ft.py tests/test_feature_transformer.py -v
   ```
4. Run the isolated microbenchmark:
   ```bash
   python tests/ft_kernel_microbench.py --batch-size 65536 --l1 1024 --repeats 50 --padding-ratio 0.8 --fake-quantize
   ```
5. Run 1-epoch training on the target GPU and batch size.

## Future work

- Validate `perf-gpu-ft-triton` on 4×GH100/GH200 with global batch 64k and 128k.
- If the Triton coalesced backward is slower than CuPy on GH100, tune the
column block size or try warp-specialized atomics.
- Keep `perf-gpu-ft-atomic-feature-centric` only as an RTX 2070-specific
experiment; do not merge it to production.
