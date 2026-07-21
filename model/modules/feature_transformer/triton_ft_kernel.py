"""Triton implementation of the fused double feature-transform kernel.

Replaces the CuPy RawKernel implementation. The fused backend is the fast
CUDA path; CPU/MPS fall back to the torch sparse path.

Design:
- One Triton program per (position, column_tile).
- For typical L1 <= 1024 the column tile covers the full L1/2 half-width, so
  each position is handled by a single program and active indices are loaded
  only once.
- The backward kernel issues coalesced atomicAdd's to grad_weight per active
  index, matching the perf-gpu-ft-atomic-coalesce CUDA strategy.
- PSQT outputs/gradients are handled by masking the first column of tile 0,
  avoiding a separate kernel launch.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def fused_double_ft_forward_triton(
    us_ptr,
    them_ptr,
    white_indices_ptr,
    black_indices_ptr,
    psqt_indices_ptr,
    weight_ptr,
    bias_ptr,
    l0_out_ptr,
    wpsqt_out_ptr,
    bpsqt_out_ptr,
    clamped_out_ptr,
    B,
    K,
    N,
    D,
    L1,
    L1_HALF,
    MAX_FT_ACT,
    BLOCK_COL: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_c = tl.program_id(1)

    col_start = pid_c * BLOCK_COL
    col_offsets = col_start + tl.arange(0, BLOCK_COL)
    col_mask = col_offsets < L1_HALF

    us_val = tl.load(us_ptr + pid_b)
    them_val = tl.load(them_ptr + pid_b)
    p_idx = tl.load(psqt_indices_ptr + pid_b)

    # Start from bias for this column tile.
    w0 = tl.load(bias_ptr + col_offsets, mask=col_mask, other=0.0)
    w1 = tl.load(bias_ptr + L1_HALF + col_offsets, mask=col_mask, other=0.0)
    b0 = w0
    b1 = w1

    w_psqt = 0.0
    b_psqt = 0.0

    # Accumulate active weight rows for this position.
    for k in range(K):
        w_idx = tl.load(white_indices_ptr + pid_b * K + k)
        if w_idx != -1:
            row_ptr = weight_ptr + w_idx * D
            w0 += tl.load(row_ptr + col_offsets, mask=col_mask, other=0.0)
            w1 += tl.load(row_ptr + L1_HALF + col_offsets, mask=col_mask, other=0.0)
            w_psqt += tl.load(weight_ptr + w_idx * D + L1 + p_idx)

        b_idx = tl.load(black_indices_ptr + pid_b * K + k)
        if b_idx != -1:
            row_ptr = weight_ptr + b_idx * D
            b0 += tl.load(row_ptr + col_offsets, mask=col_mask, other=0.0)
            b1 += tl.load(row_ptr + L1_HALF + col_offsets, mask=col_mask, other=0.0)
            b_psqt += tl.load(weight_ptr + b_idx * D + L1 + p_idx)

    # Double feature transform + clamp.
    l0_w0 = us_val * w0 + them_val * b0
    l0_w1 = us_val * w1 + them_val * b1
    l0_b0 = us_val * b0 + them_val * w0
    l0_b1 = us_val * b1 + them_val * w1

    l0_w0 = tl.maximum(0.0, tl.minimum(l0_w0, MAX_FT_ACT))
    l0_w1 = tl.maximum(0.0, tl.minimum(l0_w1, MAX_FT_ACT))
    l0_b0 = tl.maximum(0.0, tl.minimum(l0_b0, MAX_FT_ACT))
    l0_b1 = tl.maximum(0.0, tl.minimum(l0_b1, MAX_FT_ACT))

    # Write L1 output.
    l0_base = pid_b * L1
    tl.store(l0_out_ptr + l0_base + col_offsets, l0_w0 * l0_w1, mask=col_mask)
    tl.store(l0_out_ptr + l0_base + L1_HALF + col_offsets, l0_b0 * l0_b1, mask=col_mask)

    # Cache clamped activations for backward.
    clamp_base = pid_b * 4 * L1_HALF
    tl.store(clamped_out_ptr + clamp_base + 0 * L1_HALF + col_offsets, l0_w0, mask=col_mask)
    tl.store(clamped_out_ptr + clamp_base + 1 * L1_HALF + col_offsets, l0_w1, mask=col_mask)
    tl.store(clamped_out_ptr + clamp_base + 2 * L1_HALF + col_offsets, l0_b0, mask=col_mask)
    tl.store(clamped_out_ptr + clamp_base + 3 * L1_HALF + col_offsets, l0_b1, mask=col_mask)

    # Only the first column of tile 0 writes PSQT values (no contention).
    psqt_mask = col_offsets == 0
    tl.store(wpsqt_out_ptr + pid_b, w_psqt, mask=psqt_mask)
    tl.store(bpsqt_out_ptr + pid_b, b_psqt, mask=psqt_mask)


@triton.jit
def fused_double_ft_backward_triton(
    us_ptr,
    them_ptr,
    white_indices_ptr,
    black_indices_ptr,
    psqt_indices_ptr,
    clamped_out_ptr,
    grad_l0_ptr,
    grad_wpsqt_ptr,
    grad_bpsqt_ptr,
    grad_weight_ptr,
    grad_bias_ptr,
    B,
    K,
    N,
    D,
    L1,
    L1_HALF,
    MAX_FT_ACT,
    BLOCK_COL: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_c = tl.program_id(1)

    col_start = pid_c * BLOCK_COL
    col_offsets = col_start + tl.arange(0, BLOCK_COL)
    col_mask = col_offsets < L1_HALF

    us_val = tl.load(us_ptr + pid_b)
    them_val = tl.load(them_ptr + pid_b)
    p_idx = tl.load(psqt_indices_ptr + pid_b)
    gw_psqt = tl.load(grad_wpsqt_ptr + pid_b)
    gb_psqt = tl.load(grad_bpsqt_ptr + pid_b)

    # Load cached clamped activations and upstream gradients.
    clamp_base = pid_b * 4 * L1_HALF
    clamped_w0 = tl.load(clamped_out_ptr + clamp_base + 0 * L1_HALF + col_offsets, mask=col_mask)
    clamped_w1 = tl.load(clamped_out_ptr + clamp_base + 1 * L1_HALF + col_offsets, mask=col_mask)
    clamped_b0 = tl.load(clamped_out_ptr + clamp_base + 2 * L1_HALF + col_offsets, mask=col_mask)
    clamped_b1 = tl.load(clamped_out_ptr + clamp_base + 3 * L1_HALF + col_offsets, mask=col_mask)

    gl0_base = pid_b * L1
    gl0_i = tl.load(grad_l0_ptr + gl0_base + col_offsets, mask=col_mask)
    gl0_i_h = tl.load(grad_l0_ptr + gl0_base + L1_HALF + col_offsets, mask=col_mask)

    # Recompute the ReLU-clamp derivative.
    dw0 = tl.where((clamped_w0 == 0.0) | (clamped_w0 == MAX_FT_ACT), 0.0, gl0_i * clamped_w1)
    dw1 = tl.where((clamped_w1 == 0.0) | (clamped_w1 == MAX_FT_ACT), 0.0, gl0_i * clamped_w0)
    db0 = tl.where((clamped_b0 == 0.0) | (clamped_b0 == MAX_FT_ACT), 0.0, gl0_i_h * clamped_b1)
    db1 = tl.where((clamped_b1 == 0.0) | (clamped_b1 == MAX_FT_ACT), 0.0, gl0_i_h * clamped_b0)

    # Per-perspective gradients.
    g_w0 = us_val * dw0 + them_val * db0
    g_w1 = us_val * dw1 + them_val * db1
    g_b0 = them_val * dw0 + us_val * db0
    g_b1 = them_val * dw1 + us_val * db1

    # Accumulate bias for this tile.
    tl.atomic_add(grad_bias_ptr + col_offsets, g_w0 + g_b0, mask=col_mask)
    tl.atomic_add(grad_bias_ptr + L1_HALF + col_offsets, g_w1 + g_b1, mask=col_mask)

    # PSQT bias update only from tile 0.
    psqt_mask = col_offsets == 0
    tl.atomic_add(grad_bias_ptr + L1 + p_idx, gw_psqt + gb_psqt, mask=psqt_mask)

    # Scatter coalesced updates to grad_weight for each active index.
    for k in range(K):
        w_idx = tl.load(white_indices_ptr + pid_b * K + k)
        if w_idx != -1:
            gw_row = grad_weight_ptr + w_idx * D
            tl.atomic_add(gw_row + col_offsets, g_w0, mask=col_mask)
            tl.atomic_add(gw_row + L1_HALF + col_offsets, g_w1, mask=col_mask)
            tl.atomic_add(gw_row + L1 + p_idx, gw_psqt, mask=psqt_mask)

        b_idx = tl.load(black_indices_ptr + pid_b * K + k)
        if b_idx != -1:
            gw_row = grad_weight_ptr + b_idx * D
            tl.atomic_add(gw_row + col_offsets, g_b0, mask=col_mask)
            tl.atomic_add(gw_row + L1_HALF + col_offsets, g_b1, mask=col_mask)
            tl.atomic_add(gw_row + L1 + p_idx, gb_psqt, mask=psqt_mask)


# ---------------------------------------------------------------------------
# Python wrappers with kernel caching
# ---------------------------------------------------------------------------

_fused_double_ft_forward_cache = dict()
_fused_double_ft_backward_cache = dict()


def _get_block_col(l1_half: int) -> int:
    """Power-of-two block size covering the L1/2 columns.

    A single tile is used whenever L1/2 <= 512, which avoids reloading the
    active indices and keeps atomics coalesced over the full half-width.
    Larger L1 is tiled into 512-column chunks.
    """
    return max(64, min(512, 1 << (l1_half - 1).bit_length()))


def fused_double_ft_forward(
    us,
    them,
    white_indices,
    black_indices,
    psqt_indices,
    weight,
    bias,
    max_ft_activation,
    l1_size,
):
    batch_size, max_active = white_indices.shape
    num_inputs, output_size = weight.shape
    l1_half = l1_size // 2
    device = us.device

    l0_ = torch.empty(batch_size, l1_size, dtype=torch.float32, device=device)
    wpsqt = torch.empty(batch_size, 1, dtype=torch.float32, device=device)
    bpsqt = torch.empty(batch_size, 1, dtype=torch.float32, device=device)
    clamped_out = torch.empty(batch_size, 4, l1_half, dtype=torch.float32, device=device)

    block_col = _get_block_col(l1_half)
    num_col_tiles = (l1_half + block_col - 1) // block_col

    key = (max_active, l1_size, block_col)
    if key not in _fused_double_ft_forward_cache:
        _fused_double_ft_forward_cache[key] = fused_double_ft_forward_triton

    kernel = _fused_double_ft_forward_cache[key]
    kernel[(batch_size, num_col_tiles)](
        us, them,
        white_indices, black_indices, psqt_indices,
        weight, bias,
        l0_, wpsqt, bpsqt, clamped_out,
        batch_size, max_active, num_inputs, output_size,
        l1_size, l1_half, max_ft_activation,
        BLOCK_COL=block_col,
    )
    return l0_, wpsqt, bpsqt, clamped_out


def fused_double_ft_backward(
    us,
    them,
    white_indices,
    black_indices,
    psqt_indices,
    clamped_out,
    grad_l0,
    grad_wpsqt,
    grad_bpsqt,
    weight,
    bias,
    max_ft_activation,
    l1_size,
):
    batch_size, max_active = white_indices.shape
    num_inputs, output_size = weight.shape
    l1_half = l1_size // 2
    device = us.device

    grad_weight = torch.zeros(num_inputs, output_size, dtype=torch.float32, device=device)
    grad_bias = torch.zeros(output_size, dtype=torch.float32, device=device)

    block_col = _get_block_col(l1_half)
    num_col_tiles = (l1_half + block_col - 1) // block_col

    key = (max_active, l1_size, block_col)
    if key not in _fused_double_ft_backward_cache:
        _fused_double_ft_backward_cache[key] = fused_double_ft_backward_triton

    kernel = _fused_double_ft_backward_cache[key]
    kernel[(batch_size, num_col_tiles)](
        us.view(-1), them.view(-1),
        white_indices, black_indices, psqt_indices,
        clamped_out, grad_l0,
        grad_wpsqt.view(-1), grad_bpsqt.view(-1),
        grad_weight, grad_bias,
        batch_size, max_active, num_inputs, output_size,
        l1_size, l1_half, max_ft_activation,
        BLOCK_COL=block_col,
    )
    return grad_weight, grad_bias
