import torch
from torch import autograd
import numpy as np

_HAS_CUPY_KERNELS = False
try:
    from .fused_ft_kernel import (
        make_fused_double_ft_forward_kernel,
        make_fused_double_ft_backward_kernel,
        make_fused_double_ft_backward_feature_centric_kernel,
    )
    _HAS_CUPY_KERNELS = True
except (ImportError, OSError, RuntimeError):
    pass


def _build_feature_centric_index(white_indices, black_indices, device):
    """Build a combined inverted index for feature-centric grad_weight accumulation.

    Returns unique indices and, for each unique index, the sorted list of
    positions where it occurs (either white or black perspective).
    """
    batch_size, max_active = white_indices.shape

    # Position index for every active slot.
    pos = torch.arange(batch_size, device=device, dtype=torch.int32).unsqueeze(1).expand(-1, max_active)
    pos = pos.reshape(-1)
    wi = white_indices.reshape(-1)
    bi = black_indices.reshape(-1)

    valid_w = wi != -1
    valid_b = bi != -1

    # Concatenate both perspectives; perspective flag 0=white, 1=black.
    all_idx = torch.cat([wi[valid_w], bi[valid_b]])
    all_pos = torch.cat([pos[valid_w], pos[valid_b]])
    all_persp = torch.cat([
        torch.zeros(valid_w.sum(), dtype=torch.int8, device=device),
        torch.ones(valid_b.sum(), dtype=torch.int8, device=device),
    ])

    # Sort by feature index (stable so order within an index is deterministic).
    order = torch.argsort(all_idx, stable=True)
    sorted_idx = all_idx[order]
    sorted_pos = all_pos[order]
    sorted_persp = all_persp[order]

    unique_idx, counts = torch.unique_consecutive(sorted_idx, return_counts=True)
    boundaries = torch.cat([
        torch.zeros(1, device=device, dtype=torch.int32),
        counts.cumsum(0).to(torch.int32),
    ])

    return (
        sorted_pos,
        sorted_persp,
        boundaries,
        unique_idx.to(torch.int32),
    )


class FusedDoubleFtFunction(autograd.Function):
    @staticmethod
    def forward(ctx, us, them, white_indices, black_indices, psqt_indices, weight, bias, max_ft_activation, l1_size):
        ctx.save_for_backward(us, them, white_indices, black_indices, psqt_indices, weight, bias)
        ctx.max_ft_activation = float(max_ft_activation)
        ctx.l1_size = int(l1_size)

        assert l1_size % 2 == 0

        assert us.is_cuda and them.is_cuda
        assert white_indices.is_cuda and black_indices.is_cuda and psqt_indices.is_cuda
        assert weight.is_cuda and bias.is_cuda
        assert us.device == them.device == white_indices.device == black_indices.device == psqt_indices.device == weight.device == bias.device

        assert us.dtype == torch.float32 and them.dtype == torch.float32
        assert white_indices.dtype == torch.int32 and black_indices.dtype == torch.int32
        assert psqt_indices.dtype == torch.int64
        assert weight.dtype == torch.float32 and bias.dtype == torch.float32

        assert white_indices.ndim == 2 and black_indices.ndim == 2
        assert psqt_indices.ndim == 1
        assert len(weight.shape) == 2
        assert len(bias.shape) == 1
        assert weight.shape[1] == bias.shape[0]
        assert white_indices.shape == black_indices.shape
        assert white_indices.shape[0] == psqt_indices.shape[0]

        assert us.is_contiguous() and them.is_contiguous()
        assert white_indices.is_contiguous() and black_indices.is_contiguous() and psqt_indices.is_contiguous()
        assert weight.is_contiguous() and bias.is_contiguous()

        batch_size = white_indices.shape[0]
        max_active_features = white_indices.shape[1]
        l1_half = l1_size // 2

        l0_ = torch.empty(batch_size, l1_size, dtype=torch.float32, device=us.device)
        wpsqt = torch.empty(batch_size, 1, dtype=torch.float32, device=us.device)
        bpsqt = torch.empty(batch_size, 1, dtype=torch.float32, device=us.device)
        clamped_out = torch.empty(batch_size, 4, l1_half, dtype=torch.float32, device=us.device)

        output_size = bias.shape[0]
        kernel = make_fused_double_ft_forward_kernel(max_active_features, l1_size)
        kernel(
            grid=(batch_size,),
            args=(
                us.data_ptr(),
                them.data_ptr(),
                white_indices.data_ptr(),
                black_indices.data_ptr(),
                psqt_indices.data_ptr(),
                weight.data_ptr(),
                bias.data_ptr(),
                np.float32(max_ft_activation),
                l0_.data_ptr(),
                wpsqt.data_ptr(),
                bpsqt.data_ptr(),
                clamped_out.data_ptr(),
                np.int32(output_size),
            )
        )

        ctx.save_for_backward(us, them, white_indices, black_indices, psqt_indices, weight, bias, clamped_out)
        return l0_, wpsqt, bpsqt

    @staticmethod
    def backward(ctx, grad_l0, grad_wpsqt, grad_bpsqt):
        us, them, white_indices, black_indices, psqt_indices, weight, bias, clamped_out = ctx.saved_tensors
        max_ft_activation = ctx.max_ft_activation
        l1_size = ctx.l1_size
        l1_half = l1_size // 2

        grad_l0 = grad_l0.contiguous()
        grad_wpsqt = grad_wpsqt.contiguous()
        grad_bpsqt = grad_bpsqt.contiguous()

        batch_size = white_indices.shape[0]
        output_size = bias.shape[0]
        num_psqt_buckets = output_size - l1_size

        # Compute dense per-position gradients (same formula as the fused kernel).
        clamped_w0 = clamped_out[:, 0, :]
        clamped_w1 = clamped_out[:, 1, :]
        clamped_b0 = clamped_out[:, 2, :]
        clamped_b1 = clamped_out[:, 3, :]

        grad_l0_w = grad_l0[:, :l1_half]
        grad_l0_b = grad_l0[:, l1_half:]

        mask_w0 = (clamped_w0 != 0.0) & (clamped_w0 != max_ft_activation)
        mask_w1 = (clamped_w1 != 0.0) & (clamped_w1 != max_ft_activation)
        mask_b0 = (clamped_b0 != 0.0) & (clamped_b0 != max_ft_activation)
        mask_b1 = (clamped_b1 != 0.0) & (clamped_b1 != max_ft_activation)

        dw0 = torch.where(mask_w0, grad_l0_w * clamped_w1, torch.zeros_like(grad_l0_w))
        dw1 = torch.where(mask_w1, grad_l0_w * clamped_w0, torch.zeros_like(grad_l0_w))
        db0 = torch.where(mask_b0, grad_l0_b * clamped_b1, torch.zeros_like(grad_l0_b))
        db1 = torch.where(mask_b1, grad_l0_b * clamped_b0, torch.zeros_like(grad_l0_b))

        us = us.view(-1)
        them = them.view(-1)
        us_u = us.unsqueeze(1)
        them_u = them.unsqueeze(1)
        g_w0 = us_u * dw0 + them_u * db0
        g_w1 = us_u * dw1 + them_u * db1
        g_b0 = them_u * dw0 + us_u * db0
        g_b1 = them_u * dw1 + us_u * db1

        # grad_bias is a simple global sum across the batch.
        grad_bias = torch.zeros(output_size, dtype=torch.float32, device=us.device)
        grad_bias[:l1_half] = (g_w0 + g_b0).sum(dim=0).view(-1)
        grad_bias[l1_half:l1_size] = (g_w1 + g_b1).sum(dim=0).view(-1)
        grad_bias.index_add_(
            0, l1_size + psqt_indices, grad_wpsqt.view(-1) + grad_bpsqt.view(-1)
        )

        # Build combined inverted index and launch feature-centric grad_weight kernel.
        sorted_pos, sorted_persp, boundaries, unique_idx = _build_feature_centric_index(
            white_indices, black_indices, us.device
        )

        grad_weight = torch.zeros(weight.shape[0], output_size, dtype=torch.float32, device=us.device)

        if unique_idx.numel() > 0:
            kernel = make_fused_double_ft_backward_feature_centric_kernel(l1_size, num_psqt_buckets)
            kernel(
                grid=(unique_idx.numel(),),
                args=(
                    g_w0.data_ptr(),
                    g_w1.data_ptr(),
                    g_b0.data_ptr(),
                    g_b1.data_ptr(),
                    grad_wpsqt.data_ptr(),
                    grad_bpsqt.data_ptr(),
                    psqt_indices.data_ptr(),
                    sorted_pos.data_ptr(),
                    sorted_persp.data_ptr(),
                    boundaries.data_ptr(),
                    unique_idx.data_ptr(),
                    grad_weight.data_ptr(),
                    np.int32(l1_size),
                    np.int32(output_size),
                    np.int32(num_psqt_buckets),
                )
            )

        return None, None, None, None, None, grad_weight, grad_bias, None, None
