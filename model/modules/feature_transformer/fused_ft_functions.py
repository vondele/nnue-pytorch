import importlib.util
import torch
from torch import autograd

_HAS_TRITON_KERNELS = importlib.util.find_spec("triton") is not None
if _HAS_TRITON_KERNELS:
    try:
        from .triton_ft_kernel import (
            fused_double_ft_forward,
            fused_double_ft_backward,
        )
    except (ImportError, OSError, RuntimeError):
        _HAS_TRITON_KERNELS = False


class FusedDoubleFtFunction(autograd.Function):
    @staticmethod
    def forward(ctx, us, them, white_indices, black_indices, psqt_indices, weight, bias, max_ft_activation, l1_size):
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

        l0_, wpsqt, bpsqt, clamped_out = fused_double_ft_forward(
            us, them,
            white_indices, black_indices, psqt_indices,
            weight, bias,
            max_ft_activation, l1_size,
        )

        ctx.save_for_backward(us, them, white_indices, black_indices, psqt_indices, weight, bias, clamped_out)
        return l0_, wpsqt, bpsqt

    @staticmethod
    def backward(ctx, grad_l0, grad_wpsqt, grad_bpsqt):
        us, them, white_indices, black_indices, psqt_indices, weight, bias, clamped_out = ctx.saved_tensors
        max_ft_activation = ctx.max_ft_activation
        l1_size = ctx.l1_size

        grad_l0 = grad_l0.contiguous()
        grad_wpsqt = grad_wpsqt.contiguous()
        grad_bpsqt = grad_bpsqt.contiguous()

        grad_weight, grad_bias = fused_double_ft_backward(
            us, them,
            white_indices, black_indices, psqt_indices,
            clamped_out, grad_l0,
            grad_wpsqt, grad_bpsqt,
            weight, bias,
            max_ft_activation, l1_size,
        )

        return None, None, None, None, None, grad_weight, grad_bias, None, None
