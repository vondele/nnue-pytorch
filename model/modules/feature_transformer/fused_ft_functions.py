import numpy as np
import torch
from torch import autograd

_HAS_CUPY_KERNELS = False
try:
    from .aggregated_ft_kernel import (
        aggregated_ft_backward,
        aggregated_ft_backward_u8,
    )
    from .fused_ft_kernel import (
        BACKWARD_TILE_SIZE,
        make_ft_quantize_to_int16_kernel,
        make_fused_double_ft_backward_kernel,
        make_fused_double_ft_backward_kernel_u8,
        make_fused_double_ft_forward_kernel,
        make_fused_double_ft_forward_kernel_int16,
    )
    _HAS_CUPY_KERNELS = True
except (ImportError, OSError, RuntimeError):
    BACKWARD_TILE_SIZE = 1


class FusedDoubleFtFunction(autograd.Function):
    @staticmethod
    def forward(ctx, us, them, white_indices, black_indices, weight, bias, max_ft_activation, l1_size):
        ctx.max_ft_activation = float(max_ft_activation)
        ctx.l1_size = int(l1_size)

        assert l1_size % 2 == 0

        assert us.is_cuda and them.is_cuda
        assert white_indices.is_cuda and black_indices.is_cuda
        assert weight.is_cuda and bias.is_cuda
        assert us.device == them.device == white_indices.device == black_indices.device == weight.device == bias.device

        assert us.dtype == torch.float32 and them.dtype == torch.float32
        assert white_indices.dtype == torch.int32 and black_indices.dtype == torch.int32
        assert weight.dtype == torch.float32 and bias.dtype == torch.float32

        assert white_indices.ndim == 2 and black_indices.ndim == 2
        assert len(weight.shape) == 2
        assert len(bias.shape) == 1
        assert weight.shape[1] == bias.shape[0]
        assert white_indices.shape == black_indices.shape

        assert us.is_contiguous() and them.is_contiguous()
        assert white_indices.is_contiguous() and black_indices.is_contiguous()
        assert weight.is_contiguous() and bias.is_contiguous()

        batch_size = white_indices.shape[0]
        max_active_features = white_indices.shape[1]
        l1_half = l1_size // 2

        l0_ = torch.empty(batch_size, l1_size, dtype=torch.float32, device=us.device)
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
                weight.data_ptr(),
                bias.data_ptr(),
                np.float32(max_ft_activation),
                l0_.data_ptr(),
                clamped_out.data_ptr(),
                np.int32(output_size),
            )
        )

        ctx.save_for_backward(us, them, white_indices, black_indices, weight, bias, clamped_out)
        return l0_

    @staticmethod
    def backward(ctx, grad_l0):
        us, them, white_indices, black_indices, weight, bias, clamped_out = ctx.saved_tensors
        max_ft_activation = ctx.max_ft_activation
        l1_size = ctx.l1_size

        grad_l0 = grad_l0.contiguous()

        batch_size = white_indices.shape[0]
        max_active_features = white_indices.shape[1]
        output_size = bias.shape[0]

        grad_weight = torch.zeros(weight.shape[0], output_size, dtype=torch.float32, device=us.device)
        grad_bias = torch.zeros(output_size, dtype=torch.float32, device=us.device)

        # Aggregation pays for its feature-union pass on large master-net batches.
        # Keep direct scatter for unsupported widths/devices and small batches.
        if (512 <= l1_size <= 4096 and l1_size % 128 == 0 and batch_size >= 1024
                and 0 < max_active_features <= 288
                and torch.version.hip is None
                and torch.cuda.get_device_capability(us.device) == (9, 0)):
            aggregated_ft_backward(
                us, them, white_indices, black_indices, grad_l0, clamped_out,
                grad_weight, grad_bias, max_ft_activation,
            )
            return None, None, None, None, grad_weight, grad_bias, None, None

        kernel = make_fused_double_ft_backward_kernel(max_active_features, l1_size)
        grid_size = (batch_size + BACKWARD_TILE_SIZE - 1) // BACKWARD_TILE_SIZE
        kernel(
            grid=(grid_size,),
            args=(
                us.data_ptr(),
                them.data_ptr(),
                white_indices.data_ptr(),
                black_indices.data_ptr(),
                weight.data_ptr(),
                bias.data_ptr(),
                np.float32(max_ft_activation),
                grad_l0.data_ptr(),
                clamped_out.data_ptr(),
                grad_weight.data_ptr(),
                grad_bias.data_ptr(),
                np.int32(batch_size),
                np.int32(output_size),
            )
        )

        return None, None, None, None, grad_weight, grad_bias, None, None


_INT16_FLAG_CHECK_INTERVAL = 256
_OVERFLOW_FLAGS = {}
_INT16_STEPS = 0


def int16_ft_available(weight, bias, max_ft_activation, l1_size) -> bool:
    """Runtime eligibility for the int16 FT kernels.

    Requires the fake-quantized weight/bias on the k/256 grid (the caller
    guarantees this by only requesting the backend under fake quantization
    with the default 256 scales) and a clamp level that is an exact uint8
    quantization step.
    """
    if not _HAS_CUPY_KERNELS:
        return False
    if torch.version.hip is not None:
        return False
    if l1_size % 2 != 0:
        return False
    level = max_ft_activation * 256.0
    if level != int(level) or not (0 < int(level) <= 255):
        return False
    return (
        weight.is_cuda
        and weight.dtype == torch.float32
        and weight.is_contiguous()
        and bias.is_cuda
        and bias.dtype == torch.float32
        and bias.is_contiguous()
    )


class FusedDoubleFtIntFunction(autograd.Function):
    """int16-table variant of FusedDoubleFtFunction.

    The fake-quantized weight and bias live on the k/256 grid, so gathering
    an int16 table and accumulating in int32 computes exactly the same values
    as the fp32 path (whose partial sums stay on the grid below 2^24), and
    clamped_out stores exact uint8 quantization levels. The fp32 merged
    weight is not saved for backward, only the int16 table.
    """

    @staticmethod
    def forward(ctx, us, them, white_indices, black_indices, weight, bias, max_ft_activation, l1_size):
        ctx.max_ft_activation = float(max_ft_activation)
        ctx.max_ft_activation_k = int(round(max_ft_activation * 256.0))
        ctx.l1_size = int(l1_size)

        assert l1_size % 2 == 0

        assert us.is_cuda and them.is_cuda
        assert white_indices.is_cuda and black_indices.is_cuda
        assert weight.is_cuda and bias.is_cuda
        assert us.device == them.device == white_indices.device == black_indices.device == weight.device == bias.device

        assert us.dtype == torch.float32 and them.dtype == torch.float32
        assert white_indices.dtype == torch.int32 and black_indices.dtype == torch.int32
        assert weight.dtype == torch.float32 and bias.dtype == torch.float32

        assert white_indices.ndim == 2 and black_indices.ndim == 2
        assert len(weight.shape) == 2
        assert len(bias.shape) == 1
        assert weight.shape[1] == bias.shape[0]
        assert white_indices.shape == black_indices.shape

        assert us.is_contiguous() and them.is_contiguous()
        assert white_indices.is_contiguous() and black_indices.is_contiguous()

        batch_size = white_indices.shape[0]
        max_active_features = white_indices.shape[1]
        l1_half = l1_size // 2
        rows, width = weight.shape

        global _INT16_STEPS
        _INT16_STEPS += 1

        flag = _OVERFLOW_FLAGS.get(us.device.index)
        if flag is None:
            flag = torch.zeros(1, dtype=torch.int32, device=us.device)
            _OVERFLOW_FLAGS[us.device.index] = flag
        elif _INT16_STEPS % _INT16_FLAG_CHECK_INTERVAL == 0:
            # Deferred (and therefore sync-free in the steady state) check of
            # the saturation/off-grid flag accumulated over past steps.
            if flag.item() != 0:
                raise RuntimeError(
                    "FT weight/bias left the int16 k/256 grid; falling back is "
                    "not safe mid-run. Use NNUE_FT_INT16=0 to disable the path."
                )
            flag.zero_()

        weight_i16 = torch.empty(rows, width, dtype=torch.int16, device=us.device)
        bias_i16 = torch.empty(width, dtype=torch.int16, device=us.device)
        quantize = make_ft_quantize_to_int16_kernel()
        n_weight = rows * width
        # Four elements per thread plus scalar tail elements and the bias.
        total = (n_weight + 3) // 4 + n_weight % 4 + width
        quantize(
            grid=((total + 255) // 256,),
            args=(
                weight.data_ptr(),
                bias.data_ptr(),
                weight_i16.data_ptr(),
                bias_i16.data_ptr(),
                flag.data_ptr(),
                np.int64(n_weight),
                np.int32(width),
            ),
        )

        l0_ = torch.empty(batch_size, l1_size, dtype=torch.float32, device=us.device)
        clamped_out = torch.empty(batch_size, 4, l1_half, dtype=torch.uint8, device=us.device)

        kernel = make_fused_double_ft_forward_kernel_int16(max_active_features, l1_size)
        kernel(
            grid=(batch_size,),
            args=(
                us.data_ptr(),
                them.data_ptr(),
                white_indices.data_ptr(),
                black_indices.data_ptr(),
                weight_i16.data_ptr(),
                bias_i16.data_ptr(),
                np.float32(max_ft_activation),
                np.int32(ctx.max_ft_activation_k),
                l0_.data_ptr(),
                clamped_out.data_ptr(),
                np.int32(width),
            )
        )

        ctx.save_for_backward(us, them, white_indices, black_indices, weight_i16, bias_i16, clamped_out)
        return l0_

    @staticmethod
    def backward(ctx, grad_l0):
        us, them, white_indices, black_indices, weight_i16, bias_i16, clamped_out = ctx.saved_tensors
        max_ft_activation = ctx.max_ft_activation
        max_ft_activation_k = ctx.max_ft_activation_k
        l1_size = ctx.l1_size

        grad_l0 = grad_l0.contiguous()

        batch_size = white_indices.shape[0]
        max_active_features = white_indices.shape[1]
        rows, width = weight_i16.shape

        grad_weight = torch.zeros(rows, width, dtype=torch.float32, device=us.device)
        grad_bias = torch.zeros(width, dtype=torch.float32, device=us.device)

        # Aggregation pays for its feature-union pass on large master-net batches.
        # Keep direct scatter for unsupported widths/devices and small batches.
        if (512 <= l1_size <= 4096 and l1_size % 128 == 0 and batch_size >= 1024
                and 0 < max_active_features <= 288
                and torch.version.hip is None
                and torch.cuda.get_device_capability(us.device) == (9, 0)):
            aggregated_ft_backward_u8(
                us, them, white_indices, black_indices, grad_l0, clamped_out,
                grad_weight, grad_bias, max_ft_activation_k,
            )
            return None, None, None, None, grad_weight, grad_bias, None, None

        kernel = make_fused_double_ft_backward_kernel_u8(max_active_features, l1_size)
        grid_size = (batch_size + BACKWARD_TILE_SIZE - 1) // BACKWARD_TILE_SIZE
        kernel(
            grid=(grid_size,),
            args=(
                us.data_ptr(),
                them.data_ptr(),
                white_indices.data_ptr(),
                black_indices.data_ptr(),
                np.float32(max_ft_activation),
                np.int32(max_ft_activation_k),
                grad_l0.data_ptr(),
                clamped_out.data_ptr(),
                grad_weight.data_ptr(),
                grad_bias.data_ptr(),
                np.int32(batch_size),
                np.int32(width),
            )
        )

        return None, None, None, None, grad_weight, grad_bias, None, None
