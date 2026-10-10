"""FT autocast/auto compacts only forward weights and keeps FP32 gradient routing."""
import pytest
import torch

from model.modules.feature_transformer.double_ft_functions import (
    double_feature_transform,
)
from model.modules.feature_transformer.fused_ft_functions import _HAS_CUPY_KERNELS


@pytest.mark.skipif(not torch.cuda.is_available() or not _HAS_CUPY_KERNELS,
                    reason="CUDA and CuPy required")
@pytest.mark.parametrize("width,batch", [(128, 17), (1024, 1025), (1152, 1025), (1280, 17)])
def test_ft_half_buffer_matches_explicit_cast(width, batch):
    torch.manual_seed(314)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        torch.cuda._sleep(2_000_000)
        us = torch.randint(0, 2, (batch, 1), device="cuda").float()
        them = 1 - us
        white = torch.randint(0, 64, (batch, 17), dtype=torch.int32, device="cuda")
        black = torch.randint(0, 64, (batch, 17), dtype=torch.int32, device="cuda")
        white[:, 9:] = -1
        black[:, 13:] = -1
        weight = (torch.randn(64, width, device="cuda") / 16).requires_grad_()
        bias = torch.full((width,), 0.5, device="cuda", requires_grad=True)
        dy = torch.randn(batch, width, device="cuda") * 1e-5
        rounded = weight.detach().half().float().requires_grad_()
        reference = double_feature_transform(us, them, white, black, rounded, bias,
                                             255 / 256, width, "fused")
        with torch.autocast("cuda", dtype=torch.float16):
            actual = double_feature_transform(us, them, white, black, weight, bias,
                                              255 / 256, width, "fused")
        actual_grads = torch.autograd.grad(actual, (weight, bias), dy)
        expected_grads = torch.autograd.grad(reference, (rounded, bias), dy)
        assert actual.dtype == torch.float32
        torch.testing.assert_close(actual, reference, rtol=0, atol=0)
        for got, want in zip(actual_grads, expected_grads):
            assert got.dtype == torch.float32
            torch.testing.assert_close(got, want, rtol=4e-4, atol=4e-8)
    stream.synchronize()


@pytest.mark.skipif(not torch.cuda.is_available() or not _HAS_CUPY_KERNELS,
                    reason="CUDA and CuPy required")
@pytest.mark.parametrize("width,batch", [(1024, 1025), (1152, 1025)])
def test_ft_auto_compact_weights(width, batch):
    """The automatic H100 gate stores forward weights in half without autocast."""
    if torch.cuda.get_device_capability() != (9, 0):
        pytest.skip("H100 specialization")
    torch.manual_seed(271)
    us = torch.randint(0, 2, (batch, 1), device="cuda").float()
    them = 1 - us
    white = torch.randint(0, 64, (batch, 17), dtype=torch.int32, device="cuda")
    black = torch.randint(0, 64, (batch, 17), dtype=torch.int32, device="cuda")
    white[:, 9:] = -1
    black[:, 13:] = -1
    weight = (torch.randn(64, width, device="cuda") / 16).requires_grad_()
    bias = torch.full((width,), 0.5, device="cuda", requires_grad=True)
    dy = torch.randn(batch, width, device="cuda") * 1e-5
    rounded = weight.detach().half().float().requires_grad_()
    reference = double_feature_transform(us, them, white, black, rounded, bias,
                                         255 / 256, width, "fused")
    actual = double_feature_transform(us, them, white, black, weight, bias,
                                      255 / 256, width, "fused")
    actual_grads = torch.autograd.grad(actual, (weight, bias), dy)
    expected_grads = torch.autograd.grad(reference, (rounded, bias), dy)
    torch.testing.assert_close(actual, reference, rtol=0, atol=0)
    for got, want in zip(actual_grads, expected_grads):
        assert got.dtype == torch.float32
        torch.testing.assert_close(got, want, rtol=4e-4, atol=4e-8)


@pytest.mark.skipif(not torch.cuda.is_available() or not _HAS_CUPY_KERNELS,
                    reason="CUDA and CuPy required")
def test_ft_small_batch_stays_fp32():
    """Below the automatic batch threshold the forward stays exact FP32."""
    if torch.cuda.get_device_capability() != (9, 0):
        pytest.skip("H100 specialization")
    torch.manual_seed(272)
    batch, width = 17, 1024
    us = torch.randint(0, 2, (batch, 1), device="cuda").float()
    them = 1 - us
    white = torch.randint(0, 64, (batch, 17), dtype=torch.int32, device="cuda")
    black = torch.randint(0, 64, (batch, 17), dtype=torch.int32, device="cuda")
    white[:, 9:] = -1
    black[:, 13:] = -1
    weight = (torch.randn(64, width, device="cuda") / 16).requires_grad_()
    bias = torch.full((width,), 0.5, device="cuda", requires_grad=True)
    actual = double_feature_transform(us, them, white, black, weight, bias,
                                      255 / 256, width, "fused")
    expected = double_feature_transform(us, them, white, black, weight, bias,
                                        255 / 256, width, "torch")
    torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-6)
