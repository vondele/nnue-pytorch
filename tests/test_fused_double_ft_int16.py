"""Parity of the int16 fused FT kernels against the fp32 fused kernels.

With fake-quantization the merged FT weight and bias live on the k/256 grid,
so gathering an int16 table and accumulating in int32 computes exactly the
values of the fp32 path (whose partial sums stay on the grid below 2^24), and
clamped_out stores exact uint8 quantization levels. The forward must be
bit-identical for one-hot us/them (as produced by the training data loader);
gradients additionally go through atomics, so they match up to summation
order.
"""

import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.modules.feature_transformer.double_ft_functions import double_feature_transform
from model.modules.feature_transformer.fused_ft_functions import int16_ft_available

_HAS_CUPY = True
try:
    import cupy  # noqa: F401

    from model.modules.feature_transformer.fused_ft_functions import _HAS_CUPY_KERNELS
    _HAS_CUPY = _HAS_CUPY_KERNELS
except (ImportError, OSError, RuntimeError):
    _HAS_CUPY = False

requires_cupy = pytest.mark.skipif(not _HAS_CUPY, reason="CuPy kernels required")


def _make_inputs(batch, active, width, rows, k_range=127, seed=1234):
    torch.manual_seed(seed)
    device = torch.device("cuda")
    us = (torch.rand(batch, 1, device=device) > 0.5).float()
    them = 1.0 - us
    weight = torch.randint(-k_range, k_range + 1, (rows, width), device=device).float() / 256.0
    bias = torch.randint(-k_range, k_range + 1, (width,), device=device).float() / 256.0
    white = torch.full((batch, active), -1, dtype=torch.int32, device=device)
    black = torch.full_like(white, -1)
    for row in range(batch):
        n = min(31, row % 32)
        white[row, :n] = torch.randperm(rows, device=device)[:n]
        black[row, : n // 2] = torch.randperm(rows, device=device)[: n // 2]
    white[-1] = torch.arange(active, device=device) % rows
    return us, them, white, black, weight, bias


MAXACT = 127 / 256
MAXACT_K = 127


def _forward(backend, us, them, white, black, weight, bias, l1):
    return double_feature_transform(
        us, them, white, black, weight, bias, MAXACT, l1, backend
    )


def _forward_backward(backend, us, them, white, black, weight, bias, l1, upstream):
    w = weight.clone().requires_grad_(True)
    b = bias.clone().requires_grad_(True)
    out = _forward(backend, us, them, white, black, w, b, l1)
    (out * upstream).sum().backward()
    return out.detach(), w.grad, b.grad


@requires_cupy
@pytest.mark.parametrize("l1", [1024, 1152])
@pytest.mark.parametrize("batch,rows", [(17, 512), (1025, 512)])
@pytest.mark.parametrize("k_range", [127, 3000])
def test_int16_forward_bit_identical(l1, batch, rows, k_range):
    active = 288
    us, them, white, black, weight, bias = _make_inputs(batch, active, l1, rows, k_range)
    expected = _forward("fused", us, them, white, black, weight, bias, l1)
    actual = _forward("fused_int16", us, them, white, black, weight, bias, l1)
    assert torch.equal(actual, expected)


@requires_cupy
def test_int16_forward_matches_torch_backend():
    l1, batch, rows, active = 1024, 33, 256, 288
    us, them, white, black, weight, bias = _make_inputs(batch, active, l1, rows)
    expected = _forward("torch", us, them, white, black, weight, bias, l1)
    actual = _forward("fused_int16", us, them, white, black, weight, bias, l1)
    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)


@requires_cupy
@pytest.mark.parametrize("l1", [1024, 1152])
@pytest.mark.parametrize("batch", [17, 1025])
def test_int16_backward_matches(l1, batch):
    active, rows = 288, 512
    us, them, white, black, weight, bias = _make_inputs(batch, active, l1, rows)
    torch.manual_seed(99)
    upstream = torch.randn(batch, l1, device="cuda")

    _, gw_exp, gb_exp = _forward_backward("fused", us, them, white, black, weight, bias, l1, upstream)
    _, gw_act, gb_act = _forward_backward("fused_int16", us, them, white, black, weight, bias, l1, upstream)

    # Both paths accumulate the same values through atomics; only the
    # completion order can differ.
    torch.testing.assert_close(gw_act, gw_exp, atol=3e-4, rtol=3e-4)
    torch.testing.assert_close(gb_act, gb_exp, atol=3e-4, rtol=3e-4)


@requires_cupy
def test_int16_clamp_boundaries():
    """Pre-activations exactly at 0 and the clamp level, and saturated above."""
    l1, batch, rows, active = 1024, 8, 8, 4
    device = torch.device("cuda")
    us = torch.ones(batch, 1, device=device)
    them = torch.zeros(batch, 1, device=device)
    # k=0 rows exercise the zero gate; large sums saturate at the clamp.
    weight = torch.zeros(rows, l1, device=device)
    weight[0] = 0.0
    weight[1] = 300.0
    bias = torch.zeros(l1, device=device)
    white = torch.tensor([[0, 1, -1, -1]] * batch, dtype=torch.int32, device=device)
    black = torch.full((batch, active), -1, dtype=torch.int32, device=device)

    expected = _forward("fused", us, them, white, black, weight, bias, l1)
    actual = _forward("fused_int16", us, them, white, black, weight, bias, l1)
    assert torch.equal(actual, expected)

    # The saturated output must be exactly maxact^2 on both paths.
    assert torch.equal(actual[:, : l1 // 2], expected[:, : l1 // 2])
    assert (actual[:, : l1 // 2] == MAXACT * MAXACT).all()


@requires_cupy
def test_int16_overflow_flag_trips_on_offgrid_weight():
    l1, batch, rows, active = 1024, 8, 8, 4
    device = torch.device("cuda")
    us = torch.ones(batch, 1, device=device)
    them = torch.zeros(batch, 1, device=device)
    weight = torch.zeros(rows, l1, device=device)
    weight[0, 0] = 1.0 / 3.0  # off the k/256 grid
    bias = torch.zeros(l1, device=device)
    white = torch.tensor([[0, -1, -1, -1]] * batch, dtype=torch.int32, device=device)
    black = torch.full((batch, active), -1, dtype=torch.int32, device=device)

    from model.modules.feature_transformer.fused_ft_functions import _OVERFLOW_FLAGS

    _OVERFLOW_FLAGS.clear()
    _forward("fused_int16", us, them, white, black, weight, bias, l1)
    flag = _OVERFLOW_FLAGS[torch.cuda.current_device()]
    # The flag is set asynchronously; it is checked on a deferred cadence.
    torch.cuda.synchronize()
    assert flag.item() == 1
    _OVERFLOW_FLAGS.clear()


def test_int16_eligibility():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    weight = torch.zeros(8, 16, device=device)
    bias = torch.zeros(16, device=device)
    if torch.cuda.is_available() and _HAS_CUPY:
        assert int16_ft_available(weight, bias, 127 / 256, 16)
    # Clamp level not an exact uint8 step (0.3 * 256 = 76.8).
    assert not int16_ft_available(weight, bias, 0.3, 16)
    # Clamp level above the uint8 range.
    assert not int16_ft_available(weight, bias, 1.0, 16)
    # Odd width.
    assert not int16_ft_available(weight, bias, 127 / 256, 15)


def test_int16_env_kill_switch():
    """NNUE_FT_INT16=0 must keep the composed transformer on the fp32 path."""

    class FakeQuant:
        weight_scales_dict = {"ft_weight": 256.0, "ft_bias": 256.0}
        max_ft_activation = 127 / 256

    from model.modules.feature_transformer.composed_feature_transformer import (
        ComposedFeatureTransformer,
    )

    fake = object.__new__(ComposedFeatureTransformer)
    fake.quantization = FakeQuant()
    assert fake._int16_backend_ok(True)
    os.environ["NNUE_FT_INT16"] = "0"
    try:
        assert not fake._int16_backend_ok(True)
        assert not fake._int16_backend_ok(False)
    finally:
        del os.environ["NNUE_FT_INT16"]
    assert fake._int16_backend_ok(True)
