"""Compact accumulation: rounding, overflow recovery, cross-row overlap and streams."""

import pytest
import torch

from model.modules.feature_transformer.fused_ft_functions import _HAS_CUPY_KERNELS


@pytest.mark.skipif(
    not torch.cuda.is_available() or not _HAS_CUPY_KERNELS,
    reason="CUDA and CuPy required",
)
@pytest.mark.parametrize("width", [1024, 1152, 1280])
@pytest.mark.parametrize("kind", ["normal", "overflow", "overlap", "empty"])
def test_compact_gradients(width, kind):
    if torch.cuda.get_device_capability() != (9, 0):
        pytest.skip("H100 specialization")
    from model.modules.feature_transformer.aggregated_ft_kernel import (
        aggregated_ft_backward,
    )

    torch.manual_seed(190)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        torch.cuda._sleep(2_000_000)
        batch = 1025
        active = 17
        us = torch.randint(0, 2, (batch, 1), device="cuda").float()
        them = 1 - us
        white = torch.stack(
            [torch.randperm(64, device="cuda")[:active] for _ in range(batch)]
        ).int()
        black = white.roll(3, 0).clone()
        if kind == "overlap":
            # Every position shares IDs; each perspective still has unique indices.
            white[:] = torch.arange(active, device="cuda")
            black[:] = white
        white[:, 9:] = -1
        black[:, 13:] = -1
        if kind == "empty":
            white.fill_(-1)
            black.fill_(-1)
        clamps = torch.rand(batch, 4, width // 2, device="cuda")
        clamps[clamps < 0.2] = 0
        clamps[clamps > 0.8] = 1
        dy = torch.randn(batch, width, device="cuda") * (
            1.0 if kind == "overflow" else 1e-5
        )
        ref = torch.zeros(64, width, device="cuda")
        rb = torch.zeros(width, device="cuda")
        actual = torch.empty_like(ref)
        ab = torch.zeros_like(rb)
        aggregated_ft_backward(us, them, white, black, dy, clamps, ref, rb, 1.0)
        aggregated_ft_backward(
            us, them, white, black, dy, clamps, actual, ab, 1.0, compact=True
        )
        assert torch.isfinite(actual).all()
        relative = torch.linalg.vector_norm(actual - ref) / torch.linalg.vector_norm(
            ref
        ).clamp_min(1e-30)
        assert relative < (1e-5 if kind == "overflow" else 0.008), float(relative)
        if kind == "empty":
            assert not torch.count_nonzero(actual)
        torch.testing.assert_close(ab, rb, atol=1e-4, rtol=1e-4)
    stream.synchronize()
