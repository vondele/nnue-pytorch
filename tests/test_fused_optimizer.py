"""Bit-identity of the fused RangerLite paths against the unfused reference ops."""
import pytest
import torch

from model.optimizers import ranger_lite as RL

cuda = pytest.mark.skipif(not (torch.cuda.is_available() and RL._HAS_CUPY), reason="CUDA + CuPy")


def _run(fuse_lookahead, alpha, pnm, steps=13, seed=0):
    RL._FUSE_LOOKAHEAD = fuse_lookahead
    g = torch.Generator(device="cuda").manual_seed(seed)
    params = [torch.nn.Parameter(torch.randn(n, device="cuda", generator=g)) for n in (1000, 77, 4096)]
    opt = RL.RangerLite([{"params": params, "lr": 1e-2, "weight_decay": 0.0}], normloss_active=False,
                        use_stable_weight_decay=False, pnm_activate=pnm, lookahead_blending_alpha=alpha,
                        lookahead_mergetime=5)
    for s in range(steps):
        for p in params:
            p.grad = torch.randn(p.shape, device="cuda", generator=g)
            if s % 3 == 0:
                p.grad[: p.numel() // 4] = 0
        scale = torch.full((), 0.7, device="cuda") if s % 2 else None
        opt.step(grad_scale=scale)
    out = []
    for p in params:
        st = opt.state[p]
        out += [p.detach().clone()] + [st[k].clone() for k in sorted(st) if torch.is_tensor(st[k]) and st[k].dim()]
    return out, opt.lookahead_step


@cuda
@pytest.mark.parametrize("alpha", [0.5, 0.3, 0.8])
@pytest.mark.parametrize("pnm", [True, False])
def test_fused_lookahead_bit_identical(alpha, pnm):
    try:
        a, sa = _run(True, alpha, pnm)
        b, sb = _run(False, alpha, pnm)
    finally:
        RL._FUSE_LOOKAHEAD = True
    assert sa == sb
    for x, y in zip(a, b):
        assert torch.equal(x, y)


def _run_rows(use_flags, pnm, steps=12, seed=1):
    from model.optimizers import grad_row_flags as GRF
    RL._ROW_FLAGS = use_flags
    g = torch.Generator(device="cuda").manual_seed(seed)
    rows, cols = 300, 64
    w = torch.nn.Parameter(torch.randn(rows, cols, device="cuda", generator=g))
    d = torch.nn.Parameter(torch.randn(50, device="cuda", generator=g))
    opt = RL.RangerLite([{"params": [w, d], "lr": 1e-2, "weight_decay": 0.0}], normloss_active=False,
                        use_stable_weight_decay=False, pnm_activate=pnm, lookahead_mergetime=5)
    for s in range(steps):
        # rows >= 200 are never touched (dormant); the others are touched at random
        touched = (torch.rand(rows, device="cuda", generator=g) < 0.5) & (torch.arange(rows, device="cuda") < 200)
        full = torch.randn(rows, cols, device="cuda", generator=g) * touched[:, None]
        base = torch.cat([torch.zeros(7, cols, device="cuda"), full])   # w's gradient is a row-range view
        w.grad = base[7:]
        GRF.register(w.grad, touched.to(torch.uint8), reset=True)
        assert GRF.lookup(w.grad) is not None
        d.grad = torch.randn(50, device="cuda", generator=g)
        opt.step(grad_scale=torch.full((), 0.6, device="cuda") if s % 2 else None)
    return [w.detach().clone(), d.detach().clone()] + [t.clone() for p in (w, d) for t in opt.state[p].values()
                                                        if torch.is_tensor(t) and t.dim()]


@cuda
@pytest.mark.parametrize("pnm", [True, False])
def test_row_flags_bit_identical(pnm):
    from model.optimizers import grad_row_flags as GRF
    try:
        a = _run_rows(True, pnm)
        b = _run_rows(False, pnm)
    finally:
        RL._ROW_FLAGS = True
    for x, y in zip(a, b):
        assert torch.equal(x, y)
    # a gradient changed in place after registration is not looked up
    base = torch.zeros(4, 8, device="cuda")
    v = base[1:3]
    GRF.register(v, torch.zeros(2, dtype=torch.uint8, device="cuda"), reset=True)
    assert GRF.lookup(v) is not None and GRF.lookup(base[1:3]) is not None and GRF.lookup(base[0:2]) is None
    base.add_(1)
    assert GRF.lookup(v) is None


@cuda
def test_flagged_total_norm():
    from model.optimizers import grad_row_flags as GRF
    g = torch.Generator(device="cuda").manual_seed(3)
    touched = torch.rand(5000, device="cuda", generator=g) < 0.5
    base = torch.randn(5000, 1024, device="cuda", generator=g) * touched[:, None]
    a, b = base[:3000], base[3000:]
    GRF.register(a, touched[:3000].to(torch.uint8), reset=True)
    GRF.register(b, touched[3000:].to(torch.uint8))
    other = torch.randn(777, device="cuda", generator=g)
    ref = torch.nn.utils.get_total_norm([a, b, other], 2.0).double()
    got = GRF.total_norm([a, b, other]).double()
    exact = torch.sqrt((base.double() ** 2).sum() + (other.double() ** 2).sum())
    assert abs(got - ref) / ref < 1e-6 and abs(got - exact) / exact < 1e-6
    GRF.clear()
    assert torch.equal(GRF.total_norm([a, b, other]), torch.nn.utils.get_total_norm([a, b, other], 2.0))

