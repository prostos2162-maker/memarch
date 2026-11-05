"""M4: паритет memarch_jax.moe з torch BankMoE (MOE_DISP=onehot, MOE_K=4, MOE_ACT=relu2), W=1, CPU.

Запуск:  cd memarch_jax && PYTHONIOENCODING=utf-8 python tests/test_moe.py
(або python -m pytest tests/test_moe.py -q, якщо є pytest)
"""
import os
import sys

os.environ.update(DEVICE="cpu", NO_COMPILE="1", ATTN_IMPL="block", MOE_DISP="onehot", MOE_K="4",
                  MOE_ACT="relu2")
os.environ.pop("BF16_ALL", None)
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")))  # repo root: torch reference code
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import numpy as np                                   # noqa: E402
import torch                                         # noqa: E402
import jax                                           # noqa: E402
import jax.numpy as jnp                              # noqa: E402

import full2                                         # noqa: E402
from memarch_jax import moe                          # noqa: E402

jax.config.update("jax_default_matmul_precision", "highest")


def _np(t):
    return t.detach().float().numpy()


def _cmp(name, a, b, res):
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    ad = np.abs(a - b).max() if a.size else 0.0
    rd = ad / max(np.abs(b).max(), 1e-30)
    res[name] = (ad, rd)
    return rd


def build_torch(d, E, group, align, seed, skew=0.0, fill=0.2 / 0.25):
    full2.MOE_GROUP, full2.MOE_CAP_ALIGN = group, align
    torch.manual_seed(seed)
    m = full2.BankMoE(d, E=E)
    m.fill = fill
    with torch.no_grad():
        m.norm.weight.add_(torch.randn(d) * 0.1)
        if skew:                                      # перекіс роутера: експерт 0 переповнюється
            m.router.weight[0] += skew * m.norm.weight.sign() / d ** 0.5
    m.train()
    return m


def params_of(m, dt):
    return {'norm': jnp.asarray(_np(m.norm.weight), dt), 'router': jnp.asarray(_np(m.router.weight), dt),
            'up': jnp.asarray(_np(m.up), dt), 'down': jnp.asarray(_np(m.down), dt)}


def run_case(d=128, E=8, B=2, A=256, group=128, align=0, skew=0.0, dtype="f32", seed=0, pad_frac=0.15,
             fill=0.2 / 0.25):
    tdt = torch.float32 if dtype == "f32" else torch.bfloat16
    jdt = jnp.float32 if dtype == "f32" else jnp.bfloat16
    m = build_torch(d, E, group, align, seed, skew, fill).to(tdt)
    g = torch.Generator().manual_seed(seed + 1)
    x = torch.randn(B, A, d, generator=g).to(tdt)
    sa = torch.rand(B, A, generator=g) * 10
    sa[torch.rand(B, A, generator=g) < pad_frac] = -1.0
    R = torch.randn(B, A, d, generator=g).to(tdt)
    xt = x.clone().requires_grad_(True)
    y = m(xt, sa)
    loss = (y.float() * R.float()).sum() + 3.0 * m.aux
    loss.backward()
    cfg = moe.MoECfg(E=E, k=4, group=group, cap_align=align, fill=fill)

    def f(p, xj):
        out, aux, dr = moe.bank_moe(p, xj, jnp.asarray(sa.numpy()), cfg)
        return (out.astype(jnp.float32) * jnp.asarray(_np(R))).sum() + 3.0 * aux, (out, aux, dr)
    p = params_of(m, jdt)
    (lj, (out, aux, dr)), (gp, gx) = jax.value_and_grad(f, (0, 1), has_aux=True)(p, jnp.asarray(_np(x), jdt))
    res = {}
    _cmp("out", out, _np(y), res)
    _cmp("aux", aux, m.aux.item(), res)
    _cmp("dropped", dr, float(m.dropped), res)
    _cmp("loss", lj, loss.item(), res)
    _cmp("g_x", gx, _np(xt.grad), res)
    for k, t in (("norm", m.norm.weight), ("router", m.router.weight), ("up", m.up), ("down", m.down)):
        _cmp("g_" + k, gp[k], _np(t.grad), res)
    c = moe.capacity(B * A, cfg)
    return res, c, float(m.dropped)


def report(title, res, c, drop):
    print(f"[{title}] C={c['C']} G={c['G']} Cg={c['Cg']} Ct={c['Ct']} dropped(torch)={drop:.4f}")
    for k, (ad, rd) in res.items():
        print(f"   {k:9s} max_abs={ad:.3e} max_rel={rd:.3e}")


def _check(res, tol, skip=()):
    bad = {k: v for k, v in res.items() if k not in skip and v[1] > tol}
    assert not bad, bad


def test_capacity_formula():
    cfg = moe.MoECfg()                                   # E=40, k=4, align 256, group 512, fill .8
    c = moe.capacity(2048, cfg)
    assert (c['C'], c['G'], c['Cg'], c['Ct']) == (256, 4, 64, 256), c
    c = moe.capacity(8192, cfg)
    assert (c['C'], c['G'], c['Cg'], c['Ct']) == (768, 16, 48, 768), c


def test_f32_basic():
    res, c, drop = run_case(dtype="f32")
    report("f32 basic", res, c, drop)
    _check(res, 1e-5)


def test_f32_overflow():
    res, c, drop = run_case(dtype="f32", fill=0.2, seed=3)
    report("f32 overflow", res, c, drop)
    assert drop > 0.05, drop
    _check(res, 1e-5)


def test_f32_align256_uneven_groups():
    # N = 3·160 = 480: g=min(128,480) → G=4, Np=512 (доповнення групи), C вирівняно до 256
    res, c, drop = run_case(dtype="f32", B=3, A=160, align=256, seed=5)
    report("f32 align256 pad-group", res, c, drop)
    _check(res, 1e-5)


def test_bf16():
    for title, kw in (("bf16 basic", {}), ("bf16 overflow", dict(fill=0.2, seed=3))):
        res, c, drop = run_case(dtype="bf16", **kw)
        report(title, res, c, drop)
        assert res["dropped"][0] < 1e-6
        # bf16: логіти роутера на CPU-torch відрізняються на 1 ulp (інший порядок накопичення) → кілька
        # слів міняють top-k; тож жорстко перевіряємо лише aux і частку відкинутих, решта — звіт
        assert res["aux"][1] < 1e-3, res["aux"]
        assert res["out"][1] < 1e-2, res["out"]                 # фактично побітово (0)
        _check(res, 2e-2)                                         # градієнти bf16: ~1e-2 max-rel


def test_heavy_stack_f32():
    """Стек L шарів + gather/scatter_add якорів проти torch-циклу full2.py:1688-1708."""
    d, E, L, B, T, A = 64, 8, 3, 2, 256, 64
    full2.MOE_GROUP, full2.MOE_CAP_ALIGN = 32, 0
    layers = [build_torch(d, E, 32, 0, 10 + i) for i in range(L)]
    g = torch.Generator().manual_seed(7)
    h = torch.randn(B, T, d, generator=g)
    s_ = torch.rand(B, T, generator=g) * 60
    idx = torch.stack([torch.randperm(T, generator=g)[:A].sort().values for _ in range(B)])
    valid = torch.rand(B, A, generator=g) > 0.2
    R = torch.randn(B, T, d, generator=g)
    ht = h.clone().requires_grad_(True)
    a_in = ht.gather(1, idx[..., None].expand(-1, -1, d))
    a = a_in
    sa = torch.where(valid, s_.gather(1, idx).clamp(0, 50), -1.0)
    auxs = []
    for b in layers:
        a = b(a, sa)
        auxs.append(b.aux)
    ho = ht.scatter_add(1, idx[..., None].expand(-1, -1, d), (a - a_in) * valid[..., None])
    loss = (ho * R).sum() + 0.01 * torch.stack(auxs).mean() * 100
    loss.backward()
    ps = {k: jnp.stack([params_of(b, jnp.float32)[k] for b in layers]) for k in ("norm", "router", "up", "down")}
    cfg = moe.MoECfg(E=E, k=4, group=32, cap_align=0)

    def f(ps, hj):
        ho, aux, dr = moe.heavy_stack(ps, hj, jnp.asarray(idx.numpy()), jnp.asarray(valid.numpy()),
                                      jnp.asarray(s_.numpy()), cfg)
        return (ho * jnp.asarray(R.numpy())).sum() + aux, (ho, aux, dr)
    (lj, (hoj, aux, dr)), (gp, gh) = jax.value_and_grad(f, (0, 1), has_aux=True)(ps, jnp.asarray(h.numpy()))
    res = {}
    _cmp("h_out", hoj, _np(ho), res)
    _cmp("aux_mean", aux, torch.stack(auxs).mean().item(), res)
    _cmp("drop_mean", dr, float(np.mean([float(b.dropped) for b in layers])), res)
    _cmp("g_h", gh, _np(ht.grad), res)
    for k, nm in (("norm", "norm.weight"), ("router", "router.weight"), ("up", "up"), ("down", "down")):
        tg = torch.stack([dict(b.named_parameters())[nm].grad for b in layers])
        _cmp("g_" + k, gp[k], _np(tg), res)
    report("f32 heavy_stack L=3", res, {'C': '-', 'G': '-', 'Cg': '-', 'Ct': '-'}, float('nan'))
    _check(res, 1e-5)


if __name__ == "__main__":
    for nm, fn in list(globals().items()):
        if nm.startswith("test_") and callable(fn):
            fn()
            print("OK", nm)
