"""Паритет ядер rope_pallas (interpret на CPU) з rope.py і причинність. Запуск:
cd D:\\ai\\memarch_jax && JAX_PLATFORMS=cpu PYTHONIOENCODING=utf-8 python tests/test_rope_pallas.py"""
import dataclasses
import os
import sys

os.environ["JAX_PLATFORMS"] = "cpu"
os.environ.setdefault("ROPE_HALF", "1")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np                          # noqa: E402
import jax                                  # noqa: E402
import jax.numpy as jnp                     # noqa: E402

from memarch_jax import rope as R           # noqa: E402

B, H, T, hd = 1, 2, 512, 8


def data(seed, cap=0.1):
    g = np.random.default_rng(seed)
    q, k, v = (g.normal(0, 1, (B, H, T, hd)).astype(np.float32) * s for s in (2.0, 2.0, 1.0))
    s = g.exponential(1.0, (B, T)).astype(np.float32)
    s[:, 0] = 1e4
    return q, k, v, s, max(1, round(T * cap))


def cfg(pallas, **kw):
    base = dict(K=4, links=8, win=32, span_cap=64, gather_cap=None, route="exact", excl_win=True, route_rot=True,
                rgrad=True, BM=64, chunk=64, pallas=pallas)
    base.update(kw)
    return R.RopeCfg(**base)


def fn(rc, own, apos, av, dt):
    def f(q, k, v):
        o, sh, st = R.rope_attention(q.astype(dt), k.astype(dt), q.astype(dt), k.astype(dt), v.astype(dt), own, apos, av, rc)
        return ((o.astype(jnp.float32) * W_).sum(), (o, sh, st))
    return f


W_ = None


def compare(seed, dt, tol, **kw):
    global W_
    q, k, v, s, A = data(seed)
    own, apos, av = R.anchors_from_surprise(jnp.asarray(s), jnp.asarray(0.7), A)
    W_ = jnp.asarray(np.random.default_rng(seed + 9).normal(0, 1, (B, H, T, hd)).astype(np.float32))
    qj, kj, vj = jnp.asarray(q), jnp.asarray(k), jnp.asarray(v)
    r0 = jax.value_and_grad(fn(cfg(False, **kw), own, apos, av, dt), argnums=(0, 1, 2), has_aux=True)(qj, kj, vj)
    r1 = jax.value_and_grad(fn(cfg(True, **kw), own, apos, av, dt), argnums=(0, 1, 2), has_aux=True)(qj, kj, vj)
    (l0, (o0, sh0, st0)), g0 = r0
    (l1, (o1, sh1, st1)), g1 = r1
    o0, o1 = np.asarray(o0.astype(jnp.float32)), np.asarray(o1.astype(jnp.float32))
    d = np.abs(o0 - o1).max() / np.abs(o0).max()
    dg = [float(np.abs(np.asarray(a) - np.asarray(b)).max() / max(np.abs(np.asarray(b)).max(), 1e-9)) for a, b in zip(g1, g0)]
    print(f"seed {seed} {dt.__name__} {kw}: out {d:.2e}, grad q/k/v {dg[0]:.2e}/{dg[1]:.2e}/{dg[2]:.2e}, share {float(sh1):.5f} vs {float(sh0):.5f}")
    assert d <= tol and max(dg) <= tol and abs(float(sh1) - float(sh0)) <= tol, (d, dg)


def test_route():
    """Вибір якорів ядром == rope._route_exact."""
    from memarch_jax import rope_pallas as rp
    for seed in range(3):
        q, k, v, s, A = data(seed)
        own, apos, av = R.anchors_from_surprise(jnp.asarray(s), jnp.asarray(0.7), A)
        rc = cfg(True)
        q0, k0 = jnp.asarray(q), jnp.asarray(k)
        ti0, tv0, tp0 = R._route_exact(q0, k0, own, apos, av, rc)
        kl, lval, lp = R._links(k0, own, apos, av, 8)
        nxt = R.next_anchor(apos, av, T)
        ti1, tv1, tp1, ts, st, nn = rp.route_exact(q0, kl, lval, lp, av, R._route_nxt(nxt, apos, rc), apos, nxt, min(4, A), 64)
        for a, b in ((ti0, ti1), (tv0, tv1), (tp0, tp1)):
            assert (np.asarray(a) == np.asarray(b)).all()
        print(f"route seed {seed}: tidx/tval/tpos exact, opened {float(np.asarray(tv1).mean()):.3f}")


def test_leak():
    """Причинність: зміна q/k/v ПІСЛЯ i не змінює вихід у ≤ i (ядро)."""
    q, k, v, s, A = data(5)
    own, apos, av = R.anchors_from_surprise(jnp.asarray(s), jnp.asarray(0.7), A)
    rc = cfg(True)
    f = lambda a, b, c: R.rope_attention(a, b, a, b, c, own, apos, av, rc)[0]
    o0 = np.asarray(f(jnp.asarray(q), jnp.asarray(k), jnp.asarray(v)))
    for cut in (100, 256, 400):
        q2, k2, v2 = q.copy(), k.copy(), v.copy()
        for a in (q2, k2, v2):
            a[:, :, cut + 1:] += 5.0
        o1 = np.asarray(f(jnp.asarray(q2), jnp.asarray(k2), jnp.asarray(v2)))
        assert (o0[:, :, :cut + 1] == o1[:, :, :cut + 1]).all(), cut
        assert np.abs(o0[:, :, cut + 1:] - o1[:, :, cut + 1:]).max() > 0
    print("leak test OK: майбутні q/k/v не змінюють минулих виходів")


def test_static():
    """Один трейс на різні щільності якорів."""
    q, k, v, s, A = data(0)
    rc = cfg(True)
    tr = {"n": 0}

    def f(a, b, c, own, apos, av):
        tr["n"] += 1
        return R.rope_attention(a, b, a, b, c, own, apos, av, rc)[0]
    fj = jax.jit(f)
    for tau in (0.7, 2.5, 100.0):
        own, apos, av = R.anchors_from_surprise(jnp.asarray(s), jnp.asarray(tau), A)
        assert np.isfinite(np.asarray(fj(jnp.asarray(q), jnp.asarray(k), jnp.asarray(v), own, apos, av))).all()
    assert tr["n"] == 1
    print("static shapes OK: 1 trace, 3 densities")


def test_shared():
    """Спільний пошук: другий шар із готовим вибором першого — те саме, що власний вибір тих самих ключів; bias через ядро."""
    q, k, v, s, A = data(1)
    own, apos, av = R.anchors_from_surprise(jnp.asarray(s), jnp.asarray(0.7), A)
    rc = cfg(True)
    q0, k0, v0 = jnp.asarray(q), jnp.asarray(k), jnp.asarray(v)
    o1, _, st = R.rope_attention(q0, k0, q0, k0, v0, own, apos, av, rc)
    o2, _, _ = R.rope_attention(q0, k0, q0, k0, v0, own, apos, av, rc, routed=st["_route"])
    d = float(jnp.abs(o1 - o2).max() / jnp.abs(o1).max())
    print(f"shared route, same keys: rel {d:.2e}")
    assert d < 2e-6


if __name__ == "__main__":
    test_route()
    test_leak()
    test_static()
    test_shared()
    compare(0, jnp.float32, 2e-5)
    compare(1, jnp.float32, 2e-5, excl_win=False)
    compare(2, jnp.float32, 2e-5, rgrad=False)
    compare(3, jnp.bfloat16, 3e-2)
    print("PALLAS_OK")
