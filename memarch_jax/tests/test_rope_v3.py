"""Паритет rope_read_v3 (interpret на CPU) зі старим ядром rope_pallas.read і з rope.py (XLA). Запуск:
cd D:\\ai\\memarch_jax && JAX_PLATFORMS=cpu PYTHONIOENCODING=utf-8 python tests/test_rope_v3.py"""
import os
import sys

os.environ["JAX_PLATFORMS"] = "cpu"
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np                          # noqa: E402
import jax                                  # noqa: E402
import jax.numpy as jnp                     # noqa: E402

from memarch_jax import rope as R           # noqa: E402

B, H, T, hd = 2, 2, 512, 8


def data(seed, cap=0.1):
    g = np.random.default_rng(seed)
    q, k, v = (g.normal(0, 1, (B, H, T, hd)).astype(np.float32) * s for s in (2.0, 2.0, 1.0))
    s = g.exponential(1.0, (B, T)).astype(np.float32)
    s[:, 0] = 1e4
    return q, k, v, s, max(1, round(T * cap))


def cfg(pallas, read="v1", **kw):
    base = dict(K=4, links=8, win=32, span_cap=64, gather_cap=None, route="exact", excl_win=True, route_rot=True,
                rgrad=True, BM=64, chunk=64, pallas=pallas, read=read)
    base.update(kw)
    return R.RopeCfg(**base)


def run(rc, own, apos, av, dt, q, k, v, W, qmask, routed=None):
    def f(q, k, v):
        o, sh, st = R.rope_attention(q.astype(dt), k.astype(dt), q.astype(dt), k.astype(dt), v.astype(dt), own, apos, av,
                                     rc, qmask=qmask, routed=routed)
        return (o.astype(jnp.float32) * W).sum(), (o, sh)
    return jax.value_and_grad(f, argnums=(0, 1, 2), has_aux=True)(q, k, v)


def compare(seed, dt, tol, ref="v1", tau=0.7, qfrac=1.0, tiles=(128, 256), **kw):
    os.environ["ROPE_V3_BQ"], os.environ["ROPE_V3_BK"] = str(tiles[0]), str(tiles[1])
    q, k, v, s, A = data(seed)
    own, apos, av = R.anchors_from_surprise(jnp.asarray(s), jnp.asarray(tau), A)
    W = jnp.asarray(np.random.default_rng(seed + 9).normal(0, 1, (B, H, T, hd)).astype(np.float32))
    qmask = None if qfrac >= 1 else jnp.asarray(s >= np.quantile(s, 1 - qfrac))
    a = (own, apos, av, dt, jnp.asarray(q), jnp.asarray(k), jnp.asarray(v), W, qmask)
    (l0, (o0, sh0)), g0 = run(cfg(ref == "v1", **kw), *a)
    (l1, (o1, sh1)), g1 = run(cfg(True, "v3", **kw), *a)
    o0, o1 = np.asarray(o0.astype(jnp.float32)), np.asarray(o1.astype(jnp.float32))
    d = np.abs(o0 - o1).max() / np.abs(o0).max()
    dg = [float(np.abs(np.asarray(x) - np.asarray(y)).max() / max(np.abs(np.asarray(y)).max(), 1e-9)) for x, y in zip(g1, g0)]
    print(f"seed {seed} {dt.__name__} ref={ref} tau={tau} qfrac={qfrac} tiles={tiles} {kw}: out {d:.2e}, "
          f"grad q/k/v {dg[0]:.2e}/{dg[1]:.2e}/{dg[2]:.2e}, share {float(sh1):.5f} vs {float(sh0):.5f}", flush=True)
    assert np.isfinite(o1).all() and d <= tol and max(dg) <= tol and abs(float(sh1) - float(sh0)) <= tol, (d, dg)
    return float(sh0)


def test_leak():
    q, k, v, s, A = data(5)
    own, apos, av = R.anchors_from_surprise(jnp.asarray(s), jnp.asarray(0.7), A)
    rc = cfg(True, "v3")
    f = lambda a, b, c: R.rope_attention(a, b, a, b, c, own, apos, av, rc)[0]
    o0 = np.asarray(f(jnp.asarray(q), jnp.asarray(k), jnp.asarray(v)))
    for cut in (100, 256, 400):
        q2, k2, v2 = q.copy(), k.copy(), v.copy()
        for a in (q2, k2, v2):
            a[:, :, cut + 1:] += 5.0
        o1 = np.asarray(f(jnp.asarray(q2), jnp.asarray(k2), jnp.asarray(v2)))
        assert (o0[:, :, :cut + 1] == o1[:, :, :cut + 1]).all(), cut
    print("leak OK")


def test_shared():
    """Спільний пошук (routed від іншого шару): v3 == v1."""
    q, k, v, s, A = data(1)
    own, apos, av = R.anchors_from_surprise(jnp.asarray(s), jnp.asarray(0.7), A)
    g = np.random.default_rng(3)
    q0, k0, v0 = jnp.asarray(q), jnp.asarray(k), jnp.asarray(v)
    _, _, st = R.rope_attention(q0, k0, q0, k0, v0, own, apos, av, cfg(True))
    q2, k2 = jnp.asarray(g.normal(0, 2, q.shape).astype(np.float32)), jnp.asarray(g.normal(0, 2, k.shape).astype(np.float32))
    W = jnp.asarray(g.normal(0, 1, (B, H, T, hd)).astype(np.float32))
    a = (own, apos, av, jnp.float32, q2, k2, v0, W, None)
    (_, (o0, _)), g0 = run(cfg(True), *a, routed=st["_route"])
    (_, (o1, _)), g1 = run(cfg(True, "v3"), *a, routed=st["_route"])
    d = float(jnp.abs(o0 - o1).max() / jnp.abs(o0).max())
    dg = max(float(jnp.abs(x - y).max() / jnp.abs(y).max()) for x, y in zip(g1, g0))
    print(f"shared route: out {d:.2e} grad {dg:.2e}")
    assert d < 2e-5 and dg < 2e-5


if __name__ == "__main__":
    shares = [compare(0, jnp.float32, 2e-5),
              compare(1, jnp.float32, 2e-5, excl_win=False),
              compare(2, jnp.float32, 2e-5, rgrad=False),
              compare(3, jnp.float32, 2e-5, tiles=(512, 512)),
              compare(4, jnp.float32, 2e-5, tiles=(64, 128), tau=0.2),      # багато коротких якорів
              compare(5, jnp.float32, 2e-5, tiles=(256, 64), tau=3.0)]      # мало якорів -> прольоти впираються в span_cap
    assert min(shares) > 0.01, shares                                        # тести справді читають трос
    compare(6, jnp.float32, 2e-5, qfrac=0.25)                               # частковий трос
    compare(7, jnp.float32, 2e-5, tau=100.0)                                # один якір
    compare(0, jnp.float32, 2e-4, ref="xla")                                # проти чистого XLA rope.py
    compare(3, jnp.bfloat16, 3e-2)
    test_shared()
    test_leak()
    print("V3_OK")
