"""Новий шлях MoE (MoECfg.fast: bf16 one-hot, без транспозицій, конвеєр шматками a2a/ppermute) проти старого,
W=8 у shard_map на CPU. Очікування: побітово однакові out/aux/dropped і градієнти (лише переміщення/розбиття рядків).

Запуск: XLA_FLAGS=--xla_force_host_platform_device_count=8 python tests/test_moe_ep.py
"""
import os
_f = os.environ.get("XLA_FLAGS", "")
if "device_count" not in _f:
    _f += " --xla_force_host_platform_device_count=8"
if "excess_precision" not in _f:      # інакше CPU-XLA викидає проміжні bf16-округлення залежно від злиття (не наш шум)
    _f += " --xla_allow_excess_precision=false"
os.environ["XLA_FLAGS"] = _f
import sys                                                   # noqa: E402
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from dataclasses import replace                              # noqa: E402
from functools import partial                                # noqa: E402

import numpy as np                                           # noqa: E402
import jax                                                   # noqa: E402
import jax.numpy as jnp                                      # noqa: E402
from jax.sharding import Mesh, PartitionSpec as P            # noqa: E402

from memarch_jax import moe                                  # noqa: E402

try:
    _sm = partial(jax.shard_map, check_vma=False)
except AttributeError:
    from jax.experimental.shard_map import shard_map
    _sm = partial(shard_map, check_rep=False)

W = 8


def _run(cfg, impl, dt, L=2, B=2, A=256, d=64, seed=0):
    E, h2 = cfg.E, 96
    k = jax.random.split(jax.random.PRNGKey(seed), 8)
    ps = dict(norm=1 + 0.1 * jax.random.normal(k[0], (L, d)),
              router=jax.random.normal(k[1], (L, E, d)) * 0.3,
              up=jax.random.normal(k[2], (L, E, d, h2)) * 0.2,
              down=jax.random.normal(k[3], (L, E, h2, d)) * 0.2)
    ps = jax.tree.map(lambda t: t.astype(dt), ps)
    T = A
    h = jax.random.normal(k[4], (W * B, T, d)).astype(dt)
    idx = jnp.broadcast_to(jnp.arange(A), (W * B, A))
    valid = jax.random.uniform(k[5], (W * B, A)) > 0.1
    s_ = jax.random.uniform(k[6], (W * B, T)) * 10
    R = jax.random.normal(k[7], (W * B, T, d))
    mesh = Mesh(np.asarray(jax.devices()[:W]), ("x",))
    ex = moe.a2a_exchange("x", W, impl)
    pspec = dict(norm=P(), router=P(), up=P(None, "x"), down=P(None, "x"))

    def local(ps, h, idx, valid, s_, R):
        st = {}
        ho, aux, dr = moe.heavy_stack(ps, h, idx, valid, s_, cfg, ex, remat=True, stats=st)
        l = (ho.astype(jnp.float32) * R).sum() + 3.0 * aux
        return jax.lax.psum(l, "x"), jax.lax.pmean(st["moe_fill"], "x"), ho

    f = _sm(local, mesh=mesh, in_specs=(pspec, P("x"), P("x"), P("x"), P("x"), P("x")),
            out_specs=(P(), P(), P("x")))

    def lossf(ps, h):
        l, fill, ho = f(ps, h, idx, valid, s_, R)
        return l, (fill, ho)
    (l, (fill, ho)), (gp, gh) = jax.jit(jax.value_and_grad(lossf, (0, 1), has_aux=True))(ps, h)
    return dict(l=l, ho=ho, gh=gh, **{"g_" + n: v for n, v in gp.items()}), float(fill)


def _same(a, b, tag, reorder=()):
    """Побітово, крім reorder (градієнти ваг експертів при K>1: dW = сума K часткових згорток — інший порядок fp32)."""
    worst = 0
    for n in a:
        x, y = np.asarray(a[n]), np.asarray(b[n])
        if n in reorder:
            rel = float(np.abs(x.astype(np.float32) - y.astype(np.float32)).max() / np.abs(x.astype(np.float32)).max())
            assert rel < 1e-5 if x.dtype == np.float32 else rel < 1e-2, (tag, n, u, rel)
            worst = max(worst, rel)
        else:
            assert np.array_equal(x, y), (tag, n, float(np.abs(x.astype(np.float32) - y.astype(np.float32)).max()))
    return worst


def test_fast_equals_old():
    base = moe.MoECfg(E=40, k=4, group=128, cap_align=0)
    for dt in (jnp.float32, jnp.bfloat16):
        ref, fill = _run(replace(base, fast=False), "a2a", dt)
        assert 0.0 < fill <= 1.0, fill
        for impl in ("a2a", "ppermute"):
            for K in (1, 2):
                got, _ = _run(replace(base, fast=True, a2a_chunks=K), impl, dt)
                u = _same(ref, got, (str(dt), impl, K), reorder=() if K == 1 else ("g_up", "g_down"))
                print(f"    {jnp.dtype(dt).name} {impl} K={K}: решта побітово; g_up/g_down max|Δ|/max|g|={u:.1e}")
        print(f"  {jnp.dtype(dt).name}: fast==old побітово (a2a/ppermute, K=1; K=2 — крім dW експертів); moe_fill={fill:.3f}")


if __name__ == "__main__":
    test_fast_equals_old()
    print("OK test_fast_equals_old")
