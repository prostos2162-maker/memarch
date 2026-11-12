"""Фішки (Стефан 11.11) у моделі, CPU fp32, мала модель — перевірка коду, не прогін.
(1) W=1: loss скінченний; градієнт мають рівно фішки з комбінації кроку (решта — нуль) і роутер; інша комбінація → інший loss.
(2) W=8 (shard_map, ваги фішок порізані по d і збираються all_gather) на однакових даних == W=1: loss і градієнти.
Запуск: PYTHONIOENCODING=utf-8 python tests/test_fishki.py"""
import os
import sys

os.environ["XLA_FLAGS"] = os.environ.get("XLA_FLAGS", "") + " --xla_force_host_platform_device_count=8"
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np                                   # noqa: E402
import jax                                           # noqa: E402
import jax.numpy as jnp                              # noqa: E402
from jax.sharding import PartitionSpec as P          # noqa: E402

import test_dist as TD                               # noqa: E402
from memarch_jax import dist, model, fishki          # noqa: E402

TD.CFG = __import__("dataclasses").replace(TD.CFG, fish_n=12, fish_hf=256, fish_kinds=3, fish_pool=5, fish_c=2)
CFG = TD.CFG


def prep_fish(prep, B, order, kind):
    return dict(prep, fish=jnp.asarray(np.tile(np.asarray(order, np.int32), (B, 1))),
                kind=jnp.full((B, 1), kind, jnp.int32))


def run_w1(params, consts, xy, order, kind):
    prep = prep_fish(model.make_prep(xy[:, :-1], CFG, consts), xy.shape[0], order, kind)
    f = lambda p: model.train_loss(p, consts, jnp.asarray(xy, jnp.int32), prep, jnp.float32(TD.TAU), CFG)
    (l, (tn, terms, st)), g = jax.jit(jax.value_and_grad(f, has_aux=True))(params)
    return float(l), g, {k: float(v) for k, v in terms.items()}, float(st["fish_router_acc"])


def main():
    params, consts = TD.setup()
    assert "heavy" not in params and params["fish"]["up"].shape == (12, CFG.d, 256)
    pools = fishki.pools(CFG)
    print("набори:", pools)
    xy = TD.make_xy(1, 1)
    order = pools[1][:2]
    l1, g1, terms, acc = run_w1(params, consts, xy, order, 1)
    gu = np.abs(np.asarray(g1["fish"]["up"])).sum((1, 2))
    print(f"(1) loss {l1:.6f} terms {terms}; |grad up| по фішках: {np.round(gu, 4).tolist()}")
    assert np.isfinite(l1)
    assert all((gu[i] > 0) == (i in order) for i in range(12)), "градієнт не там"
    assert float(np.abs(np.asarray(g1["fish_router"])).sum()) > 0
    l1b = run_w1(params, consts, xy, pools[2][:2], 2)[0]
    assert abs(l1b - l1) > 1e-6
    print(f"    інша комбінація → loss {l1b:.6f}")
    mesh = dist.make_mesh(TD.W)
    xy8 = np.tile(xy, (TD.W, 1))
    prep8 = prep_fish(model.make_prep(xy8[:, :-1], CFG, consts), TD.W, order, 1)
    ps = dist.put(params, mesh, dist.param_specs(params))
    gf = dist.make_grad_fn(CFG, consts, mesh, params)
    (l8, (tn, terms8, st, per, sit)), g8 = gf(ps, dist.put(jnp.asarray(xy8, jnp.int32), mesh, P("x")),
                                              dist.put(prep8, mesh, P("x")), jnp.float32(TD.TAU))
    print(f"(2) loss W=1 {l1:.8f}  W=8 {float(l8):.8f}")
    assert abs(float(l8) - l1) <= 1e-5 * abs(l1)
    bad = TD.cmp_grads(g8, g1, 2e-3)
    assert not bad, bad
    print("FISHKI_OK")


if __name__ == "__main__":
    main()
