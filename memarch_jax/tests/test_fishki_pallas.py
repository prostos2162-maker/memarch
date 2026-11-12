"""Фішки + Pallas-ядра Instinct у 8-чиповому кроці (CPU, interpret=True) — те, що впало на TPU 12.11 (масштаб рядка
fish.down (NF,HF,1) з шардуванням по останній осі). Крок з ядрами == крок чистим jnp (до шуму SR), розкладка стану
збережена, листки фішок пройшли через ядро шардованими.
Запуск: PYTHONIOENCODING=utf-8 python tests/test_fishki_pallas.py"""
import dataclasses
import os
import sys

os.environ["XLA_FLAGS"] = os.environ.get("XLA_FLAGS", "") + " --xla_force_host_platform_device_count=8"
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np                                               # noqa: E402
import jax                                                       # noqa: E402
import jax.numpy as jnp                                          # noqa: E402

import test_dist_pallas as TP                                    # noqa: E402
from memarch_jax import model, dist, instinct, step as S         # noqa: E402

CFG = dataclasses.replace(TP.CFG, fish_n=16, fish_hf=128, fish_kinds=3, fish_pool=5, fish_c=2)


def run(use_pallas):
    mesh = dist.make_mesh(TP.W)
    consts = model.build_consts(CFG, TP.COUNTS, TP.LT)
    params = model.init_params(jax.random.PRNGKey(0), CFG, TP.LT)
    spec = instinct.make_spec(params)
    icfg = instinct.InstinctCfg(m_bf16=True, sr=True, use_pallas=use_pallas, interpret=True)
    xy = np.random.default_rng(1).integers(0, TP.V - 1, (TP.W, TP.T + 1))
    ts, st, place = dist.make_step(CFG, consts, mesh, params, spec, icfg, remat="none")
    sh0 = jax.tree_util.tree_map(lambda x: x.sharding, st)
    p0 = jax.tree_util.tree_map(lambda x: np.asarray(x.astype(jnp.float32)), st["params"])
    b = S.make_batch(xy, CFG, consts, 3e-3)
    b["prep"] = dict(b["prep"], fish=np.tile(np.array([[5, 6]], np.int32), (TP.W, 1)),
                     kind=np.ones((TP.W, 1), np.int32))
    shapes0 = jax.tree_util.tree_map(lambda x: x.shape, st)
    st, m = ts(st, place(b))
    assert jax.tree_util.tree_map(lambda x: x.shape, st) == shapes0, "форма стану змінилась (resume зламається)"
    sh1 = jax.tree_util.tree_map(lambda x: x.sharding, st)
    assert jax.tree_util.tree_all(jax.tree_util.tree_map(lambda a, c: a == c, sh0, sh1)), "розкладка змінилась"
    p1 = jax.tree_util.tree_map(lambda x: np.asarray(x.astype(jnp.float32)), st["params"])
    return float(m["loss"]), p0, p1


def main():
    l_j, p0, pj = run(False)
    TP.CALLS.clear()
    l_p, _, pp = run(True)
    fish_calls = sorted({l for l, s in TP.CALLS if l.startswith("fish")})
    print(f"loss jnp {l_j:.6f} pallas {l_p:.6f}; ядерних викликів {len(TP.CALLS)}, листки фішок через ядро: {fish_calls}")
    assert fish_calls, "фішки не пройшли через ядро"
    assert np.isfinite(l_p) and abs(l_j - l_p) < 1e-5
    worst = 0.0
    for (path, a), b, c in zip(jax.tree_util.tree_flatten_with_path(p0)[0], jax.tree_util.tree_leaves(pj),
                               jax.tree_util.tree_leaves(pp)):
        n = instinct.path_str(path)
        if n.startswith("memory.importance") or n.startswith("memory.gate"):
            continue
        dj, dp = b - a, c - a
        step_sz = np.abs(dj).max() + 1e-12
        ulp = np.abs(a) * 2.0 ** -7 + 1e-30
        frac = (np.abs(dp - dj) > (2 * ulp + 0.05 * step_sz)).mean()
        worst = max(worst, frac)
        if frac > 1e-3:
            print(f"  {n}: частка розбіжних {frac:.4f}")
        if n.startswith("fish.up") or n.startswith("fish.down"):
            moved = np.abs(dp).reshape(dp.shape[0], -1).sum(1) > 0
            assert moved[[5, 6]].all() and not moved[[0, 1, 2]].any(), (n, np.nonzero(moved))
    print(f"найгірша частка елементів поза 2 ulp: {worst:.2e}")
    assert worst < 1e-3
    print("FISHKI_PALLAS_OK")


if __name__ == "__main__":
    main()
