"""15.11 (fable52f): відкатник не відхиляє той самий вид двічі поспіль (rb_cool).
Іграшкова втрата: вид 1 з певного кроку дає втрату ×3 («зламаний вид»). Без rb_cool (RB_NOCOOL=1) кожен крок виду 1
відхиляється → вид не вчиться; з rb_cool — кожен другий крок виду 1 застосовується.  python tests/test_rb_cool.py"""
import os, sys, types
os.environ.setdefault("JAX_PLATFORMS", "cpu")
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import numpy as np, jax, jax.numpy as jnp
from memarch_jax import instinct, step as stp


def run(nocool, n=24):
    if nocool:
        os.environ["RB_NOCOOL"] = "1"
    else:
        os.environ.pop("RB_NOCOOL", None)
    params = {"a": jax.random.normal(jax.random.PRNGKey(0), (8, 16), jnp.float32) * 0.1}
    icfg = instinct.InstinctCfg(use_pallas=False, sr=False, m_bf16=False, card_by_kind=True, cards=3, rollback=True, rb_warm=0)
    spec = instinct.make_spec(params)
    st = stp.init_state(params, spec, icfg, d_sit=16)
    st["opt"]["rb_b"] = dict(xy=jnp.zeros((2, 5), jnp.int32), prep={"kind": jnp.zeros((2, 1), jnp.int32), "boost": jnp.zeros((), jnp.float32)})

    def loss_fn(p, xy, prep, tau):
        loss = (jnp.mean(jnp.square(p["a"])) + 1.0) * prep["boost"]
        return loss, (tau, {"main": loss}, {})
    cfg = types.SimpleNamespace(remind_only=False, fish_n=0)
    coll = stp.Collectives(situation=lambda p, xy: jnp.ones((16,), jnp.float32))
    ts = stp.make_train_step(cfg, None, spec, icfg, loss_fn=loss_fn, coll=coll, donate=False)
    hits, applied1 = [], 0
    for t in range(n):
        kind = t % 2
        boost = 3.0 if (kind == 1 and t >= 6) else 1.0
        b = dict(xy=jnp.zeros((2, 5), jnp.int32), prep={"kind": jnp.full((2, 1), kind, jnp.int32), "boost": jnp.asarray(boost, jnp.float32)},
                 lr=jnp.asarray(6e-4, jnp.float32))
        st, met = ts(st, b)
        h, rep = int(float(met["rb_hit"])), int(float(met["rb_rep"]))
        hits.append(h)
        if kind == 1 and t >= 6 and not h and not rep:
            applied1 += 1
    return sum(hits), applied1


h0, a0 = run(True)
h1, a1 = run(False)
print(f"без rb_cool: відкатів {h0}, застосованих кроків зламаного виду {a0}")
print(f"з rb_cool:   відкатів {h1}, застосованих кроків зламаного виду {a1}")
assert a0 == 0 and h0 > 0, "старий відкатник мав відхиляти кожен крок зламаного виду"
assert a1 > 0 and h1 > 0 and h1 < h0, "з rb_cool зламаний вид має вчитися"
print("OK")
