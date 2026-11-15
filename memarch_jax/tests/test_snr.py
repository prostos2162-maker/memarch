"""Перевірки патча «впевненість кроку» (InstinctCfg.snr). Запуск: python tests/test_snr.py (з D:i\memarch_jax)."""
import os, sys, importlib.util, dataclasses
_HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.dirname(_HERE))
import numpy as np, jax, jax.numpy as jnp
from memarch_jax import instinct as I
spec_o = importlib.util.spec_from_file_location("instinct_orig", r"<local path>")  # оригінал до 13.11 (opt_snr)
O = importlib.util.module_from_spec(spec_o); spec_o.loader.exec_module(O)

def _tree(rng, L=3, E=4, d=32, h=40):
    mk = lambda *s: jnp.asarray(rng.standard_normal(s) * 0.1, jnp.bfloat16)
    p = {"heavy": {"norm": mk(L, d), "router": mk(L, E, d), "up": mk(L, E, d, h), "down": mk(L, E, h, d)}, "w": mk(24, 16),
         "emb": mk(50, 16), "light_b": {"0": {"qkv": mk(48, 16)}}}
    return p

def test_off_is_bitwise_original():
    rng = np.random.default_rng(1); p = _tree(rng)
    g = jax.tree_util.tree_map(lambda a: jnp.asarray(rng.standard_normal(a.shape), jnp.float32), p)
    for pal in (False, True):
        for jit in (False, True):
            ca, cb = I.InstinctCfg(rel_every=1, use_pallas=pal), O.InstinctCfg(rel_every=1, use_pallas=pal)
            sp = I.make_spec(p)
            sa, sb = I.init(p, 8, ca, sp), O.init(p, 8, cb, sp)
            fa = jax.jit(I.step, static_argnames=("spec", "cfg")) if jit else I.step
            fb = jax.jit(O.step, static_argnames=("spec", "cfg")) if jit else O.step
            pa = pb = p
            for t in range(3):
                sit = jnp.asarray(rng.standard_normal(8), jnp.float32)
                pa, sa = fa(pa, g, sa, lr=1e-3, situation=sit, loss=jnp.float32(3 - t), spec=sp, cfg=ca)
                pb, sb = fb(pb, g, sb, lr=1e-3, situation=sit, loss=jnp.float32(3 - t), spec=sp, cfg=cb)
            for a, b in zip(jax.tree_util.tree_leaves(pa), jax.tree_util.tree_leaves(pb)):
                assert np.array_equal(np.asarray(a, np.float32), np.asarray(b, np.float32))
            assert set(sa) == set(sb)
    print("OK snr=False побітово = оригінал (jnp і Pallas-interpret, eager і jit)")

def test_on_stacked_and_factor():
    rng = np.random.default_rng(2); p = _tree(rng)
    sp = I.make_spec(p); names = [s[0] for s in sp]
    for pal in (False, True):
        cfg = I.InstinctCfg(rel_every=1, use_pallas=pal, snr=True)
        st = I.init(p, 8, cfg, sp); js = jax.jit(I.step, static_argnames=("spec", "cfg"))
        g_sig = jax.tree_util.tree_map(lambda a: jnp.asarray(rng.standard_normal(a.shape), jnp.float32), p)
        pp = p
        for t in range(60):                           # сталий градієнт → множник → ~1
            pp, st = js(pp, g_sig, st, lr=1e-3, situation=jnp.ones(8), loss=jnp.float32(3.0), spec=sp, cfg=cfg)
        f_sig = np.asarray(st["snr_f"])
        st2 = I.init(p, 8, cfg, sp); pp = p
        for t in range(60):                           # чистий шум → ~0.23
            g_n = jax.tree_util.tree_map(lambda a: jnp.asarray(rng.standard_normal(a.shape), jnp.float32), p)
            pp, st2 = js(pp, g_n, st2, lr=1e-3, situation=jnp.ones(8), loss=jnp.float32(3.0), spec=sp, cfg=cfg)
        f_n = np.asarray(st2["snr_f"])
        i = names.index("light_b.0.qkv")
        print(f"pallas={pal}: множник сталий сигнал {f_sig[i]:.3f} (мін {f_sig.min():.3f}), шум {f_n[i]:.3f} (сер {f_n.mean():.3f})")
        assert f_sig.min() > 0.9 and 0.15 < f_n[i] < 0.35
    print("OK snr=True: складені шари під jit, Pallas-interpret, множник сигнал≈1 / шум≈0.23")

if __name__ == "__main__":
    test_off_is_bitwise_original(); test_on_stacked_and_factor(); print("SNR_OK")
