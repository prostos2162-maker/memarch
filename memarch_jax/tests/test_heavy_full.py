"""Перевірка «повного» важкого шару (heavy_stack_full, ідея Стефана 11.11) на CPU, мала модель.
1) full_idx=-1 ≈ старий heavy_stack (той самий результат з точністю до округлення set/add);
2) full_idx=i == ручний еталон без cond (якорі → шар i на всіх словах → якорі);
3) all_full == послідовні bank_moe на всіх словах;
4) повний шар міняє НЕ-якорі, без нього — ні; градієнти скінченні й доходять до ваг повного шару через не-якорі.
Запуск: JAX_PLATFORMS=cpu python tests/test_heavy_full.py"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np                                   # noqa: E402
import jax                                           # noqa: E402
import jax.numpy as jnp                              # noqa: E402

from memarch_jax import config, model, moe          # noqa: E402

jax.config.update("jax_enable_x64", False)
cfg = config.small(T=256, B=2)
p = model.init_params(jax.random.PRNGKey(0), cfg)
ps = p["heavy"]
mc = model.moe_cfg(cfg)
L = jax.tree.leaves(ps)[0].shape[0]
B, T, d = cfg.B, cfg.T, cfg.d
g = np.random.default_rng(0)
h = jnp.asarray(g.normal(0, 1, (B, T, d)), jnp.float32)
s_ = jnp.asarray(np.abs(g.normal(3, 2, (B, T))), jnp.float32)
A = max(1, round(T * cfg.cap))
idx = jnp.asarray(np.stack([np.sort(g.choice(T, A, replace=False)) for _ in range(B)]), jnp.int32)
valid = jnp.asarray(g.random((B, A)) < 0.8)
anchor_mask = np.zeros((B, T), bool)
for b in range(B):
    anchor_mask[b, np.asarray(idx[b])[np.asarray(valid[b])]] = True
print(f"L={L} B={B} T={T} A={A} d={d}")

old, _, _ = moe.heavy_stack(ps, h, idx, valid, s_, mc)
new_m1, _, _ = moe.heavy_stack_full(ps, h, idx, valid, s_, mc, full_idx=jnp.int32(-1))
e1 = float(jnp.abs(old - new_m1).max())
print("1) full_idx=-1 vs старий:", e1)
assert e1 < 1e-4

layer = lambda i: jax.tree.map(lambda l: l[i], ps)
fcfg = moe._full_cfg(mc)
bi = jnp.arange(B)[:, None]
sa = jnp.where(valid, jnp.clip(jnp.take_along_axis(s_, idx, 1), 0, 50), -1.0)


def ref(i_full):
    hh, a = h, jnp.take_along_axis(h, idx[..., None], 1)
    for i in range(L):
        if i == i_full:
            cur = jnp.take_along_axis(hh, idx[..., None], 1)
            hh = hh.at[bi, idx].set(jnp.where(valid[..., None], a, cur))
            hh = moe.bank_moe(layer(i), hh, jnp.clip(s_, 0, 50), fcfg)[0]
            a = jnp.take_along_axis(hh, idx[..., None], 1)
        else:
            a = moe.bank_moe(layer(i), a, sa, mc)[0]
    cur = jnp.take_along_axis(hh, idx[..., None], 1)
    return hh.at[bi, idx].set(jnp.where(valid[..., None], a, cur))


for i in (0, L - 1):
    got, _, _ = moe.heavy_stack_full(ps, h, idx, valid, s_, mc, full_idx=jnp.int32(i))
    e = float(jnp.abs(got - ref(i)).max())
    ch = np.asarray(jnp.abs(got - h).max(-1) > 0)
    print(f"2) full_idx={i}: vs еталон {e:.2e}; змінено не-якорів {int((ch & ~anchor_mask).sum())} з {int((~anchor_mask).sum())}")
    assert e < 1e-5 and (ch & ~anchor_mask).sum() == (~anchor_mask).sum()
ch = np.asarray(jnp.abs(new_m1 - h).max(-1) > 0)
print("   без повного шару змінено не-якорів:", int((ch & ~anchor_mask).sum()))
assert (ch & ~anchor_mask).sum() == 0

allf, _, _ = moe.heavy_stack_full(ps, h, idx, valid, s_, mc, all_full=True)
hh = h
for i in range(L):
    hh = moe.bank_moe(layer(i), hh, jnp.clip(s_, 0, 50), fcfg)[0]
e3 = float(jnp.abs(allf - hh).max())
print("3) all_full vs послідовні bank_moe:", e3)
assert e3 < 1e-5

nonanch = jnp.asarray(~anchor_mask, jnp.float32)[..., None]
w = jnp.asarray(g.normal(0, 1, (B, T, d)), jnp.float32)


def lossf(ps_, fi):
    o, aux, _ = moe.heavy_stack_full(ps_, h, idx, valid, s_, mc, full_idx=fi, remat=True)
    return (o * w * nonanch).sum()


for fi in (-1, 1):
    gr = jax.grad(lossf)(ps, jnp.int32(fi))
    gn = [float(jnp.abs(x[1]).sum()) for x in jax.tree.leaves(gr)]
    fin = all(bool(jnp.isfinite(x).all()) for x in jax.tree.leaves(gr))
    print(f"4) full_idx={fi}: градієнт ваг шару 1 через не-якорі = {sum(gn):.3e}, скінченні {fin}")
    assert fin and ((sum(gn) > 0) == (fi == 1))
print("HEAVY_FULL_OK")
