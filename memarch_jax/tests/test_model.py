"""M7: паритет повної моделі (forward + loss + усі grads) memarch_jax.model проти torch full2.FullLM2, W=1, CPU.

Мала конфігурація: d=256, head_dim=64, E=8, heavy=2, light 1/1/1, window=64, seg=64, V=4096, SHORT_V=512, T=512.
Прапори як у цілі: INDUCT=lock(+IN,+MIX,+POS), SURP_INGRAPH, CE_CHUNK, SHORT_V, MOE_DISP=onehot MOE_K=4 MOE_ACT=relu2,
REREAD_SEP + REREAD_ANCH=0.25, ROPE_HALF, BLK=relu2,qkv3, LIGHT_FFN=2.
Запуск: cd memarch_jax && PYTHONIOENCODING=utf-8 python tests/test_model.py"""
import copy
import os
import sys
import tempfile

import numpy as np

V, K, CH, D, T = 4096, 512, 1000, 256, 512
_tmp = tempfile.mkdtemp()
_r = np.random.default_rng(0)
COUNTS = np.floor(1e6 / np.arange(1, V + 1) ** 1.1 * _r.uniform(0.5, 1.5, V)).astype(np.int64)
COUNTS[V - 1] = 0
np.save(os.path.join(_tmp, "cnt.npy"), COUNTS)
LT = (_r.uniform(1.0, 4.0, (3, 7))).astype(np.float32)
os.environ.update(DEVICE="cpu", NO_COMPILE="1", ATTN_IMPL="block", BLK="relu2,qkv3", LIGHT_FFN="2", ROPE_HALF="1",
                  CE_CHUNK=str(CH), SURP_INGRAPH="1", SHORT_V=str(K), RARE_CAP="0.3",
                  INDUCT="lock", INDUCT_IN="1", INDUCT_MIX="1", INDUCT_POS="1",
                  TOKEN_COUNTS=os.path.join(_tmp, "cnt.npy"), LOCK_TRUST=os.path.join(_tmp, "lt.pt"),
                  MOE_DISP="onehot", MOE_K="4", MOE_ACT="relu2", REREAD_SEP="1", REREAD_ANCH="0.25", REREAD_IDS="1")
for k in ("BF16_ALL", "MEM_BF16", "RESID_BF16", "HEAVY_ATTN", "CKPT"):
    os.environ.pop(k, None)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch                                                     # noqa: E402
torch.save(torch.from_numpy(LT), os.environ["LOCK_TRUST"])
import _mem_ref                                                  # noqa: E402  (stable_torch, XLA-прапор)
import full2                                                     # noqa: E402
import jax                                                       # noqa: E402
import jax.numpy as jnp                                          # noqa: E402
from memarch_jax import config, convert, model                   # noqa: E402

jax.config.update("jax_default_matmul_precision", "highest")
MEM_KW = dict(fresh=8, dim=12, dim_max=24, write=8, focus=4)


def cfg_of(dtype, B):
    return config.small(d=D, head_dim=64, E=8, heavy=2, light_a=1, light_b=1, final=1, window=64, seg=64, V=V,
                        short_v=K, ce_chunk=CH, T=T, B=B, dtype=dtype, cap_align=0, moe_group=512,
                        mem_fresh=8, mem_dim=12, mem_dim_max=24, mem_write=8, mem_focus=4, mem_eb=dtype == "bf16")


def build_torch(seed=0):
    torch.manual_seed(seed)
    m = full2.FullLM2(V, d=D, light_a=1, light_b=1, heavy=2, final=1, heads=D // 64, window=64, seg=64,
                      moe="fixed", n_exp=8, bank=True, mem_kw=MEM_KW)
    with torch.no_grad():                                        # ворушимо одиничні норми й ворота — щоб grads були «живі»
        for n, p in m.named_parameters():
            if n.endswith("norm.weight") or "norm_" in n or n.endswith(("n1.weight", "n2.weight")):
                p.add_(torch.randn_like(p) * 0.1)
        m.ind_in_g.add_(torch.randn(3) * 0.5)
        m.tau.fill_(7.5)                                         # біля медіани здивування: якорі ≠ «усі»
    return m.train()


def make_xy(B, seed=1):
    """Слова за Zipf + повтори фраз (щоб спрацьовували замки 1/2/3 і суміш)."""
    g = np.random.default_rng(seed)
    xy = np.minimum(g.zipf(1.3, (B, T + 1)), V - 2).astype(np.int64)
    for b in range(B):
        for _ in range(12):
            L = g.integers(3, 12)
            s = g.integers(0, T // 2)
            d_ = g.integers(L, T + 1 - s - L)
            xy[b, s + d_:s + d_ + L] = xy[b, s:s + L]
    return xy


def torch_run(m, xy, bf16):
    full2.MEM_BF16 = full2.RESID_BF16 = bf16
    x, y = torch.from_numpy(xy[:, :-1]), torch.from_numpy(xy[:, 1:])
    tau0 = float(m.tau)
    with _mem_ref.stable_torch(), torch.autocast("cpu", dtype=torch.bfloat16, enabled=bf16):
        out, cost = m(x)
        loss = m.train_loss(out, y, True) + 0.01 * cost.float()
        hn, lh = out
        with torch.no_grad():
            main = m._ce_part(hn.reshape(*y.shape, -1), y, 0) / y.numel()
            aux = sum(b.aux for b in m.heavy) / len(m.heavy)
            h4 = m.l_head[0](lh[:, ::4])
            lse, ly = full2.chunk_ce(h4.reshape(-1, h4.shape[-1]), m.l_head[1].weight, None, y[:, ::4].reshape(-1))
            light = (lse - ly).mean()
    loss.backward()
    terms = dict(main=float(main), aux=float(aux), light=float(light), cost=float(cost.float()))
    stats = {k: float(v) for k, v in m.stats.items() if k in ("mix_g", "trust", "mem_cards", "anchors", "moe_drop",
                                                              "rare_drop")}
    stats["tau"] = float(m.tau)
    grads = {n: p.grad for n, p in m.named_parameters() if p.grad is not None}
    nograd = sorted(n for n, p in m.named_parameters() if p.grad is None)
    return float(loss), terms, stats, grads, nograd, tau0


def jax_run(sd, xy, cfg):
    params, frozen, _ = convert.torch_to_jax(sd, cfg.cdt)
    consts = model.build_consts(cfg, COUNTS, LT)
    consts["lock_w"], consts["ind_a"] = frozen["lock_w"], frozen["ind_a"]
    x = xy[:, :-1]
    prep = model.make_prep(x, cfg, consts)
    tau = jnp.asarray(float(sd["tau"]), jnp.float32)
    f = lambda p: model.train_loss(p, consts, jnp.asarray(xy, jnp.int32), prep, tau, cfg)
    (loss, (tau_n, terms, stats)), g = jax.value_and_grad(f, has_aux=True)(params)
    stats = {k: float(v) for k, v in stats.items()}
    stats["tau"] = float(tau_n)
    return float(loss), {k: float(v) for k, v in terms.items()}, stats, g, params


def _flat(tree, pre=""):
    out = {}
    if isinstance(tree, dict):
        for k, v in tree.items():
            out.update(_flat(v, f"{pre}{k}."))
    elif isinstance(tree, list):
        for i, v in enumerate(tree):
            out.update(_flat(v, f"{pre}{i}."))
    else:
        out[pre[:-1]] = np.asarray(jnp.asarray(tree, jnp.float32), np.float64)
    return out


def rel(a, b):
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    m = float(np.abs(a - b).max()) if a.size else 0.0
    return m, m / max(float(np.abs(b).max()), 1e-30)


def compare(tag, B, bf16, tol_loss, tol_grad, tol_stat):
    cfg = cfg_of("bf16" if bf16 else "f32", B)
    m = build_torch()
    if bf16:
        m = m.bfloat16()
        m.tau.data = m.tau.data.float()
    sd = {k: v.detach().clone() for k, v in m.state_dict().items()}
    xy = make_xy(B)
    tl, tt, ts, tg, nograd, _ = torch_run(m, xy, bf16)
    jl, jt, js, jg, params = jax_run(sd, xy, cfg)
    print(f"== {tag} (B={B}) torch loss {tl:.6f}  jax loss {jl:.6f}")
    bad = []
    a, r = rel(jl, tl)
    print(f"  {'total':34s} abs={a:.3e} rel={r:.3e}")
    bad += [("total", r)] if r > tol_loss else []
    for k in tt:
        a, r = rel(jt[k], tt[k])
        print(f"  term {k:29s} torch={tt[k]:+.6e} abs={a:.3e} rel={r:.3e}")
        bad += [(k, r)] if r > tol_loss else []
    for k in ts:
        a, r = rel(js[k], ts[k])
        print(f"  stat {k:29s} torch={ts[k]:+.6e} jax={js[k]:+.6e} abs={a:.3e}")
        bad += [(k, a)] if a > tol_stat * max(1, abs(ts[k])) else []
    print("  no-grad у torch:", nograd)
    assert {"lock_w", "ind_a"} <= set(nograd), nograd
    gj = _flat(convert.grads_to_jax({n: g for n, g in tg.items()}, jnp.float32))
    fj = _flat(jg)
    missing = sorted(set(fj) - set(gj))
    for k in missing:                                            # нема torch-grad → JAX має дати 0
        z = float(np.abs(fj[k]).max())
        print(f"  grad {k:29s} (torch None) jax max|g|={z:.3e}")
        bad += [(k, z)] if z > 0 else []
    worst = []
    for k in sorted(gj):
        a, r = rel(fj[k], gj[k])
        worst.append((r, k, a))
        bad += [("grad " + k, r)] if r > tol_grad and a > 1e-7 else []   # крихітні grads (|g|~1e-8): лише abs
    for r, k, a in sorted(worst, reverse=True):
        print(f"  grad {k:29s} abs={a:.3e} rel={r:.3e}")
    print(f"  grads: {len(worst)} тензорів, max rel {max(worst)[0]:.3e}")
    return bad


def test_f32_B1():
    bad = compare("fp32", 1, False, 1e-4, 1e-2, 1e-4)
    assert not bad, bad


def test_f32_B2():
    bad = compare("fp32", 2, False, 1e-4, 1e-2, 1e-4)
    assert not bad, bad


def test_init_shapes():
    """init_params: та сама структура/форми, що конвертований torch; loss скінченний; head[V-1]=0."""
    cfg = cfg_of("f32", 1)
    p_t, _, _ = convert.torch_to_jax(build_torch().state_dict())
    p_j = model.init_params(jax.random.PRNGKey(0), cfg, LT)
    st = jax.tree_util.tree_structure
    assert st(p_t) == st(p_j), (st(p_t), st(p_j))
    ft, fj = _flat(p_t), _flat(p_j)
    assert set(ft) == set(fj)
    for k in ft:
        assert ft[k].shape == fj[k].shape, (k, ft[k].shape, fj[k].shape)
    assert float(jnp.abs(p_j["head"][V - 1]).max()) == 0
    consts = model.build_consts(cfg, COUNTS, LT)
    xy = make_xy(1)
    l, _ = model.train_loss(p_j, consts, jnp.asarray(xy, jnp.int32), model.make_prep(xy[:, :-1], cfg, consts),
                            jnp.float32(5.0), cfg)
    print("  init loss", float(l))
    assert np.isfinite(float(l))


def test_bf16_B1_report():
    compare("bf16 (лише звіт)", 1, True, 1e9, 1e9, 1e9)


if __name__ == "__main__":
    for nm, fn in list(globals().items()):
        if nm.startswith("test_") and callable(fn):
            fn()
            print("OK", nm)
    print("ALL OK")
