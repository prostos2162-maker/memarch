r"""M12: комбінований тест — УСІ стовпи разом (induct lock+IN+MIX+POS, surprise in-graph, якорі/tau, MoE, пам'ять,
reread, легка голова, короткий словник) + Instinct, W=1, fp32, SR вимкнено. 20 кроків JAX (step.make_train_step, одна
jit) проти torch (full2.FullLM2 + exp_opt.InstinctOptB): та сама ініціалізація (convert), ті самі батчі, той самий lr.
Порівнюємо криву loss, члени, стати, tau, s/stake/k Instinct і розбіжність ваг щокроку.
Запуск: cd memarch_jax && PYTHONIOENCODING=utf-8 python tests/test_combined.py"""
import os
import sys
import tempfile

import numpy as np

V, K, CH, D, T, B, STEPS, LR = 10240, 512, 1024, 128, 256, 1, 20, 2e-3   # V >= 10000: torch бачить таблиці як vocab
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
                  MOE_DISP="onehot", MOE_K="4", MOE_ACT="relu2", REREAD_SEP="1", REREAD_ANCH="0.25", REREAD_IDS="1",
                  OPT_HS_LAG="1", OPT_SR_OFF="1", OPT_FN="1", OPT_FUSED="1")
for k in ("BF16_ALL", "MEM_BF16", "RESID_BF16", "HEAVY_ATTN", "CKPT", "PALLAS_OPT"):
    os.environ.pop(k, None)
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.dirname(_HERE))
import torch                                                     # noqa: E402
torch.save(torch.from_numpy(LT), os.environ["LOCK_TRUST"])
import _mem_ref                                                  # noqa: E402  (шлях до the torch code root, stable_torch)
import full2                                                     # noqa: E402
import torch.distributed as _dist                                # noqa: E402
if not _dist.is_initialized():
    _dist.init_process_group("gloo", init_method="tcp://127.0.0.1:29541", rank=0, world_size=1)
import exp_opt                                                   # noqa: E402
import jax                                                       # noqa: E402
import jax.numpy as jnp                                          # noqa: E402
from memarch_jax import config, convert, model, instinct, step as S   # noqa: E402

jax.config.update("jax_default_matmul_precision", "highest")
MEM_KW = dict(fresh=8, dim=12, dim_max=24, write=8, focus=4)
CFG = config.small(d=D, head_dim=64, E=8, heavy=2, light_a=1, light_b=1, final=1, window=64, seg=64, V=V,
                   short_v=K, ce_chunk=CH, T=T, B=B, dtype="f32", cap_align=0, moe_group=512,
                   mem_fresh=8, mem_dim=12, mem_dim_max=24, mem_write=8, mem_focus=4, mem_eb=False)
ICFG = instinct.InstinctCfg(lr_peak=LR, weight_decay=0.1, scale=0.45, m_bf16=False, rel_every=2, hs_lag=True,
                            sr=False, use_pallas=False)
NOISE = ("memory.importance", "memory.gate")      # grad ~1e-8..1e-9 (шум порядку сумування) → знак кроку випадковий


def build_torch(seed=0):
    torch.manual_seed(seed)
    m = full2.FullLM2(V, d=D, light_a=1, light_b=1, heavy=2, final=1, heads=D // 64, window=64, seg=64,
                      moe="fixed", n_exp=8, bank=True, mem_kw=MEM_KW)
    with torch.no_grad():
        for n, p in m.named_parameters():
            if n.endswith("norm.weight") or "norm_" in n or n.endswith(("n1.weight", "n2.weight")):
                p.add_(torch.randn_like(p) * 0.1)
        m.ind_in_g.add_(torch.randn(3) * 0.5)
        m.tau.fill_(7.5)
    return m.train()


def make_xy(seed):
    g = np.random.default_rng(seed)
    xy = np.minimum(g.zipf(1.3, (B, T + 1)), V - 2).astype(np.int64)
    for b in range(B):
        for _ in range(12):
            L = g.integers(3, 12)
            s = g.integers(0, T // 2)
            d_ = g.integers(L, T + 1 - s - L)
            xy[b, s + d_:s + d_ + L] = xy[b, s:s + L]
    return xy


def jname(tn):
    """torch-ім'я параметра → ім'я в instinct.make_spec (складені heavy — heavy.{i}.{k})."""
    top = {"emb.weight": "emb", "book.weight": "book", "head.weight": "head", "norm.weight": "norm",
           "l_norm.weight": "l_norm", "l_head.0.weight": "l_head0", "l_head.1.weight": "l_head1"}
    if tn in top:
        return top[tn]
    p = tn.split(".")
    if p[0] in ("light_a", "light_b", "final"):
        return f"{p[0]}.{p[1]}.{p[2]}"
    if p[0] == "heavy":
        return f"heavy.{p[1]}.{p[2]}"
    return tn


def convert_name(tn):
    return "memory." + tn[len("memory."):] if tn.startswith("memory.") else tn


def zero_noise_grads(g):
    """grad_sync-хук для jax: обнулити шумові листки (memory.importance.*, memory.gate.*)."""
    return dict(g, memory={k: (jnp.zeros_like(v) if f"memory.{k}".startswith(NOISE) else v)
                           for k, v in g["memory"].items()})


def torch_step(m, opt, xy, lr, zero_noise=False):
    x, y = torch.from_numpy(xy[:, :-1]), torch.from_numpy(xy[:, 1:])
    with _mem_ref.stable_torch():
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
    if zero_noise:                                               # шумові grads (~1e-8) — нуль, як у jax grad_sync
        for n, q in m.named_parameters():
            if q.grad is not None and convert_name(n).startswith(NOISE):
                q.grad.zero_()
    with torch.no_grad():
        sit = m.emb(torch.from_numpy(xy[:, :-1])).float().mean((0, 1))   # ситуація за таблицею ДО кроку
    opt.param_groups[0]["lr"] = lr
    opt.prepare(loss.detach(), sit)
    opt.update_all()
    opt.finish()
    m.zero_grad(set_to_none=True)
    st = {k: float(v) for k, v in m.stats.items() if k in ("mix_g", "trust", "mem_cards", "anchors", "moe_drop",
                                                           "rare_drop")}
    terms = dict(main=float(main), aux=float(aux), light=float(light), cost=float(cost.float()))
    return float(loss), terms, st, float(m.tau), dict(s=float(opt.s), stake=float(opt.stake), k=int(opt.k))


def _flat(tree):
    return {instinct.path_str(p): np.asarray(x, np.float64) for p, x in jax.tree_util.tree_flatten_with_path(tree)[0]}


def param_div(p_j, m):
    """max по листках |w_jax − w_torch| / max|w_torch|: (усі, без NOISE, найгірший листок без NOISE)."""
    pt, _, _ = convert.torch_to_jax({k: v.detach() for k, v in m.state_dict().items()}, jnp.float32)
    fj, ft = _flat(p_j), _flat(pt)
    worst_all, worst_real, wn = 0.0, 0.0, ""
    for n in ft:
        r = np.abs(fj[n] - ft[n]).max() / max(np.abs(ft[n]).max(), 1e-12)
        worst_all = max(worst_all, r)
        if not n.startswith(NOISE) and r > worst_real:
            worst_real, wn = r, n
    return worst_all, worst_real, wn


def run(zero_noise):
    m = build_torch()
    sd = {k: v.detach().clone() for k, v in m.state_dict().items()}
    tparams = [(n, p) for n, p in m.named_parameters() if p.requires_grad]
    opt = exp_opt.InstinctOptB([p for _, p in tparams], lr=LR, weight_decay=0.1, scale=0.45, norm="hs", m_bf16=False,
                               rel_every=2)
    params, frozen, tau = convert.torch_to_jax(sd, jnp.float32)
    consts = model.build_consts(CFG, COUNTS, LT)
    consts["lock_w"], consts["ind_a"] = frozen["lock_w"], frozen["ind_a"]
    names = [jname(n) for n, _ in tparams]
    fz = tuple(n for n in names if n in ("lock_w", "ind_a"))
    spec = instinct.make_spec(params, order=names, frozen=fz)
    for (n, p), e in zip(tparams, spec):                         # види кроку: torch (від форми) == jax (від імені)
        if e[5] is None:
            continue
        tk, twd, tv = opt._spec(p)
        assert (tk, twd, tv) == (e[1], e[2], e[3]), (n, (tk, twd, tv), e[1:4])
    print(f"P={len(spec)} (frozen заглушки: {fz}); види кроку torch == jax")
    st = S.init_state(params, spec, ICFG, tau=float(tau))
    coll = S.Collectives(grad_sync=zero_noise_grads) if zero_noise else S.Collectives()
    ts = S.make_train_step(CFG, consts, spec, ICFG, remat="none", coll=coll)
    print(f"== zero_noise={zero_noise}")
    rows, worst = [], dict(loss=0.0, terms=0.0, stats=0.0, tau=0.0, s=0.0, stake=0.0, k=0, w_real=0.0, w_all=0.0)
    print(f"{'step':>4} {'loss torch':>11} {'loss jax':>11} {'|dl|/l':>9} {'tau t':>9} {'|dtau|':>8} "
          f"{'s t':>9} {'|ds|':>8} {'k':>3}  max|dw|/max|w| real (leaf) / all")
    for t in range(STEPS):
        xy = make_xy(100 + t)
        lr = instinct.lr_at(t, LR, STEPS, warmup=5)
        tl, tt, tst, ttau, topt = torch_step(m, opt, xy, lr, zero_noise)
        st, mt = ts(st, S.make_batch(xy, CFG, consts, lr))
        jl = float(mt["loss"])
        dl = abs(jl - tl) / abs(tl)
        dterm = max(abs(float(mt["terms"][k]) - tt[k]) / max(abs(tt[k]), 1e-6) for k in tt)
        dst = max([abs(float(mt["stats"][k]) - tst[k]) / max(1.0, abs(tst[k])) for k in tst if k in mt["stats"]]
                  or [0.0])
        dtau = abs(float(st["tau"]) - ttau)
        ds, dsk = abs(float(mt["s"]) - topt["s"]), abs(float(mt["stake"]) - topt["stake"])
        kk = int(mt["k"]) == topt["k"]
        w_all, w_real, wn = param_div(st["params"], m)
        print(f"{t:4d} {tl:11.6f} {jl:11.6f} {dl:9.2e} {ttau:9.5f} {dtau:8.1e} {topt['s']:+9.5f} {ds:8.1e} "
              f"{'ok' if kk else 'X':>3}  {w_real:9.2e} ({wn}) / {w_all:9.2e}  terms {dterm:.1e} stats {dst:.1e}")
        rows.append((tl, jl))
        for k_, v in (("loss", dl), ("terms", dterm), ("stats", dst), ("tau", dtau), ("s", ds), ("stake", dsk),
                      ("w_real", w_real), ("w_all", w_all)):
            worst[k_] = max(worst[k_], v)
        worst["k"] += 0 if kk else 1
    print("max divergence over", STEPS, "steps:",
          {k: (f"{v:.3e}" if isinstance(v, float) else v) for k, v in worst.items()})
    print(f"loss torch {rows[0][0]:.4f} -> {rows[-1][0]:.4f};  jax {rows[0][1]:.4f} -> {rows[-1][1]:.4f}")
    return worst


def test_combined():
    """(1) шумові листки обнулено в обох — перевірка (порогове); (2) як є — лише звіт: Instinct нормує крок, тож
    grad ~1e-8 з різним порядком сумування дає ±lr у memory.importance → різні рішення пам'яті → розбіжність росте."""
    w = run(True)
    assert w["k"] == 0, "картка Instinct розійшлась"
    # поріг — з прогону 2026-11-05: loss 7.1e-5, tau 1.5e-4, ваги 3.3e-2 (memory.focus_gate, малі grads під
    # нормованим кроком); перші 4 кроки — 1e-7 (teacher-forced grads збігаються ≤1e-3 щокроку, див. decisions)
    assert w["loss"] < 3e-4 and w["tau"] < 1e-3, w
    assert w["w_real"] < 0.1, w
    run(False)


if __name__ == "__main__":
    test_combined()
    print("OK")
