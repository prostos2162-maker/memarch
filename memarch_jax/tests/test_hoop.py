"""Обруч (hoop.py): вибірка, адаптивний бюджет, 8 CPU-чипів (EP all_to_all з меншою кількістю слів), навчання.

(1) select: статична форма A=budget·T, це справді top-k за ŝ, слово 0 завжди вибране, позиції зростають;
(2) next_budget: обруч лише звужується, по одному рівню;
(3) W=8 (shard_map, EP 16/8) з обручем == середнє 8 окремих W=1 (loss і градієнти, зокрема голови обруча);
(4) навчання з Instinct на синтетиці «легкі повтори + важкий шум»: голова обруча вчиться передбачати власну
    помилку (кореляція росте, пропущені слова мають меншу справжню помилку, ніж вибрані), і бюджет стискається
    (кожен рівень — своя компіляція).
Запуск: cd memarch_jax && PYTHONIOENCODING=utf-8 python tests/test_hoop.py"""
import dataclasses
import os
import sys

os.environ["XLA_FLAGS"] = os.environ.get("XLA_FLAGS", "") + " --xla_force_host_platform_device_count=8"
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np                                               # noqa: E402
import jax                                                       # noqa: E402
import jax.numpy as jnp                                          # noqa: E402

from memarch_jax import model, hoop, instinct                    # noqa: E402
import test_dist as td                                           # noqa: E402

HCFG = dataclasses.replace(td.CFG, hoop=True, hoop_budget=0.25)


def test_select():
    g = np.random.default_rng(0)
    pred = jnp.asarray(g.uniform(0, 5, (3, 64)), jnp.float32)
    idx, valid = hoop.select(pred, 0.25)
    assert idx.shape == (3, 16) and bool(valid.all())
    for b in range(3):
        i = np.asarray(idx[b])
        assert (np.diff(i) > 0).all() and i[0] == 0
        top = set(np.argsort(-np.asarray(pred[b, 1:]))[:15] + 1) | {0}
        assert set(i) == top
    print("(1) select OK")


def test_budget():
    L = (0.25, 0.18, 0.12, 0.08)
    assert hoop.next_budget(0.25, 0.0, L) == 0.25          # нічого не з'їдено — тримаємо
    assert hoop.next_budget(0.25, 0.9, L) == 0.18          # багато з'їдено — на один рівень униз
    assert hoop.next_budget(0.18, 0.95, L) == 0.12
    assert hoop.next_budget(0.08, 0.99, L) == 0.08         # дно
    assert hoop.next_budget(0.12, 0.5, L) == 0.12          # ніколи не розширюється
    assert hoop.next_budget(0.25, 0.9, L, corr=0.1) == 0.25   # голова ще не вміє — не стискаємо
    print("(2) next_budget OK")


def test_dist_hoop():
    td.CFG = HCFG                                            # w1/w8/setup з test_dist беруть модульний CFG
    mesh = td.dist.make_mesh(td.W)
    params, consts = td.setup()
    assert "hoop" in params
    xys = [td.make_xy(1, 20 + r) for r in range(td.W)]
    runs = [td.w1(params, consts, x) for x in xys]
    l8, g8, t8, terms8, per, _, _ = td.w8(params, consts, np.concatenate(xys), mesh)
    lm = np.mean([r[0] for r in runs])
    print(f"(3) обруч W=8 loss {l8:.8f}  mean(W=1) {lm:.8f}  hoop-term {terms8['hoop']:.5f}")
    assert abs(l8 - lm) <= 1e-5 * abs(lm)
    gmean = jax.tree_util.tree_map(lambda *g: sum(g) / td.W, *[r[1] for r in runs])
    bad = td.cmp_grads(g8, gmean, 2e-3)
    assert not bad, bad
    gh = np.abs(np.asarray(g8["hoop"]["w1"])).max()
    assert gh > 0, "голова обруча не отримує градієнт"


def synth(B, T, V, seed, easy=0.85):
    """Суміш: «легкі» шматки (частка easy; у тексті більшість слів передбачувані) — короткий мотив, що повторюється (передбачуваний), «важкі» — випадкові слова."""
    g = np.random.default_rng(seed)
    xy = np.zeros((B, T + 1), np.int64)
    for b in range(B):
        t = 0
        while t < T + 1:
            L = int(g.integers(16, 48))
            if g.random() < easy:
                m = g.integers(1, 64, int(g.integers(2, 5)))      # мотив з малого алфавіту
                seg = np.resize(m, L)
            else:
                seg = g.integers(64, V - 2, L)
            xy[b, t:t + L] = seg[:T + 1 - t]
            t += L
    return xy


def test_learn():
    V = td.V
    counts = td.COUNTS
    cfg = dataclasses.replace(HCFG, hoop_budget=0.25, hoop_eat=4.0, hoop_w=0.3)   # мала модель за 160 кроків не доходить до <1 нат
    params = model.init_params(jax.random.PRNGKey(3), cfg, td.LT)
    consts = model.build_consts(cfg, counts, td.LT)
    ocfg = instinct.InstinctCfg(m_bf16=False, sr=False, use_pallas=False)
    spec = instinct.make_spec(params)
    st = instinct.init(params, cfg.d, ocfg, spec)
    steps_fn = {}

    def get_step(c):
        if c.hoop_budget not in steps_fn:                    # один рівень бюджету = одна компіляція
            def f(p, s, xy, prep):
                (l, (tn, terms, stats)), gr = jax.value_and_grad(model.train_loss, has_aux=True)(
                    p, consts, xy, prep, jnp.float32(7.5), c)
                sit = p["emb"][xy[:, :-1]].astype(jnp.float32).mean((0, 1))
                p2, s2 = instinct.step(p, gr, s, lr=3e-3, situation=sit, loss=l, spec=spec, cfg=ocfg)
                return p2, s2, l, terms, stats
            steps_fn[c.hoop_budget] = jax.jit(f)
        return steps_fn[c.hoop_budget]

    hist = []
    for it in range(160):
        xy = synth(2, cfg.T, V, it)
        prep = model.make_prep(xy[:, :-1], cfg, consts)
        params, st, l, terms, stats = get_step(cfg)(params, st, jnp.asarray(xy, jnp.int32), prep)
        s = {k: float(stats[k]) for k in ("hoop_eaten", "hoop_corr", "hoop_nll_sel", "hoop_nll_skip")}
        hist.append((float(l), float(terms["hoop"]), s, cfg.hoop_budget))
        if it % 10 == 9:
            win = [h[2] for h in hist[-10:]]                 # рішення — за середнім вікна, не за одним батчем
            nb = hoop.next_budget(cfg.hoop_budget, np.mean([w["hoop_eaten"] for w in win]), cfg.hoop_levels,
                                  corr=np.mean([w["hoop_corr"] for w in win]))
            print(f"  it {it+1:3d} loss {float(l):.3f} hoopMSE {float(terms['hoop']):.3f} corr {s['hoop_corr']:+.3f} "
                  f"eaten {s['hoop_eaten']:.2f} nll sel/skip {s['hoop_nll_sel']:.2f}/{s['hoop_nll_skip']:.2f} "
                  f"budget {cfg.hoop_budget}" + (f" → {nb}" if nb != cfg.hoop_budget else ""))
            cfg = dataclasses.replace(cfg, hoop_budget=nb)
    last = hist[-10:]
    corr = np.mean([h[2]["hoop_corr"] for h in last])
    sel = np.mean([h[2]["hoop_nll_sel"] for h in last])
    skip = np.mean([h[2]["hoop_nll_skip"] for h in last])
    print(f"(4) кінець: corr {corr:+.3f}  nll вибраних {sel:.2f} > пропущених {skip:.2f}  "
          f"бюджет {hist[0][3]} → {hist[-1][3]}  компіляцій {len(steps_fn)}")
    assert hist[-1][0] < hist[0][0]
    assert corr > 0.3, corr
    assert sel > skip
    assert hist[-1][3] < hist[0][3], "обруч не стиснувся"


if __name__ == "__main__":
    print("пристроїв:", len(jax.devices()))
    test_select()
    test_budget()
    test_dist_hoop()
    test_learn()
    print("OK")
