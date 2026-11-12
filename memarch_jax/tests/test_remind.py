"""Нагадування (--remind, порт RECALL_REMIND з memarch/rope_recall.py). Лише CPU.
1) Reminders: форма незмінна, маска = ключі-питання, ціль там = значення, відстань ≥ dist, детерміновано.
2) Прапорець вимкнено = біт у біт як раніше: втрати train_tpu без --remind == втрати копії коду ДО порту
   (OLD_PKG — тека з пакетом memarch_jax без правок; якщо не задана — лише порівняння «без прапорця» vs «--remind 0»).
3) Прапорець увімкнено: та сама програма (рекомпіляцій 0), втрата скінченна, remind_acc у eval.
Запуск: cd D:\\ai\\memarch_jax && PYTHONIOENCODING=utf-8 python tests/test_remind.py"""
import json
import os
import re
import subprocess
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import numpy as np                    # noqa: E402
import zstandard as zstd              # noqa: E402
from memarch_jax import data          # noqa: E402

EOS = 4000
IDLE = 0x40 if os.name == "nt" else 0


def test_unit():
    R = data.Reminders(1.0, 256, n=4, dist=64, tok0=1000, seed=3, V=4096)
    xy = np.random.default_rng(0).integers(0, 999, (16, 257)).astype(np.int32)
    xr, ex = R(xy)
    xr2, ex2 = R(xy)
    assert xr.shape == xy.shape and xr.dtype == xy.dtype and ex["rem"].shape == (16, 256)
    assert (xr == xr2).all() and (ex["rem"] == ex2["rem"]).all()
    for b in range(16):
        qp = np.nonzero(ex["rem"][b])[0]
        assert len(qp) == 4
        fp = np.nonzero(xr[b] == R.mark)[0]
        assert len(fp) == 4                                   # дві мітки фактів, дві мітки питань
        facts = dict(zip(xr[b, fp[0] + 1:fp[1]:2], xr[b, fp[0] + 2:fp[1]:2]))
        for t in qp:
            assert xr[b, t + 1] == facts[xr[b, t]]            # ціль y[t] = значення цього ключа
        assert fp[2] - fp[1] >= 64
    xo, eo = R(xy, p=0.0)
    assert (xo == xy).all() and eo["rem"].sum() == 0
    print("unit OK")


def fixture(root):
    d = os.path.join(root, "data", "syn")
    os.makedirs(d)
    g = np.random.default_rng(0)
    parts = []
    for _ in range(3000):
        parts += [np.minimum(g.zipf(1.3, int(g.integers(300, 1500))), EOS - 2).astype(np.uint32),
                  np.asarray([EOS], np.uint32)]
    with open(os.path.join(d, "p0.u32.zst"), "wb") as f:
        f.write(zstd.ZstdCompressor().compress(np.concatenate(parts).tobytes()))


def run(tmp, tag, extra=(), pkg_root=ROOT):
    log = os.path.join(tmp, f"{tag}.jsonl")
    c = [sys.executable, "-m", "memarch_jax.train_tpu", "--model", "small", "--seq", "256", "--batch", "1", "--cpu8",
         "--rope_mem", "--steps", "8", "--warmup", "2", "--mix", '{"syn": 1.0}', "--data_root", os.path.join(tmp, "data"),
         "--cache", os.path.join(tmp, "cache"), "--eos", str(EOS), "--heldout_tok", "20000", "--eval_every", "4",
         "--log_every", "2", "--ckpt_every", "0", "--ckpt_local", os.path.join(tmp, "ck_" + tag), "--no_final_ckpt",
         "--loss_log", log, *extra]
    e = dict(os.environ, PYTHONIOENCODING="utf-8", JAX_PLATFORMS="cpu", CUDA_VISIBLE_DEVICES="", PYTHONPATH=pkg_root,
             PYTHONDONTWRITEBYTECODE="1")
    r = subprocess.run(c, cwd=pkg_root, env=e, capture_output=True, text=True, encoding="utf-8", creationflags=IDLE)
    if r.returncode:
        print(r.stdout[-3000:], r.stderr[-3000:])
        raise SystemExit(tag)
    with open(log) as f:
        return [json.loads(x)["loss"] for x in f], r.stdout


def test_runs():
    tmp = tempfile.mkdtemp(prefix="remind_")
    fixture(tmp)
    off, _ = run(tmp, "off")
    zero, _ = run(tmp, "zero", ("--remind", "0"))
    assert off == zero, (off, zero)
    old = os.environ.get("OLD_PKG")
    if old:
        ref, _ = run(tmp, "old", (), old)
        assert ref == off, f"вимкнено ≠ старий код: {ref} vs {off}"
        print(f"off == старий код побітово: {off}")
    on, out = run(tmp, "on", ("--remind", "0.5", "--remind_q", "4"))
    assert all(np.isfinite(on)), on
    res = json.loads(out.split("RESULT ", 1)[1].splitlines()[0])
    assert res["recompiles"] == 0, res
    acc = re.findall(r"RECALL step \d+ .*", out)
    assert len(acc) == 2 and "nomem" in acc[0], out[-2000:]
    print(f"on: втрати {[round(x, 4) for x in on]}, рекомпіляцій {res['recompiles']}, remind_acc {acc}")
    print("runs OK")




def test_learns(steps=150):
    """Механізм: мала модель на фіксованому батчі з голкою в кожному рядку (q=4, відстань > вікна 64).
    Точність на відповідях (p>0.5) і nll на відповідь мають рости/падати: градієнт доходить через маску rem."""
    import dataclasses
    import jax
    from memarch_jax import config, model, instinct, step as S
    V = 1024
    cfg = config.small(V=V, short_v=256, ce_chunk=512, T=256, d=128, head_dim=64, E=8, window=64, seg=64)
    cfg = dataclasses.replace(cfg, remind=True, remind_q=4)
    r = np.random.default_rng(0)
    counts = np.floor(1e6 / np.arange(1, V + 1) ** 1.1 * r.uniform(0.5, 1.5, V)).astype(np.int64)
    counts[V - 1] = 0
    lt = r.uniform(1.0, 4.0, (3, 7)).astype(np.float32)
    consts = model.build_consts(cfg, counts, lt)
    params = model.init_params(jax.random.PRNGKey(0), cfg, lt)
    spec = instinct.make_spec(params)
    icfg = instinct.InstinctCfg(use_pallas=False, sr=False)
    R = data.Reminders(1.0, cfg.T, n=4, dist=96, tok0=100, seed=0, V=V)
    xy, ex = R(r.integers(0, 99, (4, cfg.T + 1)).astype(np.int32))
    b = S.make_batch(xy, cfg, consts, 3e-3)
    b["prep"]["rem"] = jax.numpy.asarray(ex["rem"])
    ts = S.make_train_step(cfg, consts, spec, icfg, remat="none")
    st = S.init_state(params, spec, icfg)
    hist = []
    for i in range(steps):
        st, m = ts(st, b)
        if i % 25 == 0 or i == steps - 1:
            sts = m["stats"]
            n = float(sts["remind_n"])
            hist.append((i, float(m["loss"]), float(sts["remind_nll"]) / n, float(sts["remind_hit"]) / n))
            print(f"  step {i}: loss {hist[-1][1]:.3f} remind nll/відп {hist[-1][2]:.3f} acc {hist[-1][3]:.2f}", flush=True)
    assert all(np.isfinite(h[1]) for h in hist)
    assert hist[-1][2] < hist[0][2] - 1.0 and hist[-1][3] > hist[0][3] + 0.25, hist
    print("learns OK")


def test_schedule_ladder():
    """Розклад p і драбина: детерміновані, стан відновлюється; відстані — дані (форма незмінна)."""
    R = data.Reminders(0.02, 256, n=4, dist=64, seed=1, V=4096, p0=0.5, warm=10, decay=10,
                       ladder=[16, 64, 221], thr=0.9, win=8, lag=5)
    assert R.p_at(0) == 0.5 and R.p_at(9) == 0.5 and abs(R.p_at(15) - 0.26) < 1e-9 and abs(R.p_at(25) - 0.02) < 1e-12
    assert R.range_at(0) == (0, 16)
    assert R.observe(3, 7.0, 8.0) is None                    # 0.875 < 0.9
    msg = R.observe(4, 8.0, 8.0)
    assert msg and R.level_at(8) == 0 and R.level_at(9) == 1 and R.range_at(9) == (16, 64), msg
    R2 = data.Reminders(0.02, 256, n=4, dist=64, seed=1, V=4096, p0=0.5, warm=10, decay=10,
                        ladder=[16, 64, 221], thr=0.9, win=8, lag=5)
    R2.load(json.loads(json.dumps(R.state())))
    xy = np.random.default_rng(0).integers(0, 999, (32, 257)).astype(np.int32)
    for n in (0, 9, 30):
        a, ea = R(xy, n=n)
        b, eb = R2(xy, n=n)
        assert a.shape == xy.shape and (a == b).all() and (ea["rem"] == eb["rem"]).all()
    R.observe(20, 8.0, 8.0)                                   # → рівень 2 (останній): уся [0, 221]
    assert R.range_at(40) == (0, 221)
    lo, hi = R2.range_at(9)
    xr, ex = R2(xy, p=None, n=9)
    assert ex["rem"].any(1).sum() > 5
    for b in np.nonzero(ex["rem"].any(1))[0]:
        fp = np.nonzero(xr[b] == R2.mark)[0]
        d = fp[2] - fp[1] - 1
        assert lo <= d <= hi, (d, lo, hi)
    print("schedule/ladder OK")


def test_v3_mass_grad():
    """rope_read_v3 з mgrad: градієнт mass (і out) за q,k,v = jnp-еталон того самого _tile; mgrad=False — mass без похідної."""
    import jax
    import jax.numpy as jnp
    from memarch_jax import rope_read_v3 as r3
    B, H, T, hd, K, W = 1, 2, 64, 16, 2, 8
    r = np.random.default_rng(0)
    q, k, v = (jnp.asarray(r.normal(size=(B, H, T, hd)), jnp.float32) for _ in range(3))
    kid = jnp.asarray(np.repeat(np.arange(8), 8)[None], jnp.int32)          # 8 якорів по 8 слів
    qid = jnp.asarray(np.where(r.random((B, H, T, K)) < 0.7, np.stack([r.permutation(8)[:K] for _ in range(B * H * T)])
                               .reshape(B, H, T, K), -1), jnp.int32)
    gb = jnp.asarray(r.normal(size=(B, H, T, K)) * 0.3, jnp.float32)
    tpos = jnp.zeros((B, H, T, K), jnp.int32)
    c1 = jnp.asarray(r.normal(size=(B, H, T, hd)), jnp.float32)
    c2 = jnp.asarray(r.normal(size=(B, H, T, 1)), jnp.float32)
    sc = 1 / np.sqrt(hd)

    def kern(mg):
        p = r3.P3(T, hd, K, W, 32, 32, sc, sc, mgrad=mg)
        def f(q, k, v):
            o, m = r3.read(q, k, v, kid, qid, gb, tpos, p, True, False)
            return (o * c1).sum() + (m * c2).sum()
        return jax.grad(f, argnums=(0, 1, 2))(q, k, v)

    pf = r3.P3(T, hd, K, W, T, T, sc, sc)
    def ref(q, k, v):
        tot = 0.0
        for b in range(B):
            for h in range(H):
                qids = [qid[b, h, :, kk:kk + 1] for kk in range(K)]
                gbs = [gb[b, h, :, kk:kk + 1] for kk in range(K)]
                sf, ok, win, far, _ = r3._tile(q[b, h], k[b, h], kid[b][None, :], qids, gbs, 0, 0, pf)
                pr = jax.nn.softmax(jnp.where(ok, sf, -jnp.inf), axis=1)
                o = pr @ v[b, h]
                m = jnp.where(far & ok, pr, 0.0).sum(1, keepdims=True)
                tot = tot + (o * c1[b, h]).sum() + (m * c2[b, h]).sum()
        return tot
    gr = jax.grad(ref, argnums=(0, 1, 2))(q, k, v)
    gk = kern(True)
    errs = [float(jnp.abs(a - b).max() / jnp.abs(b).max()) for a, b in zip(gk, gr)]
    g0 = kern(False)
    diff0 = float(jnp.abs(g0[0] - gr[0]).max() / jnp.abs(gr[0]).max())
    print(f"v3 mgrad: rel err dq/dk/dv {errs}; без mgrad dq відрізняється на {diff0:.3f} (mass без похідної)")
    assert max(errs) < 1e-4, errs
    assert diff0 > 1e-3


if __name__ == "__main__":
    test_unit()
    test_schedule_ladder()
    test_v3_mass_grad()
    test_runs()
    test_learns()
