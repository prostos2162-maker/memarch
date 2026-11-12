"""Неблокуючі ckpt у train_tpu: kill під час завантаження і resume. Лише CPU, сховище — локальна тека (file://).

Прогони мають Idle-пріоритет. C — без переривань, еталон втрат. A — копія кроку 8 проходить, копія кроку 16 висить
посередині; A вбивається, коли навчання вже записало ckpt ≥24 локально. Потім:
  B1 (впав лише процес): resume з локальної теки, втрати = C;
  B2 (втрачено машину): resume зі сховища — має взяти крок 8 (останній із COMPLETE), а не недокопійований 16; втрати = C.
Запуск: cd D:\\ai\\memarch_jax && PYTHONIOENCODING=utf-8 python tests/test_ckpt_kill.py"""
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import numpy as np                    # noqa: E402
import zstandard as zstd              # noqa: E402

from memarch_jax import train_tpu as TT   # noqa: E402

EOS = 4000
IDLE = 0x40 if os.name == "nt" else 0


def fixture(root):
    d = os.path.join(root, "syn")
    os.makedirs(d)
    g = np.random.default_rng(0)
    parts = []
    for _ in range(3000):
        doc = np.minimum(g.zipf(1.3, int(g.integers(300, 1500))), EOS - 2).astype(np.uint32)
        parts += [doc, np.asarray([EOS], np.uint32)]
    with open(os.path.join(d, "p0.u32.zst"), "wb") as f:
        f.write(zstd.ZstdCompressor().compress(np.concatenate(parts).tobytes()))


def cmd(tmp, local, log, extra=()):
    return [sys.executable, "-m", "memarch_jax.train_tpu", "--model", "small", "--seq", "256", "--batch", "1", "--cpu8",
            "--remat", "moe_save_heavy", "--uroboros", "--steps", "40", "--stop_at", "36", "--warmup", "4",
            "--mix", '{"syn": 1.0}', "--data_root", os.path.join(tmp, "data"), "--cache", os.path.join(tmp, "cache"),
            "--eos", str(EOS), "--heldout_tok", "20000", "--eval_every", "0", "--log_every", "4", "--ckpt_every", "8",
            "--ckpt_local", local, "--loss_log", log, "--no_final_ckpt", *extra]


def env(**kw):
    e = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONDONTWRITEBYTECODE="1", JAX_PLATFORMS="cpu", CUDA_VISIBLE_DEVICES="")
    e.update({k: str(v) for k, v in kw.items()})
    return e


def run(c, e, name):
    r = subprocess.run(c, cwd=ROOT, env=e, capture_output=True, text=True, encoding="utf-8", creationflags=IDLE)
    print(f"--- {name}: exit {r.returncode}")
    if r.returncode:
        print(r.stdout[-3000:], r.stderr[-3000:])
        raise SystemExit(name)
    return r.stdout


def test_uploader_unit():
    """Найновіший перемагає; теки, що копіюються, закріплені; збій копії не чіпає старе."""
    class S:
        def __init__(self):
            self.ev, self.done, self.fail = threading.Event(), [], set()

        def is_complete(self, s):
            return False

        def upload(self, local, s):
            self.ev.wait(20)
            if s in self.fail:
                return "boom"
            self.done.append(s)
            return ""

        def delete_older(self, s):
            self.done.append(("del", s))

    st = S()
    u = TT.Uploader(st)
    t0 = time.perf_counter()
    u.start("x1", 1)
    while u.current is None:                  # копія 1 уже в роботі: далі черга заміщується
        time.sleep(0.001)
    t0 = time.perf_counter()
    for s in (2, 3):
        u.start(f"x{s}", s)
    dt = time.perf_counter() - t0
    time.sleep(0.2)
    assert u.pinned() == {1, 3}, u.pinned()
    st.fail.add(1)
    st.ev.set()
    u.wait()
    assert st.done == [3, ("del", 3)], st.done
    assert u.pinned() == set()
    print(f"unit: 3 start() за {dt * 1e3:.2f} мс при зайнятій копії; 2 пропущено, збій копії 1 не видалив нічого")
    assert dt < 0.1


def main():
    test_uploader_unit()
    tmp = tempfile.mkdtemp(prefix="ckptkill_")
    try:
        os.makedirs(os.path.join(tmp, "data"))
        fixture(os.path.join(tmp, "data"))
        # --- C: еталон
        ref_log = os.path.join(tmp, "C.jsonl")
        out = run(cmd(tmp, os.path.join(tmp, "Cl"), ref_log), env(), "C")
        ref = [json.loads(x) for x in open(ref_log)]
        assert [r["step"] for r in ref] == list(range(1, 37))
        # --- A: копія кроку ≥16 висить, kill
        store = os.path.join(tmp, "store")
        local = os.path.join(tmp, "Al")
        a_log = os.path.join(tmp, "A.jsonl")
        c = cmd(tmp, local, a_log, ("--ckpt_gcs", "file://" + store))
        p = subprocess.Popen(c, cwd=ROOT, env=env(CKPT_TEST_UPLOAD_DELAY=600, CKPT_TEST_DELAY_FROM=16), stdout=subprocess.PIPE,
                             stderr=subprocess.STDOUT, text=True, encoding="utf-8", creationflags=IDLE)
        lines, stalls, killed_at = [], [], None
        t_kill = time.time() + 900
        for ln in p.stdout:
            lines.append(ln)
            m = re.search(r"ckpt step (\d+): цикл стояв ([\d.]+)", ln)
            if m:
                stalls.append((int(m[1]), float(m[2])))
            m = re.search(r"ckpt step (\d+) → .*запис у фоні", ln)
            if m and int(m[1]) >= 24 or time.time() > t_kill:
                killed_at = int(m[1]) if m else None
                p.kill()
                break
        p.wait()
        print("A log tail:\n" + "".join(lines[-6:]))
        assert killed_at is not None, "A не дійшла до ckpt 24: " + "".join(lines[-30:])
        trained = len(open(a_log).readlines())
        sdirs = sorted(os.listdir(store))
        print(f"A вбито на ckpt {killed_at}, кроків записано {trained}, знімки циклу {stalls}; сховище {sdirs}; локально "
              f"{sorted(os.listdir(local))}")
        st = TT.FileStore("file://" + store)
        assert st.steps() == [8], st.steps()
        assert os.path.isdir(os.path.join(store, "step_16")) and not st.is_complete(16), "копія 16 не висить"
        assert killed_at >= 24 and trained >= 24, "навчання стояло під час копії"
        loc = TT._ckpt_dirs(local)
        assert loc and loc[-1] >= 24, loc
        assert 16 in loc, "тека, що копіюється, видалена"
        # --- B1: упав лише процес
        l1 = os.path.join(tmp, "B1l")
        shutil.copytree(local, l1)
        b1 = run(cmd(tmp, l1, os.path.join(tmp, "B1.jsonl"), ("--resume", "--compare", ref_log)), env(), "B1 (local)")
        print([x for x in b1.splitlines() if "resume з" in x or "RESUME_TEST" in x])
        assert "RESUME_TEST PASS" in b1
        # --- B2: втрачено машину
        l2 = os.path.join(tmp, "B2l")
        b2 = run(cmd(tmp, l2, os.path.join(tmp, "B2.jsonl"), ("--resume", "--compare", ref_log, "--ckpt_gcs", "file://" + store)),
                 env(), "B2 (store)")
        print([x for x in b2.splitlines() if "resume з" in x or "RESUME_TEST" in x])
        assert re.search(r"resume з .*step_8: step 8", b2), b2[-2000:]
        assert "RESUME_TEST PASS" in b2
        print("OK")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
