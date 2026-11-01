"""Етап «після токенізації» (PLAN 9a) одним проходом: кожен шматок качається з Диска один раз, поруч кладуться
`.contam.npz` (decontam.py), `.mh.npy` (minhash.py) і для коду `.score.npy` (score_code.py). Готове пропускається,
тож перезапуск продовжує з місця. Перетокенізації немає.
  python data/postproc.py --worker i --workers n [--sources stack_v3,fineweb_edu,...]"""
import argparse, os, subprocess, sys, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from decontam import match, BENCH                    # noqa: E402
from minhash import signatures, load                  # noqa: E402
from score_code import load_model, score_shard        # noqa: E402

ROOT = "remote:corpus/tokens"
CODE = {"stack_v3"}                                   # де рахувати якість коду (класифікатор вчився на The Stack)

p = argparse.ArgumentParser()
p.add_argument("--worker", type=int, default=0)
p.add_argument("--workers", type=int, default=1)
p.add_argument("--sources", default="")
p.add_argument("--workdir", default=os.path.join(HERE, "..", "_tmp", "postproc"))
a = p.parse_args()
lsf = lambda path, *args: subprocess.run(["rclone", "lsf", path, *args], capture_output=True, text=True).stdout.split()
srcs = a.sources.split(",") if a.sources else sorted(d.rstrip("/") for d in lsf(ROOT, "--dirs-only") if not d.startswith("_"))
jobs = []
for src in srcs:
    files = set(lsf(f"{ROOT}/{src}"))
    for nm in sorted(f for f in files if f.endswith(".u32.zst")):
        need = [x for x in ("contam.npz", "mh.npy") + (("score.npy",) if src in CODE else ())
                if nm.replace("u32.zst", x) not in files]
        if need:
            jobs.append((src, nm, need))
print(f"шматків до обробки: {len(jobs)} (джерела: {', '.join(srcs)})", flush=True)
bh = np.load(BENCH)["h"]
score_doc = load_model()
os.makedirs(a.workdir, exist_ok=True)
for src, nm, need in jobs[a.worker::a.workers]:
    t = time.time()
    loc = os.path.join(a.workdir, nm)
    subprocess.run(["rclone", "copy", f"{ROOT}/{src}/{nm}", a.workdir, "-q"], check=True)
    arr = load(loc)
    os.remove(loc)
    msg, outs = [], []
    if "contam.npz" in need:
        d, g, n = match(arr, bh)
        o = loc.replace("u32.zst", "contam.npz"); np.savez(o, doc=d, gram=g, n_docs=n); outs.append(o)
        msg.append(f"збіги з бенчмарками в {len(np.unique(d))} док.")
    if "mh.npy" in need:
        sg = signatures(arr)
        o = loc.replace("u32.zst", "mh.npy"); np.save(o, sg); outs.append(o)
        msg.append(f"порожніх смуг {np.mean(sg == 0):.1%}")
    if "score.npy" in need:
        sc = score_shard(arr, score_doc)
        o = loc.replace("u32.zst", "score.npy"); np.save(o, sc); outs.append(o)
        msg.append(f"код ≥3.0 — {np.mean(sc >= 3):.0%}")
    del arr
    for o in outs:
        subprocess.run(["rclone", "move", o, f"{ROOT}/{src}", "-q"], check=True)
    print(f"{src}/{nm}: {', '.join(msg)}; {time.time() - t:.0f} с", flush=True)
print("готово", flush=True)
