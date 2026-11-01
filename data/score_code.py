"""Оцінка якості коду для шматків The Stack v3 дешевою лінійною моделлю (копія Stack-Edu, `_tmp/cq_train.py`):
для кожного шматка — файл `<шматок>.score.npy` (float16, одна оцінка 0–5 на документ-репозиторій, у порядку документів)
поруч на Диску. Тексти не розкодовуються — лише частоти токенів. Зазвичай запускається з data/postproc.py.
python data/score_code.py [--model cq_linear.npz] [--remote remote:corpus/tokens/stack_v3] [--worker i --workers n]"""
import argparse, os, subprocess
import numpy as np, zstandard as zstd

EOS = 131072                                                # <|endoftext|> у qwen36_pruned131k
MODEL = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cq_linear.npz")


def load_model(path=MODEL):
    m = np.load(path)
    V, H = int(m["V"]), int(m["H"])
    cu, cb, cl, b0 = m["coef"][:V], m["coef"][V:V + H], float(m["coef"][V + H]), float(m["intercept"])

    def score_doc(d):
        n = max(len(d), 1)
        u, cu_n = np.unique(d, return_counts=True)
        s = float(np.sqrt(cu_n / n) @ cu[u])
        bh = (d[:-1].astype(np.int64) * 1_000_003 + d[1:]) % H
        bu, bc = np.unique(bh, return_counts=True)
        return s + float(np.sqrt(bc / n) @ cb[bu]) + cl * np.log(n) / 10 + b0
    return score_doc


def score_shard(arr, score_doc):
    assert arr[-1] == EOS, "шматок має закінчуватись <|endoftext|>"
    ends = np.flatnonzero(arr == EOS)
    starts = np.concatenate([[0], ends[:-1] + 1])
    return np.asarray([score_doc(arr[s:e]) for s, e in zip(starts, ends)], dtype=np.float16)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model", default=MODEL)
    p.add_argument("--remote", default="remote:corpus/tokens/stack_v3")
    p.add_argument("--worker", type=int, default=0)
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--workdir", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "_tmp", "score_code"))
    a = p.parse_args()
    score_doc = load_model(a.model)
    os.makedirs(a.workdir, exist_ok=True)
    names = sorted(l.strip() for l in subprocess.run(["rclone", "lsf", a.remote, "--include", "*.u32.zst"],
                                                      capture_output=True, text=True).stdout.splitlines() if l.strip())
    have = set(subprocess.run(["rclone", "lsf", a.remote, "--include", "*.score.npy"], capture_output=True, text=True).stdout.split())
    for nm in names[a.worker::a.workers]:
        out = nm.replace(".u32.zst", ".score.npy")
        if out in have:
            continue
        loc = os.path.join(a.workdir, nm)
        subprocess.run(["rclone", "copy", f"{a.remote}/{nm}", a.workdir, "-q"], check=True)
        arr = np.frombuffer(zstd.ZstdDecompressor().decompress(open(loc, "rb").read(), max_output_size=8_000_000_000),
                            dtype=np.uint32)
        sc = score_shard(arr, score_doc)
        np.save(os.path.join(a.workdir, out), sc)
        subprocess.run(["rclone", "move", os.path.join(a.workdir, out), a.remote, "-q"], check=True)
        os.remove(loc)
        print(f"{nm}: {len(sc)} документів, ≥3.0 — {np.mean(sc >= 3):.0%}, ≥2.5 — {np.mean(sc >= 2.5):.0%}", flush=True)
