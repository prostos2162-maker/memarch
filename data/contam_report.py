"""Деконтамінація, крок 2: з розріджених пар `.contam.npz` (decontam.py) — які документи справді містять задачу з бенчмарку.
Правило: документ брудний щодо задачі, якщо покриває ≥ min(FRAC × 13-грам задачі, CAP) і ≥ MIN різних її 13-грам.
Шаблонний рядок (`if __name__ == "__main__":`, умова «Write a function…») дає 1–2 збіги на задачу — не проходить.
Вихід: `<шматок>.cdrop.npy` (uint32, документи для пропуску) + звіт по джерелах і бенчмарках.
  python data/contam_report.py --dir <тека з *.contam.npz, підтеки за джерелами>"""
import argparse, glob, os
import numpy as np

FRAC, CAP, MIN = 0.5, 40, 5
HERE = os.path.dirname(os.path.abspath(__file__))


def dirty(doc, gram, b):
    """Номери брудних документів і задачі, які вони містять."""
    if len(doc) == 0:
        return np.zeros(0, np.uint32), np.zeros(0, np.uint32)
    item = b["item"][gram]
    key = np.unique(doc.astype(np.uint64) << np.uint64(32) | gram.astype(np.uint64))      # різні 13-грами
    d, it = (key >> np.uint64(32)).astype(np.uint32), b["item"][(key & np.uint64(0xFFFFFFFF)).astype(np.int64)]
    pk, cnt = np.unique(d.astype(np.uint64) << np.uint64(32) | it.astype(np.uint64), return_counts=True)
    pd, pi = (pk >> np.uint64(32)).astype(np.uint32), (pk & np.uint64(0xFFFFFFFF)).astype(np.uint32)
    need = np.maximum(np.minimum(FRAC * b["item_grams"][pi], CAP), MIN)
    ok = cnt >= need
    return pd[ok], pi[ok]


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--dir", required=True)
    a = p.parse_args()
    b = dict(np.load(os.path.join(HERE, "bench13.npz")))
    names = b["names"]
    per_src, per_bench = {}, np.zeros(len(names), np.int64)
    hit_items = set()
    for f in sorted(glob.glob(os.path.join(a.dir, "*", "*.contam.npz"))):
        z = np.load(f)
        pd, pi = dirty(z["doc"], z["gram"], b)
        drop = np.unique(pd)
        np.save(f.replace(".contam.npz", ".cdrop.npy"), drop)
        s = os.path.basename(os.path.dirname(f))
        t = per_src.setdefault(s, [0, 0]); t[0] += int(z["n_docs"]); t[1] += len(drop)
        hit_items |= set(pi.tolist())
        np.add.at(per_bench, b["item_src"][pi], 1)
    for s, (n, d) in sorted(per_src.items()):
        print(f"{s}: {n} док., брудних {d} ({d / max(n, 1):.3%})")
    items = np.array(sorted(hit_items), dtype=np.int64)
    tot = np.bincount(b["item_src"], minlength=len(names))
    hit = np.bincount(b["item_src"][items], minlength=len(names)) if len(items) else np.zeros(len(names), np.int64)
    for k, n in enumerate(names):
        print(f"  {n}: задач знайдено в корпусі {hit[k]} з {tot[k]} ({hit[k] / max(tot[k], 1):.1%}), док.-збігів {per_bench[k]}")
