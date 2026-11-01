"""MinHash-дедуп, крок 2: зіткнення смуг по ВСІХ джерелах -> кластери близьких копій -> у кожному лишаємо одну копію
(з джерела вищого пріоритету, далі — першу), решту позначаємо. Вихід поруч із шматком: `<шматок>.drop.npy`
(uint32, номери документів, які тренування пропускає). Запускати, коли `.mh.npy` є для всіх шматків.
  python data/dedup_cluster.py --dir <тека з *.mh.npy, підтеками за джерелами>   (скачати: rclone copy --include "*.mh.npy")
Пам'ять: ~16 байт на документ на одну смугу (смуги по черзі) + ребра."""
import argparse, glob, os
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

# менше число = важливіше лишити (кураторські/вузькі джерела перед широким вебом)
PRIORITY = ["finewiki_uk", "finewiki_en", "finewiki_de", "finewiki_es", "finewiki_pl", "finemath4", "openwebmath", "nemotron_math4",
            "finepdfs_edu_en", "fineweb_edu", "opc_fineweb_code", "stack_v3", "hplt_uk",
            "fineweb2_pol_Latn", "fineweb2_deu_Latn", "fineweb2_spa_Latn"]

p = argparse.ArgumentParser()
p.add_argument("--dir", required=True)
p.add_argument("--min-bands", type=int, default=1, help="скільки смуг мають збігтися, щоб пара вважалась копією")
a = p.parse_args()
files = sorted(glob.glob(os.path.join(a.dir, "*", "*.mh.npy")))
srcs = [os.path.basename(os.path.dirname(f)) for f in files]
pri = [PRIORITY.index(s) if s in PRIORITY else len(PRIORITY) for s in srcs]
mm = [np.load(f, mmap_mode="r") for f in files]
sizes = np.array([len(m) for m in mm], dtype=np.int64)
off = np.concatenate([[0], np.cumsum(sizes)])
total = int(off[-1])
print(f"{len(files)} шматків, {total} документів", flush=True)
rank = np.repeat(np.array(pri, dtype=np.int64), sizes) * total + np.arange(total)   # хто в кластері лишається
bands = mm[0].shape[1]
eu, ev = [], []
for b in range(bands):
    h = np.concatenate([np.asarray(m[:, b]) for m in mm])
    g = np.flatnonzero(h)
    h = h[g]
    o = np.argsort(h, kind="stable")
    h, g = h[o], g[o]
    same = np.flatnonzero(h[1:] == h[:-1])
    eu.append(g[same]); ev.append(g[same + 1])                # ланцюжок у групі однакових хешів = зв'язна група
    print(f"смуга {b}: {len(same)} зіткнень", flush=True)
    del h, g, o
u, v = np.concatenate(eu), np.concatenate(ev)
if a.min_bands > 1:                                           # пара має зустрітись у ≥ min_bands смугах
    key = np.minimum(u, v) * total + np.maximum(u, v)
    k, c = np.unique(key, return_counts=True)
    k = k[c >= a.min_bands]
    u, v = k // total, k % total
nodes, inv = np.unique(np.concatenate([u, v]), return_inverse=True)
n = len(nodes)
gr = coo_matrix((np.ones(len(u), dtype=np.int8), (inv[:len(u)], inv[len(u):])), shape=(n, n))
nc, lab = connected_components(gr, directed=False)
r = rank[nodes]
best = np.full(nc, np.iinfo(np.int64).max)
np.minimum.at(best, lab, r)
drop = np.sort(nodes[r != best[lab]])
print(f"документів у кластерах {n}, кластерів {nc}, до пропуску {len(drop)} ({len(drop) / total:.2%})", flush=True)
fi = np.searchsorted(off, drop, side="right") - 1
for i, f in enumerate(files):
    d = (drop[fi == i] - off[i]).astype(np.uint32)
    np.save(f.replace(".mh.npy", ".drop.npy"), d)
by = {}
for i, s in enumerate(srcs):
    t = by.setdefault(s, [0, 0]); t[0] += int(sizes[i]); t[1] += int((fi == i).sum())
for s, (nd, dr) in sorted(by.items()):
    print(f"  {s}: {nd} док., пропуск {dr} ({dr / max(nd, 1):.1%})")
