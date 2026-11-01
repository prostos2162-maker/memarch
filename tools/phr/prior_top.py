"""«Слова на язиці»: для кожного контексту (попереднє слово; пара слів) — топ-32 продовження з тренувальних даних
і їхній PMI (наскільки ймовірніше за звичайну частоту). Модель додає це до своїх логітів на виході."""
import numpy as np, torch
V, KTOP = 50257, 32
a = np.load("fineweb.npy", mmap_mode="r")[2_000_000:].astype(np.int64)
cn = np.bincount(a, minlength=V).astype(np.float64)
lpu = np.log((cn + 1) / cn.sum())

def top_table(ctx_rows, nxt, n_rows):
    key = ctx_rows * V + nxt
    u, c = np.unique(key, return_counts=True)
    r, n = u // V, u % V
    tot = np.bincount(r, weights=c, minlength=n_rows)
    pmi = np.log(c / tot[r]) - lpu[n]
    score = np.where(c >= 2, pmi, -1e9)
    order = np.lexsort((-score, r))
    r, n, s = r[order], n[order], score[order]
    start = np.searchsorted(r, np.arange(n_rows))
    rank = np.arange(len(r)) - start[r]
    m = rank < KTOP
    idx = np.zeros((n_rows, KTOP), dtype=np.int64); val = np.zeros((n_rows, KTOP), dtype=np.float32)
    idx[r[m], rank[m]] = n[m]; val[r[m], rank[m]] = np.maximum(s[m], 0)
    return torch.tensor(idx, dtype=torch.int32), torch.tensor(val, dtype=torch.float16)

bi_idx, bi_val = top_table(a[:-1], a[1:], V)
tg = torch.load("_tmp/phr/trigram_k262144.pt")
keys = tg["keys"].numpy(); K = len(keys)
ctx = a[:-2] * V + a[1:-1]
pos = np.searchsorted(keys, ctx).clip(max=K - 1); hit = keys[pos] == ctx
tri_idx, tri_val = top_table(pos[hit], a[2:][hit], K)
torch.save({"bi_idx": bi_idx, "bi_val": bi_val, "tri_idx": tri_idx, "tri_val": tri_val, "keys": tg["keys"]}, "_tmp/phr/prior_top32.pt")
print("ok", float(bi_val.float().mean()), float(tri_val.float().mean()))
