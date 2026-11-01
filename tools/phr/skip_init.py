"""Чернетка наперед: що зазвичай стоїть через o слів після цього слова (o = 2, 3, 4). PPMI(слово_t, слово_{t+o})
проєктується на ту саму вихідну таблицю H (як словосполучення), щоб підказку можна було додати до входу."""
import numpy as np, torch
V = 50257
a = np.load("fineweb.npy", mmap_mode="r")[2_000_000:].astype(np.int64)
ph = torch.load("_tmp/phr/bigram_d512.pt")
H = ph["head"].double()
cn = np.bincount(a, minlength=V).astype(np.float64)
out = {}
for o in (2, 3, 4):
    pair = a[:-o] * V + a[o:]
    u, c = np.unique(pair, return_counts=True)
    p, n = u // V, u % V
    pmi = np.log(c * len(pair) / (cn[p] * cn[n]))
    keep = (pmi > 0) & (c >= 3)
    M = torch.sparse_coo_tensor(torch.tensor(np.stack([p[keep], n[keep]])), torch.tensor(pmi[keep]), (V, V)).coalesce()
    P = torch.linalg.solve(H.T @ H + 1e-3 * torch.eye(H.shape[1], dtype=H.dtype), (M @ H).T).T
    out[o] = P.float()
    print("o", o, "пар", int(keep.sum()), "норма", P.norm(dim=1).mean().item(), flush=True)
torch.save(out, "_tmp/phr/skip_d512.pt")
print("готово")
