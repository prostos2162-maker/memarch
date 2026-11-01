"""Словосполучення з двох слів → третє: для K найчастіших пар (попереднє, поточне) наперед рахуємо PPMI з наступним
словом і проєктуємо на ту саму вихідну таблицю H (найменші квадрати), щоб вектор пари можна було додати до входу.
python _tmp/phr/trigram_init.py K"""
import sys, time, numpy as np, torch
K = int(sys.argv[1]); V = 50257
t0 = time.time()
a = np.load("fineweb.npy", mmap_mode="r")[2_000_000:102_000_000].astype(np.int64)
ctx = (a[:-3] * V + a[1:-2]) * V + a[2:-1]
uc, cc = np.unique(ctx, return_counts=True)
top = uc[np.argsort(-cc)[:K]]
keys = np.sort(top)
print("пар-контекстів", len(uc), "беремо", K, "покривають", f"{cc[np.argsort(-cc)[:K]].sum() / len(ctx):.1%}", f"{time.time() - t0:.0f}с", flush=True)
pos = np.searchsorted(keys, ctx)
hit = (pos < K) & (keys[np.minimum(pos, K - 1)] == ctx)
row, nxt = pos[hit], a[3:][hit]
trip = row * V + nxt
ut, ct = np.unique(trip, return_counts=True)
r, n = ut // V, ut % V
crow = np.bincount(row, minlength=K).astype(np.float64)
cn = np.bincount(a[3:], minlength=V).astype(np.float64)
pmi = np.log(ct * len(a) / (crow[r] * cn[n]))
keep = (pmi > 0) & (ct >= 2)
ph = torch.load("_tmp/phr/bigram_d512.pt")
H = ph["head"].double()                                    # (V, d)
M = torch.sparse_coo_tensor(torch.tensor(np.stack([r[keep], n[keep]])), torch.tensor(pmi[keep]), (K, V)).coalesce()
P = torch.linalg.solve(H.T @ H + 1e-3 * torch.eye(H.shape[1], dtype=H.dtype), (M @ H).T).T     # (K, d): v·Hᵀ ≈ PMI
torch.save({"keys": torch.tensor(keys), "vec": P.float()}, f"_tmp/phr/quad_k{K}.pt")
print("готово", f"{time.time() - t0:.0f}с", "норма", P.norm(dim=1).mean().item(), flush=True)
