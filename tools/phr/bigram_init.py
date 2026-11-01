"""«Народжені з готовими словосполученнями»: з тренувальних даних наперед рахуємо, що після чого йде
(PPMI пар «слово → наступне слово»), розкладаємо в d вимірів і зберігаємо як початкові таблиці входу й виходу.
python _tmp/phr/bigram_init.py D [токенів]"""
import sys, time, numpy as np, torch
D = int(sys.argv[1]); N = int(sys.argv[2]) if len(sys.argv) > 2 else 100_000_000
import os
V = int(os.environ.get('PHR_V', 50257)); DATA = os.environ.get('DATA', 'fineweb.npy'); TAG = os.environ.get('TAG', '')
t0 = time.time()
a = np.load(DATA, mmap_mode="r")[int(2_000_000 * (len(np.load(DATA, mmap_mode="r")) / 40_000_000 if DATA != "fineweb.npy" else 1)):][:N].astype(np.int64)   # лише тренувальна частина
pair = a[:-1] * V + a[1:]
u, c = np.unique(pair, return_counts=True)
print("пар", len(u), f"{time.time() - t0:.0f}с", flush=True)
prev, nxt = u // V, u % V
cp = np.bincount(a[:-1], minlength=V).astype(np.float64)
cn = np.bincount(a[1:], minlength=V).astype(np.float64)
pmi = np.log(c * len(pair) / (cp[prev] * cn[nxt]))
keep = (pmi > 0) & (c >= 2)
idx = torch.tensor(np.stack([prev[keep], nxt[keep]]), device="cpu")
M = torch.sparse_coo_tensor(idx, torch.tensor(pmi[keep], dtype=torch.float32, device="cpu"), (V, V)).coalesce()
print("PPMI ненульових", int(keep.sum()), flush=True)
U, S, Vt = torch.svd_lowrank(M, q=D, niter=4)
E = (U * S.sqrt()).cpu()          # слово як «попереднє»
H = (Vt * S.sqrt()).cpu()         # слово як «наступне»
torch.save({"emb": E, "head": H, "S": S.cpu(), "logp_next": torch.tensor(np.log((cn + 1) / cn.sum()), dtype=torch.float32)},
           f"_tmp/phr/bigram{TAG}_d{D}.pt")
print("готово", f"{time.time() - t0:.0f}с", "S[:5]", S[:5].tolist(), flush=True)
