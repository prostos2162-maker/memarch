"""Довіра до замків наперед: як часто слово після минулої появи замка (1/2/3 слова) збігається з наступним,
за рідкісністю поточного слова. Початкова вага = log(влучання / середня ймовірність правильного слова)."""
import numpy as np, torch
V = 50304; T = 1024
a = np.load("fineweb.npy", mmap_mode="r")[2_000_000:2_000_000 + 4_000_000].astype(np.int64)
cnt = np.bincount(np.load("fineweb.npy", mmap_mode="r")[2_000_000:].astype(np.int64), minlength=V)
bucket = np.clip(np.floor(np.log10(cnt + 1)), 0, 6).astype(np.int64)
x = torch.tensor(a[: len(a) // (T + 1) * (T + 1)]).view(-1, T + 1)
inp, tgt = x[:, :-1], x[:, 1:]
pos = torch.arange(T).expand_as(inp)
prev = torch.cat([torch.full_like(inp[:, :1], V), inp[:, :-1]], 1)
prev2 = torch.cat([torch.full_like(inp[:, :2], V), inp[:, :-2]], 1)
keys = (inp, prev * (V + 1) + inp, (prev2 * (V + 1) + prev) * (V + 1) + inp)
bk = torch.tensor(bucket)[inp]
W = np.zeros((3, 7))
for k, key in enumerate(keys):
    order = (key * T + pos).argsort(1)
    ks, ps = key.gather(1, order), pos.gather(1, order)
    same = torch.cat([torch.zeros_like(ks[:, :1], dtype=torch.bool), ks[:, 1:] == ks[:, :-1]], 1)
    lastp = torch.cat([torch.full_like(ps[:, :1], -1), ps[:, :-1]], 1)
    last = torch.full_like(inp, -1).scatter(1, ps, torch.where(same, lastp, -1))
    ok = last >= 0
    cand = inp.gather(1, (last + 1).clamp(max=T - 1))
    hit = (cand == tgt) & ok
    for b in range(7):
        m = ok & (bk == b)
        n = int(m.sum())
        p = float(hit[m].sum()) / max(n, 1)
        W[k, b] = max(0.0, np.log(max(p, 1e-4) / np.exp(-4.5))) if n > 50 else 1.0
        print(f"замок {k + 1} слово(а), рідкісність {b}: появ {n:8d}  влучань {p:6.1%}  довіра {W[k, b]:.2f}")
torch.save(torch.tensor(W, dtype=torch.float32), "_tmp/phr/lock_trust.pt")
