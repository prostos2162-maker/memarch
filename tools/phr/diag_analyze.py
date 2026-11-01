"""Де суперник кращий: різниця похибки на кожному слові (ми − він) за розрізами."""
import numpy as np, torch
import sys; a = torch.load(sys.argv[1] if len(sys.argv) > 1 else "_tmp/phr/tok_ours.pt").numpy(); b = torch.load("_tmp/phr/tok_nanogpt.pt").numpy()
print("ми", a.mean().round(4), "суперник", b.mean().round(4), "різниця", (a - b).mean().round(4), "слів", len(a))
t = np.load("fineweb.npy", mmap_mode="r")[:2_000_000].astype(np.int64)
g = torch.Generator(device="cuda").manual_seed(0)
xs, ys = [], []
for _ in range(20):
    st = torch.randint(0, len(t) - 1024 - 1, (8,), device="cuda", generator=g).cpu().numpy()
    for s in st:
        xs.append(t[s:s + 1024]); ys.append(t[s + 1:s + 1025])
x, y = np.concatenate(xs), np.concatenate(ys)
pos = np.tile(np.arange(1024), len(x) // 1024)
cnt = np.bincount(np.load("fineweb.npy", mmap_mode="r")[2_000_000:].astype(np.int64), minlength=50304)
d = a - b
def show(name, keys, labels):
    print(f"\n{name}:")
    tot = d.sum()
    for k, l in zip(keys, labels):
        m = k
        print(f"  {l:28s} частка слів {m.mean():6.1%}  ми {a[m].mean():.3f}  він {b[m].mean():.3f}  різниця {d[m].mean():+.3f}  частка розриву {d[m].sum() / tot:6.1%}")
fq = cnt[y]
show("частота цільового слова", [fq >= 1e5, (fq >= 1e4) & (fq < 1e5), (fq >= 1e3) & (fq < 1e4), (fq >= 100) & (fq < 1e3), fq < 100],
     ["дуже часті (≥100k)", "часті (10k–100k)", "середні (1k–10k)", "рідкі (100–1k)", "дуже рідкі (<100)"])
show("позиція у вікні", [pos < 16, (pos >= 16) & (pos < 128), (pos >= 128) & (pos < 512), pos >= 512],
     ["0–15 (майже без контексту)", "16–127", "128–511", "512–1023 (довгий контекст)"])
ph = torch.load("_tmp/phr/bigram_d512.pt")
E, H, lp = ph["emb"], ph["head"], ph["logp_next"]
En = E / E.pow(2).mean(1, keepdim=True).sqrt().clamp(min=1e-6)
xx = torch.tensor(np.minimum(x, 50256)); yy = torch.tensor(np.minimum(y, 50256))
lo = (0.05 * (En[xx] * H[yy]).sum(-1) + lp[yy]).numpy()
lse = ph["lse"][xx].numpy()
sp = lse - lo                                           # здивування розмовника (−log p пари)
show("наскільки слово передбачає розмовник пар", [sp < 2, (sp >= 2) & (sp < 5), (sp >= 5) & (sp < 8), sp >= 8],
     ["легко (<2)", "середньо (2–5)", "важко (5–8)", "дуже важко (≥8) — якорі"])
rep = np.zeros(len(x), bool)
for w in range(len(x) // 1024):
    seen = set(); o = w * 1024
    for i in range(1024):
        rep[o + i] = y[o + i] in seen; seen.add(x[o + i])
show("чи слово вже було у вікні (копіювання з контексту)", [rep, ~rep], ["вже траплялось у вікні", "нове у вікні"])
