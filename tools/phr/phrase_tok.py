"""Словник словосполучень: N найчастіших n-грам токенів GPT-2 (2..4), що не перетинають межу документа,
стають окремими одиницями (id 50257+). Дані переписуються жадібно (найдовше словосполучення спершу).
Вихід: _tmp/phr/fw_phr.npy (uint32), _tmp/phr/phr_vocab.npy (n-грами), _tmp/phr/bytes_per_tok.txt.
python _tmp/phr/phrase_tok.py N"""
import sys, time, numpy as np
N = int(sys.argv[1]); V = 50257; EOT = 50256
t0 = time.time()
a = np.load("fineweb.npy", mmap_mode="r")
a = np.asarray(a, dtype=np.int64)
tr = a[2_000_000:2_000_000 + 60_000_000]                # частоти — лише з тренувальної частини
cands = []
for n in (2, 3, 4):
    k = np.zeros(len(tr) - n + 1, dtype=np.int64)
    ok = np.ones(len(tr) - n + 1, dtype=bool)
    for j in range(n):
        k = k * V + tr[j:len(tr) - n + 1 + j]
        ok &= tr[j:len(tr) - n + 1 + j] != EOT
    u, c = np.unique(k[ok], return_counts=True)
    top = np.argsort(-c)[:N]
    # вигода від злиття ~ (n-1) кроків на кожну появу
    cands += [(int(c[i]) * (n - 1), n, int(u[i])) for i in top]
    print(n, "унікальних", len(u), f"{time.time() - t0:.0f}с", flush=True)
cands.sort(reverse=True)
cands = cands[:N]
grams = {}
for _, n, key in cands:
    toks = []
    for _ in range(n):
        toks.append(key % V); key //= V
    grams.setdefault(n, []).append(tuple(reversed(toks)))
print("словосполучень", {n: len(g) for n, g in grams.items()}, flush=True)
# жадібне переписування: на кожній позиції — найдовше словосполучення зі словника (векторно, по n від 4 до 2)
idmap, vocab = {}, []
for n in (4, 3, 2):
    for g in grams.get(n, []):
        idmap[(n, g)] = V + len(vocab); vocab.append(g)
out = a.copy()
taken = np.zeros(len(a), dtype=bool)          # позиція вже всередині словосполучення
newid = np.full(len(a), -1, dtype=np.int64)
for n in (4, 3, 2):
    if n not in grams:
        continue
    keys = np.zeros(len(a) - n + 1, dtype=np.int64)
    for j in range(n):
        keys = keys * V + a[j:len(a) - n + 1 + j]
    gk = np.array(sorted(sum(g[j] * V ** (n - 1 - j) for j in range(n)) for g in grams[n]), dtype=np.int64)
    gid = {int(sum(g[j] * V ** (n - 1 - j) for j in range(n))): idmap[(n, g)] for g in grams[n]}
    pos = np.searchsorted(gk, keys).clip(max=len(gk) - 1)
    hit = gk[pos] == keys
    # жадібно зліва направо: беремо збіг, якщо жодна його позиція ще не зайнята
    for i in np.nonzero(hit)[0]:
        if not taken[i:i + n].any():
            taken[i:i + n] = True
            newid[i] = gid[int(keys[i])]
    print("n", n, "злито", int((newid >= 0).sum()), f"{time.time() - t0:.0f}с", flush=True)
start = newid >= 0
keep = ~taken | start
res = np.where(start, newid, a)[keep]
np.save("_tmp/phr/fw_phr.npy", res.astype(np.uint32))
np.save("_tmp/phr/phr_vocab.npy", np.array([list(g) + [-1] * (4 - len(g)) for g in vocab], dtype=np.int64))
print("токенів", len(a), "→", len(res), f"(×{len(a) / len(res):.2f})", f"{time.time() - t0:.0f}с", flush=True)
