"""MinHash-підписи для дедупу по всіх джерелах (крок 1 з 2; крок 2 — dedup_cluster.py: зіткнення смуг -> кластери).
One-permutation hashing: 5-токенні шингли, 112 кошиків = 14 смуг × 8 рядків (поріг ~0.8 за Жаккаром).
Для кожного шматка — `<шматок>.mh.npy`: uint64 [документів, 14], хеш кожної смуги; 0 = смуга з порожнім кошиком
(документ без жодного шингла), її пропускаємо при зіставленні. Тексти не розкодовуються.
  python data/minhash.py --remote remote:corpus/tokens/<джерело> [--worker i --workers n]
  python data/minhash.py --local <шматок.u32.zst>     (перевірка на одному файлі)"""
import argparse, os, subprocess, time
import numpy as np, zstandard as zstd

EOS, S, BINS, BANDS = 131072, 5, 112, 14
ROWS = BINS // BANDS
CHUNK = 32_000_000
M1, M2 = np.uint64(0x9E3779B97F4A7C15), np.uint64(0xBF58476D1CE4E5B9)


def shingles(a):
    a = a.astype(np.uint64)
    h = np.zeros(len(a) - S + 1, dtype=np.uint64)
    with np.errstate(over="ignore"):
        for k in range(S):
            h = h * M1 + a[k:len(a) - S + 1 + k]
        h ^= h >> np.uint64(31)
        h *= M2
        h ^= h >> np.uint64(29)
    return h


def densify(sig):
    """Порожній кошик бере значення найближчого непорожнього праворуч (по колу) + зсув за відстань —
    однаково для всіх документів, тож короткі документи теж порівнювані (Shrivastava 2017)."""
    ok = sig != 0xFFFFFFFF
    i = np.arange(2 * BINS)
    nxt = np.where(np.concatenate([ok, ok], 1), i, 4 * BINS)
    nxt = np.minimum.accumulate(nxt[:, ::-1], axis=1)[:, ::-1][:, :BINS]
    has = nxt < 2 * BINS
    src = np.where(has, nxt % BINS, 0)
    out = np.take_along_axis(sig, src, 1) + (nxt - i[:BINS]).astype(np.uint64) * np.uint64(0x100000000)
    return np.where(has, np.where(ok, sig, out), np.uint64(0xFFFFFFFF))


def signatures(arr):
    ends = np.flatnonzero(arr == EOS)
    out = np.zeros((len(ends), BANDS), dtype=np.uint64)
    d0 = 0
    while d0 < len(ends):
        s = 0 if d0 == 0 else ends[d0 - 1] + 1
        d1 = int(np.searchsorted(ends, s + CHUNK)) + 1          # цілі документи, ~CHUNK токенів
        d1 = min(max(d1, d0 + 1), len(ends))
        e = ends[d1 - 1] + 1
        loc_ends = ends[d0:d1] - s
        if e - s >= S:
            h = shingles(arr[s:e])
            p = np.arange(len(h))
            d = np.searchsorted(loc_ends, p)                    # документ, у якому починається шингл
            ok = p + S - 1 < loc_ends[d]                        # шингл не зачіпає <|endoftext|>
            h, d = h[ok], d[ok].astype(np.uint64)
            key = (d * np.uint64(BINS) + h % np.uint64(BINS)) << np.uint64(32) | (h >> np.uint64(32))
            key.sort()
            k = key >> np.uint64(32)
            first = np.concatenate([[True], k[1:] != k[:-1]])
            sig = np.full((d1 - d0) * BINS, np.uint64(0xFFFFFFFF), dtype=np.uint64)
            sig[k[first].astype(np.int64)] = key[first] & np.uint64(0xFFFFFFFF)
            sig = densify(sig.reshape(d1 - d0, BINS)).reshape(d1 - d0, BANDS, ROWS)
            empty = (sig == 0xFFFFFFFF).any(2)                  # лише документи без жодного шингла
            with np.errstate(over="ignore"):
                bh = np.full((d1 - d0, BANDS), np.uint64(1469598103934665603), dtype=np.uint64)
                for r in range(ROWS):
                    bh = (bh ^ sig[:, :, r]) * M2
                bh += np.arange(BANDS, dtype=np.uint64) * M1     # різні смуги не зіставляються між собою
            bh[empty | (bh == 0)] = 0
            out[d0:d1] = bh
        d0 = d1
    return out


def load(path):
    return np.frombuffer(zstd.ZstdDecompressor().decompress(open(path, "rb").read(), max_output_size=8_000_000_000),
                         dtype=np.uint32)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--remote")
    p.add_argument("--local")
    p.add_argument("--worker", type=int, default=0)
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--workdir", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "_tmp", "minhash"))
    a = p.parse_args()
    if a.local:
        t = time.time(); arr = load(a.local); sg = signatures(arr)
        np.save(a.local.replace(".u32.zst", ".mh.npy"), sg)
        print(f"{len(arr) / 1e6:.0f}M токенів, {len(sg)} док. за {time.time() - t:.0f} с; порожніх смуг {np.mean(sg == 0):.1%}")
    else:
        os.makedirs(a.workdir, exist_ok=True)
        ls = lambda inc: subprocess.run(["rclone", "lsf", a.remote, "--include", inc], capture_output=True,
                                        text=True).stdout.split()
        names, have = sorted(ls("*.u32.zst")), set(ls("*.mh.npy"))
        for nm in names[a.worker::a.workers]:
            out = nm.replace(".u32.zst", ".mh.npy")
            if out in have:
                continue
            loc = os.path.join(a.workdir, nm)
            subprocess.run(["rclone", "copy", f"{a.remote}/{nm}", a.workdir, "-q"], check=True)
            t = time.time(); sg = signatures(load(loc))
            np.save(os.path.join(a.workdir, out), sg)
            subprocess.run(["rclone", "move", os.path.join(a.workdir, out), a.remote, "-q"], check=True)
            os.remove(loc)
            print(f"{nm}: {len(sg)} док. за {time.time() - t:.0f} с, порожніх смуг {np.mean(sg == 0):.1%}", flush=True)
