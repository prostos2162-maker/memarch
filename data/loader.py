"""Завантажувач корпусу для тренування: шматки `<джерело>/<ім'я>.u32.zst` (uint32, документи до <|endoftext|>)
+ маски поруч: `.drop.npy` (дедуп), `.cdrop.npy` (бенчмарки) — номери документів для пропуску;
`.score.npy` (лише код) — якість репозиторію 0–5 → повтори: ≥3.0 ×3, 2.5–3.0 ×1, <2.5 — кожен 4-й (0.25).

Джерела змішуються в заданих частках (за токенами); документи склеюються у вікна по seq+1 токенів.
Позицію (які шматки й документи вже видано, стан випадковості) можна зберегти в чекпойнт і продовжити
з того ж місця — spot-машину можуть забрати будь-коли. На кількох хостах кожен бере свої шматки
(номер шматка % world == rank).

    c = Corpus({"fineweb_edu": 0.3, "stack_v3": 0.5, ...}, root="gs://bucket/tokens", seq=8192, rank=0, world=1)
    xy = c.batch(16)          # (16, seq+1) int64, numpy
    st = c.state_dict(); c.load_state_dict(st)
"""
import json
import os
import subprocess
import threading

import numpy as np
import zstandard as zstd

EOS = None


def _ls(root, src):
    """Імена шматків джерела (без розширення), відсортовані."""
    if root.startswith("gs://"):
        out = subprocess.run(["gcloud", "storage", "ls", f"{root}/{src}/*.u32.zst"], capture_output=True, text=True,
                             check=True).stdout.split()
        names = [os.path.basename(p)[:-len(".u32.zst")] for p in out]
    else:
        names = [f[:-len(".u32.zst")] for f in os.listdir(os.path.join(root, src)) if f.endswith(".u32.zst")]
    return sorted(names)


def _fetch(root, src, name, cache):
    """Шматок і його маски — локальні шляхи (з бакета качаються в cache; чого нема — None)."""
    exts = (".u32.zst", ".drop.npy", ".cdrop.npy", ".score.npy")
    if not root.startswith("gs://"):
        paths = [os.path.join(root, src, name + e) for e in exts]
        return [p if os.path.exists(p) else None for p in paths]
    d = os.path.join(cache, src)
    os.makedirs(d, exist_ok=True)
    have = [os.path.join(d, name + e) for e in exts]
    if not os.path.exists(have[0]):
        subprocess.run(["gcloud", "storage", "cp", f"{root}/{src}/{name}.*", d + "/"], capture_output=True)
    return [p if os.path.exists(p) else None for p in have]


def _docs(paths, rng):
    """Документи шматка після масок і повторів за якістю коду (список масивів uint32, кожен — з EOS у кінці)."""
    tok, drop, cdrop, score = paths
    arr = np.frombuffer(zstd.ZstdDecompressor().decompress(open(tok, "rb").read()), dtype=np.uint32)
    ends = np.flatnonzero(arr == EOS) + 1
    starts = np.concatenate([[0], ends[:-1]])
    keep = np.ones(len(ends), dtype=bool)
    for m in (drop, cdrop):
        if m:
            idx = np.load(m).astype(np.int64)
            keep[idx[idx < len(keep)]] = False
    reps = np.ones(len(ends), dtype=np.int64)
    if score:
        s = np.load(score).astype(np.float32)[:len(ends)]
        reps = np.where(s >= 3.0, 3, np.where(s >= 2.5, 1, (rng.random(len(s)) < 0.25).astype(np.int64)))
    out = []
    for i in np.flatnonzero(keep & (reps > 0)):
        out.extend([arr[starts[i]:ends[i]]] * int(reps[i]))
    order = rng.permutation(len(out))                  # повтори одного репозиторію не стоять поруч
    return [out[j] for j in order]


class _Source:
    def __init__(self, root, src, cache, seed, rank, world):
        self.root, self.src, self.cache = root, src, cache
        names = _ls(root, src)
        self.names = [n for i, n in enumerate(names) if i % world == rank]
        assert self.names, f"{src}: немає шматків для rank {rank}/{world}"
        self.seed = seed
        self.epoch, self.si, self.di, self.off = 0, 0, 0, 0   # епоха, шматок у порядку епохи, документ, зсув у ньому
        self._docs, self._next = None, None
        self._order()

    def _order(self):
        self.order = np.random.default_rng((self.seed, self.epoch)).permutation(len(self.names))

    def _load(self, si):
        name = self.names[self.order[si % len(self.order)]]
        rng = np.random.default_rng((self.seed, self.epoch, si))
        return _docs(_fetch(self.root, self.src, name, self.cache), rng)

    def _prefetch(self):
        nxt = self.si + 1
        box = {}
        t = threading.Thread(target=lambda: box.setdefault("d", self._load(nxt)) if nxt < len(self.order) else None,
                             daemon=True)
        t.start()
        self._next = (nxt, t, box)

    def next_piece(self, n):
        """Наступні ≤ n токенів джерела: великий документ (цілий репозиторій) видається частинами,
        інакше один документ на сотні тисяч токенів збив би частки джерел."""
        if self._docs is None or self.di >= len(self._docs):
            self._advance()
        d = self._docs[self.di]
        piece = d[self.off:self.off + n]
        self.off += len(piece)
        if self.off >= len(d):
            self.di, self.off = self.di + 1, 0
        return piece

    def _advance(self):
        if self._docs is None:
            self._docs = self._load(self.si)
            self._prefetch()
        while self.di >= len(self._docs):
            self.si, self.di = self.si + 1, 0
            if self.si >= len(self.order):              # джерело вичерпано — нова епоха, новий порядок шматків
                self.epoch, self.si = self.epoch + 1, 0
                self._order()
                self._docs, self._next = self._load(0), None
            else:
                nxt, t, box = self._next
                t.join()
                self._docs = box.get("d") if nxt == self.si else self._load(self.si)
            self._prefetch()

    def state(self):
        return {"epoch": self.epoch, "si": self.si, "di": self.di, "off": self.off}

    def restore(self, st):
        self.epoch, self.si, self.di, self.off = st["epoch"], st["si"], st["di"], st.get("off", 0)
        self._order()
        self._docs, self._next = None, None


class Corpus:
    def __init__(self, weights, root, seq=8192, seed=0, rank=0, world=1, cache="/tmp/corpus_cache", eos=None):
        global EOS
        if eos is None:                                 # EOS — з маніфесту першого шматка
            src0 = next(iter(weights))
            n0 = _ls(root, src0)[0]
            if root.startswith("gs://"):
                man = json.loads(subprocess.run(["gcloud", "storage", "cat", f"{root}/{src0}/{n0}.json"],
                                                capture_output=True, text=True, check=True).stdout)
            else:
                man = json.load(open(os.path.join(root, src0, n0 + ".json")))
            eos = man["eos"]
        EOS = eos
        tot = sum(weights.values())
        self.srcs = list(weights)
        self.p = np.array([weights[s] / tot for s in self.srcs])
        self.sources = [_Source(root, s, cache, seed + 1000 * k, rank, world) for k, s in enumerate(self.srcs)]
        self.seq, self.seed, self.rank = seq, seed, rank
        self.n_docs = 0
        self.rng = np.random.default_rng((seed, rank))
        self.buf = np.zeros(0, dtype=np.uint32)
        self.taken = np.zeros(len(self.srcs))           # скільки токенів видано з кожного джерела

    def _pick(self):
        # детерміновано: джерело з найбільшим недобором від цільової частки (точні частки, без шуму)
        want = self.p * (self.taken.sum() + 1) - self.taken
        return int(np.argmax(want))

    def batch(self, B):
        need = B * (self.seq + 1)
        parts = [self.buf]
        have = len(self.buf)
        while have < need:
            k = self._pick()
            d = self.sources[k].next_piece(self.seq + 1)
            self.taken[k] += len(d)
            self.n_docs += 1
            parts.append(d)
            have += len(d)
        flat = np.concatenate(parts)
        self.buf = flat[need:]
        return flat[:need].astype(np.int64).reshape(B, self.seq + 1)

    def state_dict(self):
        return {"sources": [s.state() for s in self.sources], "taken": self.taken.tolist(), "buf": self.buf.tolist(),
                "n_docs": self.n_docs}

    def load_state_dict(self, st):
        for s, ss in zip(self.sources, st["sources"]):
            s.restore(ss)
        self.taken = np.array(st["taken"])
        self.buf = np.array(st["buf"], dtype=np.uint32)
        self.n_docs = st["n_docs"]

    def mix(self):
        """Фактичні частки джерел (за токенами) — для журналу."""
        t = self.taken.sum() or 1
        return {s: round(float(x / t), 4) for s, x in zip(self.srcs, self.taken)}
