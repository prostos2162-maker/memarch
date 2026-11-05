"""M11-дані: корпус для тренування (порт data/loader.py, та сама логіка → ті самі токени).

Шматки `<джерело>/<ім'я>.u32.zst` (uint32, документи до EOS) + маски поруч: `.drop.npy` (дедуп), `.cdrop.npy`
(бенчмарки) — номери документів для пропуску; `.score.npy` (код) — якість 0–5 → повтори ≥3.0 ×3, 2.5–3.0 ×1,
<2.5 — кожен 4-й. Джерела змішуються в заданих частках за токенами; вікна по seq+1. Корінь — gs://... (через
gcloud storage, кеш на диску) або локальна тека. Кілька хостів: шматок i береться хостом i % world == rank.

Відмінності від torch-лоадера: EOS — поле екземпляра (не глобальна змінна), вихід int32 (V=131328 < 2^31),
кеш за замовчуванням — tempdir. HostLoader додає префетч у треді + хостову підготовку induct_raw (CPU_PREP) і
state_dict, що вказує на ПЕРШИЙ НЕВИДАНИЙ батч (префетчені, але не видані батчі після resume повторяться).

    c = Corpus({"fineweb_edu": 0.3, "stack_v3": 0.7}, root="gs://your-bucket/tokens", seq=8192)
    xy = c.batch(16)            # (16, seq+1) int32 numpy
    hl = HostLoader(c, B=4, prep_fn=lambda x: induct.induct_raw(x, V, lock_bucket))
    xy, prep = hl.next()
"""
import json
import os
import queue
import subprocess
import tempfile
import threading

import numpy as np

DEFAULT_ROOT = os.environ.get("MEMARCH_DATA_ROOT", "tokens")
_EXTS = (".u32.zst", ".drop.npy", ".cdrop.npy", ".score.npy")


def _gcloud(*args):
    return subprocess.run(["gcloud", "storage", *args], capture_output=True, text=True, check=True).stdout


def _ls(root, src):
    """Імена шматків джерела (без розширення), відсортовані."""
    if root.startswith("gs://"):
        names = [os.path.basename(p)[:-len(".u32.zst")] for p in _gcloud("ls", f"{root}/{src}/*.u32.zst").split()]
    else:
        names = [f[:-len(".u32.zst")] for f in os.listdir(os.path.join(root, src)) if f.endswith(".u32.zst")]
    return sorted(names)


def _fetch(root, src, name, cache):
    """Шматок і маски — локальні шляхи (з бакета качаються в cache; відсутні — None)."""
    if not root.startswith("gs://"):
        paths = [os.path.join(root, src, name + e) for e in _EXTS]
        return [p if os.path.exists(p) else None for p in paths]
    d = os.path.join(cache, src)
    os.makedirs(d, exist_ok=True)
    have = [os.path.join(d, name + e) for e in _EXTS]
    if not os.path.exists(have[0]):
        subprocess.run(["gcloud", "storage", "cp", f"{root}/{src}/{name}.*", d + "/"], capture_output=True)
    return [p if os.path.exists(p) else None for p in have]


def _docs(paths, rng, eos):
    """Документи шматка після масок і повторів за якістю коду (список uint32-масивів, кожен з EOS у кінці)."""
    import zstandard as zstd
    tok, drop, cdrop, score = paths
    with open(tok, "rb") as f:
        arr = np.frombuffer(zstd.ZstdDecompressor().decompress(f.read()), dtype=np.uint32)
    ends = np.flatnonzero(arr == eos) + 1
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
    def __init__(self, root, src, cache, seed, rank, world, eos):
        self.root, self.src, self.cache, self.eos = root, src, cache, eos
        names = _ls(root, src)
        self.names = [n for i, n in enumerate(names) if i % world == rank]
        assert self.names, f"{src}: немає шматків для rank {rank}/{world}"
        self.seed = seed
        self.epoch, self.si, self.di, self.off = 0, 0, 0, 0
        self._docs, self._next = None, None
        self._order()

    def _order(self):
        self.order = np.random.default_rng((self.seed, self.epoch)).permutation(len(self.names))

    def _load(self, si):
        name = self.names[self.order[si % len(self.order)]]
        rng = np.random.default_rng((self.seed, self.epoch, si))
        return _docs(_fetch(self.root, self.src, name, self.cache), rng, self.eos)

    def _prefetch(self):
        nxt = self.si + 1
        box = {}
        t = threading.Thread(target=lambda: box.setdefault("d", self._load(nxt)) if nxt < len(self.order) else None,
                             daemon=True)
        t.start()
        self._next = (nxt, t, box)

    def next_piece(self, n):
        """Наступні ≤ n токенів джерела (великий документ — частинами)."""
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
            if self.si >= len(self.order):              # нова епоха — новий порядок шматків
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


def _read_eos(root, src0):
    n0 = _ls(root, src0)[0]
    if root.startswith("gs://"):
        man = json.loads(_gcloud("cat", f"{root}/{src0}/{n0}.json"))
    else:
        with open(os.path.join(root, src0, n0 + ".json")) as f:
            man = json.load(f)
    return int(man["eos"])


class Corpus:
    def __init__(self, weights, root=DEFAULT_ROOT, seq=8192, seed=0, rank=0, world=1, cache=None, eos=None):
        cache = cache or os.path.join(tempfile.gettempdir(), "corpus_cache")
        self.eos = _read_eos(root, next(iter(weights))) if eos is None else int(eos)
        tot = sum(weights.values())
        self.srcs = list(weights)
        self.p = np.array([weights[s] / tot for s in self.srcs])
        self.sources = [_Source(root, s, cache, seed + 1000 * k, rank, world, self.eos)
                        for k, s in enumerate(self.srcs)]
        self.seq, self.seed, self.rank = seq, seed, rank
        self.n_docs = 0
        self.buf = np.zeros(0, dtype=np.uint32)
        self.taken = np.zeros(len(self.srcs))

    def _pick(self):
        want = self.p * (self.taken.sum() + 1) - self.taken     # найбільший недобір від цільової частки
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
        return flat[:need].astype(np.int32).reshape(B, self.seq + 1)

    def state_dict(self):
        return {"sources": [s.state() for s in self.sources], "taken": self.taken.tolist(), "buf": self.buf.tolist(),
                "n_docs": self.n_docs}

    def load_state_dict(self, st):
        for s, ss in zip(self.sources, st["sources"]):
            s.restore(ss)
        self.taken = np.array(st["taken"], dtype=np.float64)
        self.buf = np.array(st["buf"], dtype=np.uint32)
        self.n_docs = st["n_docs"]

    def mix(self):
        t = self.taken.sum() or 1
        return {s: round(float(x / t), 4) for s, x in zip(self.srcs, self.taken)}


class HostLoader:
    """Префетч батчів хоста в треді: next() → (xy (B,T+1) int32, prep), prep = prep_fn(xy[:, :-1]) (напр. induct_raw).
    Кожен елемент черги несе стан корпусу ПІСЛЯ свого батчу, тож state_dict() — точний стан «перед першим
    НЕвиданим батчем» (префетчені, але не видані батчі після resume повторяться — нічого не губиться)."""

    def __init__(self, corpus, B, prep_fn=None, depth=2):
        self.c, self.B, self.prep_fn, self.depth = corpus, B, prep_fn, depth
        self._state = corpus.state_dict()
        self._start()

    def _start(self):
        self.q = queue.Queue(self.depth)
        self._stop = threading.Event()
        self.t = threading.Thread(target=self._run, daemon=True)
        self.t.start()

    def _run(self):
        try:
            while not self._stop.is_set():
                xy = self.c.batch(self.B)
                st = self.c.state_dict()
                prep = self.prep_fn(xy[:, :-1]) if self.prep_fn else None
                while not self._stop.is_set():
                    try:
                        self.q.put((xy, prep, st), timeout=0.1)
                        break
                    except queue.Full:
                        pass
        except BaseException as e:                          # помилку треду віддаємо споживачу
            self.q.put((e, None, None))

    def next(self):
        xy, prep, st = self.q.get()
        if isinstance(xy, BaseException):
            raise xy
        self._state = st
        return xy, prep

    def state_dict(self):
        return self._state

    def load_state_dict(self, st):
        self.close()
        self.c.load_state_dict(st)
        self._state = self.c.state_dict()
        self._start()

    def close(self):
        self._stop.set()
        self.t.join()


def induct_prep(V, lock_bucket):
    """prep_fn для HostLoader: induct_raw на хості (numpy) → dict як model.make_prep (до jnp — у step.make_batch)."""
    from .induct import induct_raw

    def fn(x):
        cand, on, bk, src = induct_raw(np.asarray(x), V, lock_bucket)
        return dict(cand=cand.astype(np.int32), on=on, bk=bk.astype(np.int32), src=src.astype(np.int32))
    return fn
