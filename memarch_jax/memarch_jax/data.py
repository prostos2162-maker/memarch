"""M11-дані: корпус для тренування (порт D:\\ai\\memarch\\data\\loader.py, та сама логіка → ті самі токени).

Шматки `<джерело>/<ім'я>.u32.zst` (uint32, документи до EOS) + маски поруч: `.drop.npy` (дедуп), `.cdrop.npy`
(бенчмарки) — номери документів для пропуску; `.score.npy` (код) — якість 0–5 → повтори ≥3.0 ×3, 2.5–3.0 ×1,
<2.5 — кожен 4-й. Джерела змішуються в заданих частках за токенами; вікна по seq+1. Корінь — gs://... (через
gcloud storage, кеш на диску) або локальна тека. Кілька хостів: шматок i береться хостом i % world == rank.

Відмінності від torch-лоадера: EOS — поле екземпляра (не глобальна змінна), вихід int32 (V=131328 < 2^31),
кеш за замовчуванням — tempdir. HostLoader додає префетч у треді + хостову підготовку induct_raw (CPU_PREP) і
state_dict, що вказує на ПЕРШИЙ НЕВИДАНИЙ батч (префетчені, але не видані батчі після resume повторяться).

    c = Corpus({"fineweb_edu": 0.3, "stack_v3": 0.7}, root="gs://YOUR_BUCKET/tokens", seq=8192)
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


class KindCorpus:
    """Фішки (Стефан 11.11): УВЕСЬ батч кроку — один вид тексту. kinds: {вид: {джерело: вага}}; pools: набори фішок
    за видом (fishki.pools). Вид кроку — детермінований недобір від частки виду (однаковий на всіх процесах, бо
    залежить лише від номера кроку), комбінація фішок — випадкова з набору виду (seed, крок) — теж однакова скрізь.
    full=True (фінальна фаза) — весь набір виду в його порядку. extra() → {'kind': (B,1), 'fish': (B,C)} для prep."""

    def __init__(self, kinds, pools, c, root=DEFAULT_ROOT, seq=8192, seed=0, rank=0, world=1, cache=None, eos=None):
        self.names = list(kinds)
        w = np.array([sum(kinds[k].values()) for k in self.names], np.float64)
        self.p = w / w.sum()
        self.subs = [Corpus(kinds[k], root=root, seq=seq, seed=seed + 7919 * i, rank=rank, world=world, cache=cache,
                            eos=eos) for i, k in enumerate(self.names)]
        self.eos = self.subs[0].eos
        self.pools, self.c, self.seed = pools, int(c), seed
        self.full = False
        self.n = 0
        self.count = np.zeros(len(self.names))
        self._extra = None

    def _kind(self):
        return int(np.argmax(self.p * (self.n + 1) - self.count))

    def batch(self, B):
        k = self._kind()
        xy = self.subs[k].batch(B)
        if self.full:
            order = list(self.pools[k])
        else:
            rng = np.random.default_rng([self.seed, 31337, self.n])
            order = [int(i) for i in rng.choice(self.pools[k], self.c, replace=False)]
        self._extra = {"kind": np.full((B, 1), k, np.int32), "fish": np.tile(np.asarray(order, np.int32), (B, 1))}
        self.count[k] += 1
        self.n += 1
        return xy

    def extra(self):
        return self._extra

    def state_dict(self):
        return {"subs": [s.state_dict() for s in self.subs], "count": self.count.tolist(), "n": self.n}

    def load_state_dict(self, st):
        for s, ss in zip(self.subs, st["subs"]):
            s.load_state_dict(ss)
        self.count = np.array(st["count"], np.float64)
        self.n = int(st["n"])

    def mix(self):
        t = self.count.sum() or 1
        return {k: round(float(x / t), 4) for k, x in zip(self.names, self.count)}


class HostLoader:
    """Префетч батчів хоста в треді: next() → (xy (B,T+1) int32, prep), prep = prep_fn(xy[:, :-1]) (напр. induct_raw).
    Кожен елемент черги несе стан корпусу ПІСЛЯ свого батчу, тож state_dict() — точний стан «перед першим
    НЕвиданим батчем» (префетчені, але не видані батчі після resume повторяться — нічого не губиться)."""

    def __init__(self, corpus, B, prep_fn=None, depth=2, xform=None):
        self.c, self.B, self.prep_fn, self.depth, self.xform = corpus, B, prep_fn, depth, xform
        self._state = corpus.state_dict()
        self.n_prod = 0                                     # індекс наступного батчу (лише з xform: розклад нагадувань)
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
                extra = None
                if self.xform is not None:                  # напр. Reminders: голки в текст + маска відповідей
                    xy, extra = self.xform(xy, n=self.n_prod)
                    self.n_prod += 1
                    st = dict(st, _n=self.n_prod)
                prep = self.prep_fn(xy[:, :-1]) if self.prep_fn else None
                kx = self.c.extra() if hasattr(self.c, "extra") else None    # фішки: вид і комбінація кроку
                if kx:                                      # синтетика (--remind_pack) має інше число рядків
                    prep = dict(prep or {}, **{k: np.repeat(v[:1], xy.shape[0], 0) for k, v in kx.items()})
                if extra:
                    prep = mask_induct(dict(prep or {}, **extra))
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
        if self.xform is not None:
            self.n_prod = int(st.get("_n", 0))
            self._state = dict(self._state, _n=self.n_prod)
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


def mask_induct(prep):
    """Нагадування: на позиціях питань вимикаємо замки індукції (on=False). Інакше induct_raw (остання попередня поява
    слова на БУДЬ-якій відстані → наступне слово) дає відповідь голки без пам'яті, і модель кидає трос."""
    rem = prep.get("rem")
    if rem is not None and "on" in prep and rem.any():
        on = np.array(prep["on"], copy=True)
        on[rem.astype(bool)] = False
        prep["on"] = on
    return prep


class Reminders:
    """exact (лише eval): відстань факти→питання рівно exact слів (заміри пригадування на кількох довжинах).
    Нагадування (порт RECALL_REMIND з memarch/rope_recall.py на справжній текст).
    З імовірністю p рядок отримує голку: блок фактів [M k1 v1 … kn vn M] і далі, не ближче ніж dist слів після
    нього, блок питань [M kq1 vq1 … M] (кожен ключ рівно раз, у випадковому порядку) — як recall_long.batch:
    ключі різні з 512, значення з 64. Решта рядка — звичайний текст. Маска rem (B,T) = 1 на позиціях x, де стоїть
    ключ-питання (ціль y там — значення). Форми не змінюються (перезапис на місці).
    Випадковість — від (seed, вміст рядка): той самий батч після resume дає ті самі голки (стану немає)."""

    N_KEYS, N_VALS = 512, 64

    def __init__(self, p, T, n=16, dist=2048, tok0=1000, seed=0, V=None, p0=None, warm=0, decay=0,
                 ladder=None, thr=0.9, win=800, lag=5, dmin=0, synth=0, pack=False,
                 pack_gap=0, pack_round=1, pack_mult=1, doc_eos=None,
                 near=0.0, near_max=0):
        """Розклад (необов'язковий): p0 перші warm батчів, далі лінійно за decay батчів до p (кінцевого).
        ladder (необов'язковий): відстані рівнів (зростають). Рівень k: відстань рівно з [ladder[k-1], ladder[k]]
        (k=0: [dmin, ladder[0]]); останній рівень — уся [dmin, ladder[-1]] (як RECALL_MIX у torch: не забувати короткі).
        Рівень росте, коли точність на останніх ≥win відповідях нагадувань ≥thr (torch: 50 кроків > 0.9).
        Новий рівень діє з батчу (поточний + lag) — lag > глибини префетчу, тож вибір детермінований; стан — state()."""
        self.p, self.T, self.n, self.dist, self.seed = float(p), int(T), int(n), int(dist), int(seed)
        self.p0, self.warm, self.decay = (None if p0 is None else float(p0)), int(warm), int(decay)
        self.ladder = [int(v) for v in ladder] if ladder else None
        self.thr, self.win, self.lag, self.dmin = float(thr), int(win), int(lag), int(dmin)
        self.synth, self.V = int(synth), V                  # етап 1: перші synth батчів — чиста синтетика (як recall_long)
        # --rope_doc_mask: трос не бачить через EOS → у сміття EOS не потрапляє, а EOS між фактами й питанням
        # у рядку справжнього тексту прибирається (голка = один «документ»), інакше вона нерозв'язна за вікном
        self.doc_eos = None if doc_eos is None or int(doc_eos) < 0 else int(doc_eos)
        # near: частка голок справжнього тексту на відстані 0..near_max (у вікні, трос відкритий) — щоб трос
        # не заважав близьким фактам (рішення Стефана 11.11: лікувати даними, не архітектурою)
        self.near, self.near_max = float(near), int(near_max)
        self.changes = [(0, 0)]                              # (з якого батчу, рівень)
        self.acc_h = self.acc_n = 0.0
        self.k0, self.v0, self.mark = tok0, tok0 + self.N_KEYS, tok0 + self.N_KEYS + self.N_VALS
        self.L = 2 * self.n + 2
        # pack (етап 1, --remind_pack): синтетичний рядок щільно викладено епізодами torch recall_long.batch
        # ([k v]·пар, сміття, [k v]·пар — без міток), рівні (пар, сміття) «за успіхом», як у torch: сміття 16,32,… × пари 4,8,n
        self.pack, self.pack_gap = bool(pack), int(pack_gap)
        self.pack_round, self.pack_mult = max(1, int(pack_round)), max(1, int(pack_mult))
        if self.pack:
            fl = [f for f in (16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536)
                  if 4 * self.n + f <= self.T + 1]
            assert fl, f"T={T} замале для епізоду з {self.n} пар і сміття 16"
            self.pfill = fl
            self.plevels = [(pp, f) for f in fl for pp in sorted({min(4, self.n), min(8, self.n), self.n})]
            self.pchanges, self.pacc_h, self.pacc_n = [(0, 0)], 0.0, 0.0
        assert V is None or self.mark < V, f"токени нагадувань {tok0}..{self.mark} ≥ V {V}"
        assert self.n <= self.N_KEYS
        assert 2 * self.L + self.dist <= self.T + 1, f"T={T} замале для 2 блоків по {self.L} і відстані {dist}"
        if self.ladder:
            assert self.ladder == sorted(self.ladder) and self.ladder[-1] <= self.T + 1 - 2 * self.L, self.ladder

    def _filler(self, rng, n):
        """Випадкові слова поза токенами голок (torch: FILL0..FILL0+20000)."""
        V = self.V or (self.mark + 20001)
        f = rng.integers(0, V - (self.mark + 1 - self.k0), n)
        f = np.where(f >= self.k0, f + (self.mark + 1 - self.k0), f).astype(np.int32)
        if self.doc_eos is not None:
            f[f == self.doc_eos] = 0
        return f

    def p_at(self, i):
        if self.p0 is None:
            return self.p
        if i < self.warm:
            return self.p0
        f = min(1.0, (i - self.warm) / max(self.decay, 1))
        return self.p0 + (self.p - self.p0) * f

    def level_at(self, i):
        k = 0
        for n0, kk in self.changes:
            if n0 <= i:
                k = kk
        return k

    def range_at(self, i):
        k = self.level_at(i)
        if k == len(self.ladder) - 1:
            return self.dmin, self.ladder[-1]
        return (self.ladder[k - 1] if k else self.dmin), self.ladder[k]

    def plevel_at(self, i):
        k = 0
        for n0, kk in self.pchanges:
            if n0 <= i:
                k = kk
        return k

    def _observe_pack(self, i, hits, n_ans):
        self.pacc_h += hits
        self.pacc_n += n_ans
        k = self.pchanges[-1][1]
        if self.pacc_n < self.win or k >= len(self.plevels) - 1:
            return None
        acc = self.pacc_h / self.pacc_n
        self.pacc_h = self.pacc_n = 0.0
        if acc < self.thr:
            return None
        self.pchanges.append((i + self.lag, k + 1))
        return (f"синтетика: рівень {k} → {k + 1} (точність {acc:.3f}), з батчу {i + self.lag} "
                f"пар {self.plevels[k + 1][0]} сміття {self.plevels[k + 1][1]}"
                + (" (останній: усі довжини впереміш)" if k + 1 == len(self.plevels) - 1 else ""))

    def _pack_row(self, row, rng, pp, f):
        """Рядок (уже сміття) → епізоди [k v]·pp, сміття f, [k v]·pp (питання в іншому порядку) до кінця рядка.
        pack_gap = 0: епізоди впритул. pack_gap = g > 0: між епізодами g однакових слів-міток M і мітки ж на початку
        рядка — у вікні питання лише свій епізод (короткий контекст, як рядок torch довжиною 4·pp+f), а не чужі факти.
        Ключі — з перестановок 512 поспіль (той самий ключ повториться не ближче ніж через ~512/pp епізодів)."""
        T1, Le, g = len(row), 4 * pp + f, self.pack_gap
        ne = (T1 + g) // (Le + g)
        off = T1 - (ne * Le + (ne - 1) * g)
        if g:
            row[:off] = self.mark
        keys = self.k0 + np.concatenate([rng.permutation(self.N_KEYS) for _ in range(-(-ne * pp // self.N_KEYS))])
        keys = keys[:ne * pp].reshape(ne, pp)
        vals = self.v0 + rng.integers(0, self.N_VALS, (ne, pp))
        pos = []
        for e in range(ne):
            b0 = off + e * (Le + g)
            if g and e:
                row[b0 - g:b0] = self.mark
            qi = rng.permutation(pp)
            row[b0:b0 + 2 * pp] = np.stack([keys[e], vals[e]], 1).reshape(-1)
            q0 = b0 + 2 * pp + f
            row[q0:q0 + 2 * pp] = np.stack([keys[e][qi], vals[e][qi]], 1).reshape(-1)
            pos.append(q0 + 2 * np.arange(pp))
        return np.concatenate(pos)

    def _short_batch(self, shape, n):
        """Етап 1 (--remind_pack), батч n: КОРОТКІ рядки точно як torch recall_long.batch — [k v]·пар на ПОЧАТКУ рядка,
        сміття, [k v]·пар (питання в іншому порядку). Довжина рядка = 4·пар + сміття (округлена вгору до pack_round),
        рядків стільки, щоб слів у батчі було як у звичайному (B·T), кратно pack_mult (чипам). Форма батчу залежить
        від рівня → крок компілюється раз на рівень. Факти на початку рядка — суть: там увага бачить лише кілька слів,
        і зв'язка «значення ← свій ключ» вчиться швидко; посеред довгого рядка та сама задача з нуля вчиться в рази довше."""
        B, T1 = shape
        k = self.plevel_at(n)
        pp, f = self.plevels[k]
        rng = np.random.default_rng([self.seed, 7919, n])
        if k == len(self.plevels) - 1:                      # останній рівень: усі довжини сміття впереміш (torch RECALL_MIX)
            f = self.pfill[int(rng.integers(0, len(self.pfill)))]
        Ts = -(-(4 * pp + f - 1) // self.pack_round) * self.pack_round          # довжина x
        f = Ts + 1 - 4 * pp
        m = self.pack_mult
        R = max(m, (B * (T1 - 1) // Ts) // m * m)
        xy = np.empty((R, Ts + 1), np.int32)
        rem = np.zeros((R, Ts), np.int8)
        ans = 2 * pp + f + 2 * np.arange(pp)
        for r in range(R):
            keys = self.k0 + rng.permutation(self.N_KEYS)[:pp]
            vals = self.v0 + rng.integers(0, self.N_VALS, pp)
            qi = rng.permutation(pp)
            xy[r, :2 * pp] = np.stack([keys, vals], 1).reshape(-1)
            xy[r, 2 * pp:2 * pp + f] = self._filler(rng, f)
            xy[r, 2 * pp + f:] = np.stack([keys[qi], vals[qi]], 1).reshape(-1)
            rem[r, ans] = 1
        return xy, {"rem": rem}

    def observe(self, i, hits, n_ans):
        """Головний цикл після кроку i (батч i): точність голок у train → можливий перехід. → рядок або None."""
        if self.pack and i < self.synth:
            return self._observe_pack(i, hits, n_ans)
        if not self.ladder:
            return None
        self.acc_h += hits
        self.acc_n += n_ans
        k = self.changes[-1][1]
        if self.acc_n < self.win or k >= len(self.ladder) - 1:
            return None
        acc = self.acc_h / self.acc_n
        self.acc_h = self.acc_n = 0.0
        if acc < self.thr:
            return None
        self.changes.append((i + self.lag, k + 1))
        lo, hi = self.range_at(i + self.lag)
        return f"нагадування: рівень {k} → {k + 1} (точність {acc:.3f}), з батчу {i + self.lag} відстань {lo}..{hi}"

    def state(self):
        st = dict(changes=[list(c) for c in self.changes], acc_h=self.acc_h, acc_n=self.acc_n)
        if self.pack:
            st.update(pchanges=[list(c) for c in self.pchanges], pacc_h=self.pacc_h, pacc_n=self.pacc_n)
        return st

    def load(self, st):
        self.changes = [tuple(c) for c in st["changes"]]
        self.acc_h, self.acc_n = st["acc_h"], st["acc_n"]
        if self.pack and "pchanges" in st:
            self.pchanges = [tuple(c) for c in st["pchanges"]]
            self.pacc_h, self.pacc_n = st["pacc_h"], st["pacc_n"]

    def _row(self, row, rng, exact=None, rng_d=None):
        n, L, T1 = self.n, self.L, len(row)
        keys = self.k0 + rng.permutation(self.N_KEYS)[:n]
        vals = self.v0 + rng.integers(0, self.N_VALS, n)
        if exact is None and rng_d is not None:              # драбина: відстань рівно з діапазону рівня
            exact = int(rng.integers(rng_d[0], rng_d[1] + 1))
        if exact is None:
            s = int(rng.integers(0, T1 - 2 * L - self.dist + 1))
            q = int(rng.integers(s + L + self.dist, T1 - L + 1))
        else:                                               # eval: рівно exact слів між блоком фактів і питань
            assert 0 <= exact <= T1 - 2 * L, (exact, T1)
            s = int(rng.integers(0, T1 - 2 * L - exact + 1))
            q = s + L + exact
        kv = np.stack([keys, vals], 1).reshape(-1)
        qi = rng.permutation(n)
        qs = np.stack([keys[qi], vals[qi]], 1).reshape(-1)
        row[s] = row[s + L - 1] = self.mark
        row[s + 1:s + L - 1] = kv
        row[q] = row[q + L - 1] = self.mark
        row[q + 1:q + L - 1] = qs
        if self.doc_eos is not None:                        # межі документів між фактами й питанням → сміття
            seg = row[s:q + L]
            hit = seg == self.doc_eos
            if hit.any():
                seg[hit] = self._filler(rng, int(hit.sum()))
        return q + 1 + 2 * np.arange(n)                     # позиції ключів-питань (у x; y там = значення)

    def __call__(self, xy, p=None, exact=None, n=None, synth=None, pack=None):
        import zlib
        if synth is None:
            synth = n is not None and n < self.synth
        rng_d = None
        if n is not None and p is None:                     # train-батч n: розклад частки і рівень драбини
            p = self.p_at(n)
            rng_d = self.range_at(n) if self.ladder else None
        p = self.p if p is None else p
        if synth and self.pack and n is not None and pack is None:
            return self._short_batch(xy.shape, n)
        xy = np.array(xy, copy=True)
        B, T1 = xy.shape
        rem = np.zeros((B, T1 - 1), np.int8)
        for b in range(B):
            rng = np.random.default_rng([self.seed, zlib.crc32(xy[b].tobytes())])
            if synth and (pack is not None or (self.pack and n is not None)):   # щільні епізоди torch (--remind_pack)
                xy[b] = self._filler(rng, T1)
                if pack is not None:                        # eval: рівно (пар, сміття)
                    pp, f = pack
                else:
                    k = self.plevel_at(n)
                    pp, f = self.plevels[k]
                    if k == len(self.plevels) - 1:          # останній рівень: усі довжини сміття впереміш (torch RECALL_MIX)
                        f = self.pfill[int(rng.integers(0, len(self.pfill)))]
                rem[b, self._pack_row(xy[b], rng, pp, f)] = 1
            elif synth:                                     # рядок — лише сміття + голка (torch recall_long.batch)
                xy[b] = self._filler(rng, T1)
                rem[b, self._row(xy[b], rng, exact, rng_d)] = 1
            elif rng.random() < p:
                ex = exact
                if ex is None and self.near > 0 and self.near_max > 0 and rng.random() < self.near:
                    ex = int(rng.integers(0, min(self.near_max, T1 - 2 * self.L) + 1))
                rem[b, self._row(xy[b], rng, ex, None if ex is not None else rng_d)] = 1
        return xy, {"rem": rem}
