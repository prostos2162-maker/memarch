"""Очищення від бенчмарків: збіги 13 токенів (нашим токенізатором) з HumanEval, MBPP, MMLU, MMLU-Pro, GSM8K, MATH, ARC,
HellaSwag. Тексти шматків не розкодовуються.
  python data/decontam.py build                      -> data/bench13.npz (відсортовані хеші 13-грам + з якої задачі кожна)
  python data/decontam.py scan --remote remote:corpus/tokens/<джерело> [--worker i --workers n]
     -> поруч із кожним шматком `<шматок>.contam.npz`: пари (doc, gram) — номер документа в шматку і номер 13-грами в
        bench13. Розріджено й без порогів: стоп-список шаблонних 13-грам (ті, що трапляються в тисячах документів)
        і поріг «скільки збігів = брудний» вирішуються потім, без повторного сканування."""
import argparse, os, subprocess, sys
import numpy as np, zstandard as zstd

HERE = os.path.dirname(os.path.abspath(__file__))
TOK = os.path.join(HERE, "..", "tok", "qwen36_pruned131k", "tokenizer.json")
BENCH = os.path.join(HERE, "bench13.npz")
EOS, N, M = 131072, 13, np.uint64(0x9E3779B97F4A7C15)
CHUNK = 64_000_000

# (набір, конфіг, спліти, поля); поле зі списком (choices, endings…) розгортається
SETS = [("openai/openai_humaneval", None, ["test"], ["prompt", "canonical_solution", "test"]),
        ("google-research-datasets/mbpp", "full", ["train", "validation", "test", "prompt"], ["text", "code", "test_list"]),
        ("cais/mmlu", "all", ["test", "validation", "dev"], ["question", "choices"]),
        ("TIGER-Lab/MMLU-Pro", None, ["test", "validation"], ["question", "options"]),
        ("openai/gsm8k", "main", ["test"], ["question", "answer"]),
        ("allenai/ai2_arc", "ARC-Challenge", ["test", "validation"], ["question", "choices"]),
        ("allenai/ai2_arc", "ARC-Easy", ["test", "validation"], ["question", "choices"]),
        ("Rowan/hellaswag", None, ["validation", "test"], ["ctx", "endings"])] + \
       [("EleutherAI/hendrycks_math", c, ["test"], ["problem", "solution"])
        for c in ("algebra", "counting_and_probability", "geometry", "intermediate_algebra", "number_theory",
                  "prealgebra", "precalculus")]


def grams(a):
    """Хеш кожних 13 підряд токенів (uint64, переповнення — частина хешу); len(a)-12 значень."""
    a = a.astype(np.uint64)
    h = np.zeros(len(a) - N + 1, dtype=np.uint64)
    with np.errstate(over="ignore"):
        for k in range(N):
            h = h * M + a[k:len(a) - N + 1 + k]
    return h


def texts(v):
    if isinstance(v, dict):                     # ARC: choices = {"text": [...], "label": [...]}
        v = v.get("text", [])
    return [str(x) for x in v] if isinstance(v, list) else [str(v)] if v is not None else []


def build():
    from datasets import load_dataset
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(TOK)
    hs, items, item_src, names = [], [], [], []           # items: номер задачі для кожної 13-грами
    for si, (ds, cfg, splits, fields) in enumerate(SETS):
        names.append(f"{ds}:{cfg}")
        for sp in splits:
            try:
                d = load_dataset(ds, cfg, split=sp)
            except Exception as e:
                print(f"  пропуск {ds} {cfg} {sp}: {type(e).__name__} {str(e)[:120]}", flush=True)
                continue
            fs, it = [], []
            for r in d:
                for f in fields:
                    for t in texts(r.get(f)):
                        fs.append(t); it.append(len(item_src))
                item_src.append(si)
            n = 0
            for e, k in zip(tok.encode_batch_fast(fs), it):
                if len(e.ids) >= N:
                    g = grams(np.asarray(e.ids, dtype=np.uint32))
                    hs.append(g); items.append(np.full(len(g), k, dtype=np.uint32)); n += len(g)
            print(f"{ds} {cfg} {sp}: {len(d)} прикладів, {n} 13-грам", flush=True)
    h, items = np.concatenate(hs), np.concatenate(items)
    item_src = np.asarray(item_src, dtype=np.uint8)
    h, i = np.unique(h, return_index=True)
    item = items[i]
    np.savez(BENCH, h=h, src=item_src[item], item=item, item_src=item_src,
             item_grams=np.bincount(item, minlength=len(item_src)).astype(np.uint32), names=np.asarray(names))
    print(f"збережено {len(h)} унікальних 13-грам -> {BENCH}")


_bits = {}


def bitmap(bh):
    if id(bh) not in _bits:
        top = (bh >> np.uint64(34)).astype(np.int64)
        b = np.zeros(1 << 27, dtype=np.uint8)
        np.bitwise_or.at(b, top >> 3, (1 << (top & 7)).astype(np.uint8))
        _bits[id(bh)] = b
    return _bits[id(bh)]


def match(arr, bh):
    """Пари (doc, gram) для всіх 13-грам шматка, що є серед бенчмарків."""
    assert arr[-1] == EOS, "шматок має закінчуватись <|endoftext|>"
    ends = np.flatnonzero(arr == EOS)
    docs, gs = [], []
    bits = bitmap(bh)
    for s in range(0, len(arr) - N + 1, CHUNK):          # шматки з перекриттям N-1, щоб не губити 13-грами на межі
        h = grams(arr[s:s + CHUNK + N - 1])
        top = (h >> np.uint64(34)).astype(np.int64)     # швидкий фільтр: ~0.5% хибних, далі точна перевірка
        pos = np.flatnonzero((bits[top >> 3] >> (top & 7).astype(np.uint8)) & 1)
        j = np.searchsorted(bh, h[pos])
        j[j == len(bh)] = 0
        ok = bh[j] == h[pos]
        pos, j = pos[ok], j[ok]
        docs.append(np.searchsorted(ends, s + pos).astype(np.uint32))
        gs.append(j.astype(np.uint32))
    return np.concatenate(docs), np.concatenate(gs), len(ends)


def scan(a):
    b = np.load(BENCH)
    bh = b["h"]
    os.makedirs(a.workdir, exist_ok=True)
    ls = lambda inc: subprocess.run(["rclone", "lsf", a.remote, "--include", inc], capture_output=True,
                                    text=True).stdout.split()
    names, have = sorted(ls("*.u32.zst")), set(ls("*.contam.npz"))
    for nm in names[a.worker::a.workers]:
        out = nm.replace(".u32.zst", ".contam.npz")
        if out in have:
            continue
        loc = os.path.join(a.workdir, nm)
        subprocess.run(["rclone", "copy", f"{a.remote}/{nm}", a.workdir, "-q"], check=True)
        arr = np.frombuffer(zstd.ZstdDecompressor().decompress(open(loc, "rb").read(), max_output_size=8_000_000_000),
                            dtype=np.uint32)
        docs, gs, n = match(arr, bh)
        np.savez(os.path.join(a.workdir, out), doc=docs, gram=gs, n_docs=n)
        subprocess.run(["rclone", "move", os.path.join(a.workdir, out), a.remote, "-q"], check=True)
        os.remove(loc)
        nd = len(np.unique(docs))
        print(f"{nm}: {n} док., зі збігами {nd} ({nd / max(n, 1):.3%}), збігів {len(docs)}", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=["build", "scan"])
    p.add_argument("--remote")
    p.add_argument("--worker", type=int, default=0)
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--workdir", default=os.path.join(HERE, "..", "_tmp", "decontam"))
    a = p.parse_args()
    build() if a.mode == "build" else scan(a)
