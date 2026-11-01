"""Токенізація джерел (HF parquet або jsonl.zst за посиланнями, як HPLT) обрізаним токенізатором → стиснуті шматки → Диск.

  * файли діляться між машинами: --worker i --workers n (кожна бере свою частину);
  * перетини між джерелами — за URL: --exclude-urls (хеші URL уже взятих джерел) пропускаються,
    --emit-urls пише хеші URL цього джерела для наступних;
  * токени + <|endoftext|> між документами, шматки по --shard-tokens, uint32 → zstd (~1.05 байта на токен);
  * маніфест на кожен шматок (джерело, ревізія, файли, токени); файл вважається обробленим лише після запису
    його токенів у шматок, а для великих jsonl.zst зберігається номер рядка — обрив машини нічого не губить;
  * заливка на Диск через rclone (--remote), локальна копія видаляється.
python data/tokenize_hf.py --dataset HuggingFaceTB/finemath --config finemath-4plus --name finemath4
python data/tokenize_hf.py --urls data/hplt_ukr.txt --name hplt_uk --hplt --target-tokens 7e9
"""
import argparse
import hashlib
import io
import json
import os
import random
import subprocess
import time
import urllib.request

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import zstandard as zstd
from huggingface_hub import HfApi, hf_hub_download
from tokenizers import Tokenizer

HERE = os.path.dirname(os.path.abspath(__file__))
TOK = os.path.join(HERE, "..", "tok", "qwen36_pruned131k", "tokenizer.json")


def url_hash(u):
    return int.from_bytes(hashlib.blake2b(u.encode("utf-8", "ignore"), digest_size=8).digest(), "little")


def list_files(dataset, config, revision):
    """Parquet-файли конфігу, у сталому випадковому порядку (різні дампи впереміш)."""
    files = [f for f in HfApi().list_repo_files(dataset, repo_type="dataset", revision=revision)
             if f.endswith(".parquet") and "/test/" not in f]            # тестова частина — не для тренування
    if config:
        sub = [f for f in files if f.startswith(f"{config}/") or f"/{config}/" in f]
        files = sub or files
    files.sort()
    random.Random(1234).shuffle(files)
    return files


def parquet_batches(a, fn, revision, out, skip_rows=0):
    """Рядки parquet пачками; skip_rows — уже оброблені (великий файл міг потрапити в шматок наполовину)."""
    local = hf_hub_download(a.dataset, fn, repo_type="dataset", revision=revision, local_dir=os.path.join(out, "dl"))
    pf = pq.ParquetFile(local)
    cols = [c for c in ("text", "url", "score") if c in pf.schema_arrow.names]
    seen = 0
    for rb in pf.iter_batches(batch_size=2048, columns=cols):
        seen += rb.num_rows
        if seen <= skip_rows:
            continue
        if seen - rb.num_rows < skip_rows:
            rb = rb.slice(skip_rows - (seen - rb.num_rows))
        d = rb.to_pydict()
        keep = [i for i in range(len(d["text"])) if a.min_score is None or "score" not in d or d["score"][i] >= a.min_score]
        yield [d["text"][i] for i in keep], [d["url"][i] for i in keep] if "url" in d else None, rb.num_rows
    getattr(pf, "close", lambda: None)()                 # Windows не дає видалити відкритий файл
    del pf
    try:
        os.remove(local)                                            # кеш не роздуваємо
    except OSError:
        pass


NOT_CODE = {"Markdown", "Text", "JSON", "YAML", "TOML", "XML", "CSV", "reStructuredText", "INI", "SVG", None}


def stack_doc(r, max_file=200_000, max_repo=4_000_000):
    """The Stack v3: один документ на репозиторій, як у Qwen-Coder — <|repo_name|>шлях, далі файли підряд
    <|file_sep|>шлях\\nзміст. Ліцензії (рішення Стефана): дозвільні + без ліцензії; vendor-файли й великі — геть."""
    parts, n, code = [f"<|repo_name|>{r['repo_path']}\n"], 0, False
    for f in r["files"] or []:
        if f["is_vendor"] or f["license_type"] not in ("permissive", "no_license"):
            continue
        c = f["content"] or ""
        if not c or len(c) > max_file:
            continue
        if n + len(c) > max_repo:
            break
        parts.append(f"<|file_sep|>{f['file_path']}\n{c}\n")
        n += len(c)
        code |= f["language"] not in NOT_CODE
    # якість: хоч один файл коду і не порожнє репо (самі README на рядок — шум)
    return "".join(parts) if code and n >= 2000 else None


try:
    pa.jemalloc_set_decay_ms(0)                         # pyarrow одразу віддає пам'ять (інакше процес росте до 15+ ГБ)
except Exception:                                       # noqa: BLE001 — збірка без jemalloc
    pass


def stack_batches(a, fn, revision, out, skip_rows=0):
    """Репозиторії The Stack v3 пачками (форки — геть: їхній код майже весь уже є в оригіналі)."""
    local = hf_hub_download(a.dataset, fn, repo_type="dataset", revision=revision, local_dir=os.path.join(out, "dl"))
    pf = pq.ParquetFile(local)
    seen = 0
    for rb in pf.iter_batches(batch_size=16, columns=["repo_path", "files", "github_metadata"]):
        seen += rb.num_rows
        if seen <= skip_rows:
            continue
        if seen - rb.num_rows < skip_rows:
            rb = rb.slice(skip_rows - (seen - rb.num_rows))
        texts, urls = [], []
        for r in rb.to_pylist():
            if (r["github_metadata"] or {}).get("is_fork"):
                continue
            d = stack_doc(r)
            if d:
                texts.append(d)
                urls.append("github.com/" + r["repo_path"])
        n = rb.num_rows
        del rb
        yield texts, urls, n
    getattr(pf, "close", lambda: None)()                 # Windows не дає видалити відкритий файл
    del pf
    try:
        os.remove(local)
    except OSError:
        pass


class ResumableHTTP(io.RawIOBase):
    """Потік байтів за посиланням, який після обриву докачує з того самого байта (Range), а не з початку:
    файли HPLT по 6–30 ГБ, а сервер рве з'єднання кожні ~15–20 хв."""

    def __init__(self, url, tries=50):
        self.url, self.tries, self.pos = url, tries, 0
        self.r = urllib.request.urlopen(url, timeout=120)
        self.total = int(self.r.headers["Content-Length"])

    def readable(self):
        return True

    def readinto(self, b):
        fails = 0
        while self.pos < self.total:
            try:
                n = self.r.readinto(b)
                if n:
                    self.pos += n
                    return n
                raise IOError("порожнє читання до кінця файлу")
            except Exception as e:                             # noqa: BLE001
                fails += 1
                if fails > self.tries:
                    raise
                print(f"  докачування {self.url} з байта {self.pos}: {type(e).__name__}", flush=True)
                time.sleep(min(60, 5 * fails))
                try:
                    self.r = urllib.request.urlopen(urllib.request.Request(
                        self.url, headers={"Range": f"bytes={self.pos}-"}), timeout=120)
                except Exception:                              # noqa: BLE001
                    pass
        return 0


def jsonl_lines(url, skip_lines, tries=20):
    """Рядки jsonl.zst потоком; обірване з'єднання (сервер HPLT часом рве довгі завантаження) —
    перепідключення й пропуск уже виданих рядків. Видає (номер рядка, рядок)."""
    i, fails = skip_lines, 0
    while True:
        try:
            r = ResumableHTTP(url)
            rd = io.TextIOWrapper(zstd.ZstdDecompressor().stream_reader(r), encoding="utf-8")
            for j, line in enumerate(rd):
                if j < i:
                    continue
                if not line.endswith("\n"):
                    json.loads(line)                           # останній рядок без \n — ок; обрізаний — помилка
                yield j, line
                i, fails = j + 1, 0
            if r.pos < r.total:                                # zstd на обрізаному потоці мовчки закінчується
                raise IOError(f"потік обірвано, недочитано {r.total - r.pos} байт")
            return
        except Exception as e:                                 # noqa: BLE001 — мережа/zstd/розкодування
            fails += 1
            if fails > tries:
                raise
            print(f"  обрив {url} на рядку {i}: {type(e).__name__}; перепідключення {fails}/{tries}", flush=True)
            time.sleep(min(60, 5 * fails))


def jsonl_batches(a, url, skip_lines):
    """jsonl.zst потоком (HPLT): пропуск уже оброблених рядків; фільтри HPLT — keep і без машинного перекладу."""
    texts, urls, n = [], [], 0
    for _, line in jsonl_lines(url, skip_lines):
        n += 1
        d = json.loads(line)
        if a.hplt:
            if d.get("filter", "keep") != "keep":
                continue
            reg = d.get("web-register")
            if isinstance(reg, str):
                try:
                    reg = json.loads(reg.replace("'", '"'))
                except ValueError:
                    reg = {}
            if (reg or {}).get("MT", 0) > a.max_mt:
                continue
        texts.append(d["text"])
        urls.append(d.get("u") or d.get("url") or "")
        if n >= 2048:
            yield texts, urls, n
            texts, urls, n = [], [], 0
    if n:
        yield texts, urls, n


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default=None)
    p.add_argument("--config", default=None)
    p.add_argument("--revision", default=None)
    p.add_argument("--urls", default=None, help="файл зі списком посилань на jsonl.zst (замість --dataset)")
    p.add_argument("--stack", action="store_true", help="The Stack v3: документ = репозиторій, фільтр ліцензій")
    p.add_argument("--hplt", action="store_true", help="фільтри HPLT: filter=keep, машинний переклад <= --max-mt")
    p.add_argument("--max-mt", type=float, default=0.5)
    p.add_argument("--name", required=True, help="назва джерела в сховищі")
    p.add_argument("--worker", type=int, default=0)
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--max-files", type=int, default=0, help="0 — усі файли своєї частини")
    p.add_argument("--target-tokens", type=float, default=0, help="зупинитись після стількох токенів (на цю машину)")
    p.add_argument("--min-score", type=float, default=None, help="поле score (FineWeb-Edu) — нижче не брати")
    p.add_argument("--shard-tokens", type=float, default=1e9)
    p.add_argument("--skip-files", default=None, help="JSON-список файлів, уже зроблених деінде (напр. з маніфестів)")
    p.add_argument("--exclude-urls", default=None, help=".npy з хешами URL, які вже взято з інших джерел")
    p.add_argument("--emit-urls", action="store_true", help="зберегти хеші URL цього джерела")
    p.add_argument("--workdir", default=os.path.join(HERE, "..", "_tmp", "tokenize"))
    p.add_argument("--remote", default="remote:corpus/tokens", help="rclone-ціль; '' — не заливати")
    a = p.parse_args()
    if a.urls:
        revision = "urls:" + os.path.basename(a.urls)
        files = [l.strip() for l in open(a.urls) if l.strip()]
    else:
        revision = a.revision or HfApi().dataset_info(a.dataset).sha
        files = list_files(a.dataset, a.config, revision)
    files = files[a.worker::a.workers]
    if a.max_files:
        files = files[:a.max_files]
    out = os.path.join(a.workdir, a.name, f"w{a.worker:03d}")
    os.makedirs(out, exist_ok=True)
    state_path = os.path.join(out, "state.json")
    state = {"done": [], "lines": {}, "shard": 0, "tokens": 0, "docs": 0, "skipped": 0}
    if os.path.exists(state_path):
        state.update(json.load(open(state_path)))
    tok = Tokenizer.from_file(TOK)
    eos = tok.token_to_id("<|endoftext|>")
    excl = set(np.load(a.exclude_urls).tolist()) if a.exclude_urls else set()
    print(f"{a.name} w{a.worker}/{a.workers}: файлів {len(files)}, ревізія {revision[:24]}, виключених URL {len(excl)}, "
          f"уже {state['tokens'] / 1e9:.2f}B токенів", flush=True)
    cctx = zstd.ZstdCompressor(level=6, threads=-1)
    buf, urls = [], []
    pend = {"files": [], "lines": {}}                   # що потрапить у стан лише після запису шматка
    buf_n, t0, tok0 = 0, time.time(), state["tokens"]

    def save_state():
        tmp = state_path + ".tmp"
        json.dump(state, open(tmp, "w"))
        os.replace(tmp, state_path)

    def upload(name, res):
        # заливка часом зависає (ліміт Google Drive на спільний client_id rclone) — тайм-аут на з'єднання і на всю
        # спробу, далі повтор; іде у фоні, нарізка тим часом продовжується
        for attempt in range(6):
            try:
                r = subprocess.run(["rclone", "move", out, f"{a.remote}/{a.name}", "--include", name + ".*",
                                    "--drive-chunk-size", "64M", "--timeout", "2m", "--contimeout", "30s",
                                    "-q"], capture_output=True, text=True, timeout=1200)
                if r.returncode == 0:
                    res["ok"] = True
                    return
                err = r.stderr[-300:]
            except subprocess.TimeoutExpired:
                err = "тайм-аут 20 хв"
            print(f"rclone помилка (спроба {attempt + 1}/6): {err}", flush=True)
            time.sleep(30)

    inflight = {}                                       # шматок, що заливається: потік + що внести в стан після заливки
    next_shard = state["shard"]

    def commit_inflight():
        """Дочекатись заливки попереднього шматка і лише тоді внести його в стан (обрив — нічого не губиться)."""
        if not inflight:
            return
        inflight["thread"].join()
        if a.remote and not inflight["res"].get("ok"):
            print("rclone: шматок не залито після 6 спроб — зупинка", flush=True)
            raise SystemExit(1)
        c = inflight["commit"]
        state["done"] += c["files"]
        state["lines"].update(c["lines"])
        state["shard"] += c["shards"]
        state["tokens"] += c["tokens"]
        save_state()
        inflight.clear()

    def flush_shard(final=False):
        nonlocal buf, buf_n, next_shard
        commit = {"files": pend["files"], "lines": dict(pend["lines"]), "shards": 0, "tokens": 0}
        name = None
        if buf_n:
            arr = np.concatenate(buf).astype(np.uint32)
            name = f"{a.name}_w{a.worker:03d}_{next_shard:05d}"
            path = os.path.join(out, name + ".u32.zst")
            with open(path, "wb") as f:
                f.write(cctx.compress(arr.tobytes()))
            man = {"source": a.name, "dataset": a.dataset or a.urls, "config": a.config, "revision": revision,
                   "files": sorted(set(pend["files"]) | set(pend["lines"])), "tokens": int(arr.size),
                   "tokenizer": "qwen36_pruned131k", "dtype": "uint32", "eos": eos, "bytes": os.path.getsize(path)}
            json.dump(man, open(os.path.join(out, name + ".json"), "w"), indent=1)
            print(f"  шматок {name}: {arr.size / 1e6:.0f}M токенів, {man['bytes'] / arr.size:.2f} байт/токен", flush=True)
            commit["shards"], commit["tokens"] = 1, buf_n
            next_shard += 1
        pend["files"], pend["lines"] = [], {}
        buf, buf_n = [], 0
        commit_inflight()                               # попередній шматок має бути на Диску до наступного
        res = {}
        th = __import__("threading").Thread(target=upload, args=(name, res), daemon=True) if (a.remote and name) else None
        if th:
            th.start()
        else:
            res["ok"] = True
            th = __import__("threading").Thread(target=lambda: None)
            th.start()
        inflight.update(thread=th, res=res, commit=commit)
        if final:
            commit_inflight()

    skip = set(json.load(open(a.skip_files))) if a.skip_files else set()
    for fn in files:
        if fn in state["done"] or fn in skip:
            continue
        lines = state["lines"].get(fn, 0)
        gen = (jsonl_batches(a, fn, lines) if a.urls else stack_batches(a, fn, revision, out, lines) if a.stack
               else parquet_batches(a, fn, revision, out, lines))
        for texts, burls, nrows in gen:
            lines += nrows
            if burls is not None:
                keep = []
                for t, u in zip(texts, burls):
                    h = url_hash(u)
                    if h in excl:
                        state["skipped"] += 1
                        continue
                    keep.append((t, h))
                texts = [t for t, _ in keep]
                if a.emit_urls:
                    urls.extend(h for _, h in keep)
            if texts:
                enc = getattr(tok, "encode_batch_fast", tok.encode_batch)   # без зміщень/рядків: ~в 10 разів менше пам'яті
                for e in enc(texts, add_special_tokens=False):
                    buf.append(np.asarray(e.ids + [eos], dtype=np.uint32))
                    buf_n += len(e.ids) + 1
                state["docs"] += len(texts)
            pend["lines"][fn] = lines                    # позиція у файлі (рядки jsonl / parquet)
            if buf_n >= a.shard_tokens:
                flush_shard()
            if a.target_tokens and state["tokens"] + inflight.get("commit", {}).get("tokens", 0) + buf_n >= a.target_tokens:
                break
        else:
            pend["files"].append(fn)                     # файл дочитано до кінця
            pend["lines"].pop(fn, None)
        el = time.time() - t0
        tot = state["tokens"] + inflight.get("commit", {}).get("tokens", 0) + buf_n
        print(f"{fn}: усього {tot / 1e9:.2f}B токенів, {state['docs']} док., дублікатів пропущено {state['skipped']}, "
              f"{(tot - tok0) / max(el, 1) / 1e6:.2f}M ток/с", flush=True)
        if a.target_tokens and tot >= a.target_tokens:
            break
    flush_shard(final=True)
    if a.emit_urls and urls:
        np.save(os.path.join(out, f"{a.name}_urls_w{a.worker:03d}.npy"), np.asarray(urls, dtype=np.uint64))
        if a.remote:
            subprocess.run(["rclone", "copy", out, f"{a.remote}/{a.name}", "--include", "*_urls_*.npy", "-q"])
    print(f"ГОТОВО {a.name} w{a.worker}: {state['tokens'] / 1e9:.2f}B токенів, {state['docs']} документів", flush=True)


if __name__ == "__main__":
    main()
