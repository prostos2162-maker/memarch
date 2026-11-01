"""Частоти токенів повного Qwen3.6 на зразку нашої суміші (Polite Chainsaw Coding 200B) — для обрізки до 131k.

Кожне джерело — свій зразок (потоково з Hugging Face), частоти пишуться окремо, плюс відкладені тексти для перевірки.
python tok/sample_counts.py [ціль_токенів_на_одиницю_ваги]
"""
import json
import os
import sys

import numpy as np
from datasets import load_dataset
from huggingface_hub import hf_hub_download
from tokenizers import Tokenizer

OUT = os.path.join(os.path.dirname(__file__), "counts")
UNIT = int(float(sys.argv[1])) if len(sys.argv) > 1 else 2_000_000      # токенів на одиницю ваги
# (назва, датасет, конфіг, поле тексту, фільтр, вага ≈ частка в суміші, у токенах на одиницю)
SOURCES = [
    ("code", "codeparrot/github-code-clean", "all-all", "code", None, 8),             # код різних мов (+ SE-подібне)
    ("frontend", "codeparrot/github-code-clean", "all-all", "code",
     {"language": ["HTML", "CSS", "JavaScript", "TypeScript"]}, 2),
    ("en_edu", "HuggingFaceFW/fineweb-edu", "sample-10BT", "text", None, 4),          # документація + англійська
    ("math", "HuggingFaceTB/finemath", "finemath-4plus", "text", None, 1.5),
    ("uk", "HuggingFaceFW/fineweb-2", "ukr_Cyrl", "text", None, 1.5),                 # 3.5% суміші, вага ×2
    ("pl", "HuggingFaceFW/fineweb-2", "pol_Latn", "text", None, 0.2),
    ("de", "HuggingFaceFW/fineweb-2", "deu_Latn", "text", None, 0.2),
    ("es", "HuggingFaceFW/fineweb-2", "spa_Latn", "text", None, 0.2),
]
HELDOUT = 300                                                                         # текстів на джерело для перевірки


def main():
    os.makedirs(OUT, exist_ok=True)
    tok = Tokenizer.from_file(hf_hub_download("Qwen/Qwen3.6-35B-A3B", "tokenizer.json"))
    V = tok.get_vocab_size(with_added_tokens=True)
    for name, ds_id, cfg, field, filt, w in SOURCES:
        path = os.path.join(OUT, f"{name}.npz")
        if os.path.exists(path):
            print(f"{name}: вже є", flush=True)
            continue
        target = int(UNIT * w)
        if ds_id == "codeparrot/github-code-clean":     # там старий скрипт — беремо автоконвертовану parquet-версію
            ds = load_dataset(ds_id, split="train", streaming=True, revision="refs/convert/parquet", data_dir=cfg)
        else:
            ds = load_dataset(ds_id, cfg, split="train", streaming=True)
        counts = np.zeros(V, dtype=np.int64)
        held, batch, n_tok, n_doc = [], [], 0, 0
        for ex in ds:
            if filt and not all(ex.get(k) in v for k, v in filt.items()):
                continue
            t = ex[field]
            if not t:
                continue
            if len(held) < HELDOUT:
                held.append(t)
                continue
            batch.append(t[:200_000])
            if len(batch) == 256:
                for enc in tok.encode_batch(batch, add_special_tokens=False):
                    counts += np.bincount(enc.ids, minlength=V)
                    n_tok += len(enc.ids)
                n_doc += len(batch)
                batch = []
                if n_tok >= target:
                    break
        np.savez(path, counts=counts, n_tok=n_tok, n_doc=n_doc)
        with open(os.path.join(OUT, f"{name}_heldout.jsonl"), "w", encoding="utf-8") as f:
            for t in held:
                f.write(json.dumps({"text": t}, ensure_ascii=False) + "\n")
        print(f"{name}: {n_tok / 1e6:.1f}M токенів, {n_doc} документів, різних токенів {int((counts > 0).sum())}", flush=True)


if __name__ == "__main__":
    main()
    os._exit(0)          # datasets лишає фонові потоки
