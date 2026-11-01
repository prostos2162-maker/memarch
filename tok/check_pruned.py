"""Перевірка обрізаного токенізатора на відкладених текстах кожного джерела.

  * текст → токени → текст має дати рівно той самий текст (байт у байт);
  * у скільки разів більше токенів, ніж у повного Qwen3.6 (по джерелах);
  * спецтокени (чат, FIM, think) кодуються одним токеном кожен і мають ті самі рядки.
python tok/check_pruned.py
"""
import glob
import json
import os

from huggingface_hub import hf_hub_download
from tokenizers import Tokenizer

HERE = os.path.dirname(__file__)
full = Tokenizer.from_file(hf_hub_download("Qwen/Qwen3.6-35B-A3B", "tokenizer.json"))
pruned = Tokenizer.from_file(os.path.join(HERE, "qwen36_pruned131k", "tokenizer.json"))
old2new = {int(k): v for k, v in json.load(open(os.path.join(HERE, "qwen36_pruned131k", "old2new.json"))).items()}
print(f"словник: повний {full.get_vocab_size()} → обрізаний {pruned.get_vocab_size()}")
bad_total = 0
for path in sorted(glob.glob(os.path.join(HERE, "counts", "*_heldout.jsonl"))):
    name = os.path.basename(path).replace("_heldout.jsonl", "")
    texts = [json.loads(l)["text"] for l in open(path, encoding="utf-8")]
    nf = sum(len(e.ids) for e in full.encode_batch(texts, add_special_tokens=False))
    enc = pruned.encode_batch(texts, add_special_tokens=False)
    npr = sum(len(e.ids) for e in enc)
    bad = sum(pruned.decode(e.ids, skip_special_tokens=False) != t for e, t in zip(enc, texts))
    bad_total += bad
    print(f"{name:9s} текстів {len(texts):4d}  токенів: повний {nf:9d}  обрізаний {npr:9d}  (+{100 * (npr / nf - 1):5.2f}%)  "
          f"не збіглось при розкодуванні: {bad}")
specials = [a["content"] for a in json.load(open(os.path.join(HERE, "qwen36_pruned131k", "tokenizer.json"), encoding="utf-8"))["added_tokens"]]
ok = all(len(pruned.encode(s, add_special_tokens=False).ids) == 1 for s in specials)
same = all(old2new[full.token_to_id(s)] == pruned.token_to_id(s) for s in specials)
print(f"спецтокенів {len(specials)}: кожен — один токен: {ok}; відповідність old2new правильна: {same}")
print("ПЕРЕВІРКА ПРОЙДЕНА" if bad_total == 0 and ok and same else "Є ПОМИЛКИ")
