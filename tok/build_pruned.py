"""Обрізаний токенізатор Qwen3.6 → 131 072 токени під суміш Polite Chainsaw (частоти з sample_counts.py).

Як обрізати BPE і нічого не зламати:
  * лишаються всі 256 базових байтових токенів (будь-який текст закодовується) і всі спецтокени;
  * токени беруться за спаданням частоти (частка в кожному джерелі × вага джерела); разом з токеном — усі його
    «предки» в злиттях, інакше BPE його ніколи не утворить;
  * лишається злиття (a, b), лише якщо a, b і a+b усі лишились — тож токенізатор видає тільки наші токени;
  * нові номери йдуть підряд; old2new.json — відповідність для дистиляції (логіти вчителя → наш словник).
python tok/build_pruned.py
"""
import json
import os

import numpy as np
from huggingface_hub import hf_hub_download

HERE = os.path.dirname(__file__)
COUNTS = os.path.join(HERE, "counts")
OUT = os.path.join(HERE, "qwen36_pruned131k")
BUDGET = 131_072                                  # звичайних токенів (спецтокени — понад це)
WEIGHTS = {"code": 8, "frontend": 2, "en_edu": 4, "math": 1.5, "uk": 1.5, "pl": 0.2, "de": 0.2, "es": 0.2}


def byte_tokens():
    """256 базових токенів byte-level BPE (символи, якими GPT-2/Qwen позначають байти)."""
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs, n = bs[:], 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return {chr(c) for c in cs}


def main():
    src = hf_hub_download("Qwen/Qwen3.6-35B-A3B", "tokenizer.json")
    t = json.load(open(src, encoding="utf-8"))
    vocab = t["model"]["vocab"]                   # токен → номер
    merges = [m.split(" ", 1) if isinstance(m, str) else m for m in t["model"]["merges"]]
    inv = {i: s for s, i in vocab.items()}
    V = max(inv) + 1
    # частота = сума за джерелами (частка токена в джерелі × вага), щоб велике джерело не задавило мале
    score = np.zeros(V)
    for name, w in WEIGHTS.items():
        c = np.load(os.path.join(COUNTS, f"{name}.npz"))["counts"][:V].astype(np.float64)
        score += w * c / max(c.sum(), 1)
    parents = {}                                  # токен → (a, b), з якого злиття він утворюється
    for a, b in merges:
        parents.setdefault(a + b, (a, b))
    keep = set(byte_tokens()) & set(vocab)
    assert len(keep) == 256, len(keep)

    def add(tok):
        stack = [tok]
        while stack:
            x = stack.pop()
            if x in keep:
                continue
            keep.add(x)
            if x in parents:
                stack.extend(parents[x])

    for i in np.argsort(-score):
        if len(keep) >= BUDGET:
            break
        if score[i] <= 0:
            break
        add(inv[int(i)])
    # якщо частот бракує — добираємо найраніші злиття (вони найзагальніші)
    for a, b in merges:
        if len(keep) >= BUDGET:
            break
        add(a + b)
    keep = sorted(keep, key=lambda s: vocab[s])   # порядок старих номерів зберігається
    new_vocab = {s: j for j, s in enumerate(keep)}
    ks = set(keep)
    new_merges = [f"{a} {b}" for a, b in merges if a in ks and b in ks and (a + b) in ks]
    old2new = {vocab[s]: new_vocab[s] for s in keep}
    nid = len(new_vocab)
    for at in t["added_tokens"]:                  # спецтокени — у кінці, підряд
        old2new[at["id"]] = nid
        at["id"] = nid
        nid += 1
    t["model"]["vocab"] = new_vocab
    t["model"]["merges"] = new_merges
    for part in ("post_processor",):               # шаблони з номерами спецтокенів (якщо є)
        if t.get(part):
            s = json.dumps(t[part])
            for old, new in old2new.items():
                s = s.replace(f'"id": {old}', f'"id": {new}')
            t[part] = json.loads(s)
    os.makedirs(OUT, exist_ok=True)
    json.dump(t, open(os.path.join(OUT, "tokenizer.json"), "w", encoding="utf-8"), ensure_ascii=False)
    json.dump({str(k): v for k, v in old2new.items()}, open(os.path.join(OUT, "old2new.json"), "w"))
    for f in ("tokenizer_config.json", "chat_template.jinja", "generation_config.json"):
        try:
            p = hf_hub_download("Qwen/Qwen3.6-35B-A3B", f)
            s = open(p, encoding="utf-8").read()
            open(os.path.join(OUT, f), "w", encoding="utf-8").write(s)
        except Exception:
            pass
    print(f"токенів {len(new_vocab)} + спецтокенів {len(t['added_tokens'])} = {nid}; злиттів {len(new_merges)} з {len(merges)}")


if __name__ == "__main__":
    main()
