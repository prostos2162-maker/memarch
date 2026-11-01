"""Довгі книжки (PG-19): дотренування повної моделі і «голка в копиці» на 16k–128k.

1. Книжки PG-19 токенізуються GPT-2 (межі книжок зберігаються).
2. Модель тренується на вікнах по --seq токенів, вирізаних з ОДНІЄЇ книжки.
3. Голка: у книжку на випадковій глибині вставляється «The secret code of <ім'я> is <число>.»,
   наприкінці — «The secret code of <ім'я> is»; чи передбачає модель перший токен числа.
   Довжини: --eval_lens. Глибина: випадкова в межах першої половини тексту (далеко за вікном).

python pg19_needle.py --models Fv2 Fv2_nomem --steps 2000 --seq 16384
"""
import argparse
import json
import os
import time

import numpy as np
import torch
import torch.nn.functional as F

import fastloop
from lm import build

NAMES = ["Marta", "Oleh", "Iryna", "Taras", "Sofia", "Bohdan", "Nadia", "Yurii", "Olena", "Maksym"]


def prepare_books(path, n_books, min_tokens):
    if os.path.exists(path):
        return
    import tiktoken
    from datasets import load_dataset
    enc = tiktoken.get_encoding("gpt2")
    ds = load_dataset("emozilla/pg19", split="train", streaming=True)
    toks, offs = [], [0]
    for ex in ds:
        ids = enc.encode_ordinary(ex["text"])
        if len(ids) < min_tokens:
            continue
        toks.append(np.asarray(ids, dtype=np.uint16))
        offs.append(offs[-1] + len(ids))
        print(f"книжок {len(toks)}  токенів {offs[-1] / 1e6:.0f}M", flush=True)
        if len(toks) >= n_books:
            break
    np.savez(path, tokens=np.concatenate(toks), offs=np.asarray(offs))


class Books:
    def __init__(self, path, seed):
        z = np.load(path)
        self.tc = z["tokens"].astype(np.int32)              # копія на процесорі: тренувальні вікна збираються тут
        self.t = torch.from_numpy(self.tc).cuda()
        self.offs = z["offs"]
        self.rng = np.random.default_rng(seed)

    def window(self, B, T):
        """B вікон по T+1 токенів, кожне цілком з однієї книжки."""
        lens = np.diff(self.offs)
        ok = np.nonzero(lens > T + 1)[0]
        out = []
        for _ in range(B):
            b = self.rng.choice(ok)
            s = self.offs[b] + self.rng.integers(0, lens[b] - T - 1)
            out.append(self.t[s:s + T + 1])
        return torch.stack(out).long()

    def window_np(self, B, T):
        """Те саме на процесорі (numpy): голки вставляються тут, а на відеокарту — одна пересилка на крок."""
        lens = np.diff(self.offs)
        ok = np.nonzero(lens > T + 1)[0]
        out = np.empty((B, T + 1), dtype=np.int32)
        for i in range(B):
            b = self.rng.choice(ok)
            s = self.offs[b] + self.rng.integers(0, lens[b] - T - 1)
            out[i] = self.tc[s:s + T + 1]
        return out


def make_needle(hay, rng, enc, near=False, answer=False, near_off=1000):
    """Вставляє «The secret code of <ім'я> is <число>.» у текст і питання в кінці; довжина не змінюється.
    near=True — голка за ~1000 токенів до питання (у межах вікна легкої частини: контроль копіювання).
    answer=True — для тренування: після питання стоїть відповідь, тож звичайний loss вчить пригадувати."""
    name = NAMES[int(rng.integers(len(NAMES)))]
    code = int(rng.integers(1000, 9999))
    needle = enc.encode_ordinary(f" The secret code of {name} is {code}.")
    ask = enc.encode_ordinary(f" The secret code of {name} is")
    target = enc.encode_ordinary(f" {code}")[0]
    if answer:
        ask = ask + [target]
    L = len(hay)
    hay_len = L - len(needle) - len(ask)
    depth = hay_len - near_off if near else int(rng.integers(hay_len // 10, hay_len // 2))   # далеко за вікном
    dev = hay.device
    x = torch.cat([hay[:depth], torch.tensor(needle, device=dev), hay[depth:hay_len], torch.tensor(ask, device=dev)])
    return x, target


def make_needles_np(hay, rng, enc, k=3, span=None):
    """make_needles на процесорі (numpy) — без дрібних пересилок на відеокарту."""
    names = rng.choice(len(NAMES), k, replace=False)
    codes = rng.integers(1000, 9999, k)
    needles = [enc.encode_ordinary(f" The secret code of {NAMES[i]} is {c}.") for i, c in zip(names, codes)]
    qa = sum((enc.encode_ordinary(f" The secret code of {NAMES[i]} is") + [enc.encode_ordinary(f" {c}")[0]]
              for i, c in zip(names, codes)), [])
    hay_len = len(hay) - sum(map(len, needles)) - len(qa)
    if span is None:
        depths = np.sort(rng.integers(hay_len // 10, hay_len // 2, k))
    else:
        span = int(min(max(span, 40), hay_len - 1))
        depths = np.sort(hay_len - rng.integers(20, span, k))
    parts, prev = [], 0
    for dpt, nd in zip(depths, needles):
        parts += [hay[prev:dpt], np.asarray(nd, dtype=np.int32)]
        prev = dpt
    parts += [hay[prev:hay_len], np.asarray(qa, dtype=np.int32)]
    return np.concatenate(parts)


def make_needles(hay, rng, enc, k=3, span=None):
    """Для тренування: k голок з різними іменами і k питань з відповідями в кінці. Довжина не змінюється.
    span — найбільша відстань голки від питань (поступове ускладнення: спершу у вікні, потім далі);
    None — як раніше, у першій половині тексту."""
    names = rng.choice(len(NAMES), k, replace=False)
    codes = rng.integers(1000, 9999, k)
    needles = [enc.encode_ordinary(f" The secret code of {NAMES[i]} is {c}.") for i, c in zip(names, codes)]
    qa = sum((enc.encode_ordinary(f" The secret code of {NAMES[i]} is") + [enc.encode_ordinary(f" {c}")[0]]
              for i, c in zip(names, codes)), [])
    L = len(hay)
    hay_len = L - sum(map(len, needles)) - len(qa)
    if span is None:
        depths = np.sort(rng.integers(hay_len // 10, hay_len // 2, k))
    else:
        span = int(min(max(span, 40), hay_len - 1))
        depths = np.sort(hay_len - rng.integers(20, span, k))
    dev = hay.device
    parts, prev = [], 0
    for dpt, nd in zip(depths, needles):
        parts += [hay[prev:dpt], torch.tensor(nd, device=dev)]
        prev = dpt
    parts += [hay[prev:hay_len], torch.tensor(qa, device=dev)]
    return torch.cat(parts)


def needle_eval(model, books, lens, n, enc, near=False, near_off=1000):
    model.eval()
    rng = np.random.default_rng(123)
    res = {}
    with torch.no_grad(), torch.autocast("cuda", torch.bfloat16):
        for L in lens:
            hit = tot = 0
            try:
                hit, tot = _needle_len(model, books, L, n, enc, rng, near, near_off)
            except Exception as e:      # одна довжина впала (напр. пам'ять) — решта результатів не губиться
                print(f"NEEDLE{'(вікно)' if near else ''}  довжина {L:7d}: ПОМИЛКА {type(e).__name__}: {str(e)[:200]}", flush=True)
                torch.cuda.empty_cache()
                continue
            res[L] = hit / tot
            print(f"NEEDLE{'(вікно)' if near else ''}  довжина {L:7d}: {hit}/{tot} = {hit / tot:.0%}", flush=True)
    model.train()
    return res


def _needle_len(model, books, L, n, enc, rng, near=False, near_off=1000):
    hit = tot = 0
    for i in range(n):
        x, target = make_needle(books.window(1, L)[0, :L], rng, enc, near, near_off=near_off)
        out, _ = model(x[None])
        hn = out[0] if isinstance(out, tuple) else None
        logits = model.head(hn[:, -1]) if hn is not None else out[:, -1]
        hit += int(logits.argmax(-1).item() == target)
        tot += 1
    return hit, tot


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--models", nargs="+", default=["Fv2ck", "Fv2_nomem"])
    p.add_argument("--steps", type=int, default=2000)
    p.add_argument("--seq", type=int, default=16384)
    p.add_argument("--batch", type=int, default=2)
    p.add_argument("--lr", type=float, default=6e-4)
    p.add_argument("--books", type=int, default=1500)
    p.add_argument("--eval_lens", nargs="+", type=int, default=[16384, 32768, 65536, 131072])
    p.add_argument("--n_eval", type=int, default=20)
    p.add_argument("--needle_frac", type=float, default=0.5)
    p.add_argument("--needles", type=int, default=1, help="голок на тренувальне вікно")
    p.add_argument("--curriculum", action="store_true", help="поступово віддаляти голку від питання")
    p.add_argument("--near_off", type=int, default=1000, help="контроль: відстань голки до питання (у межах вікна)")
    a = p.parse_args()
    import tiktoken
    enc = tiktoken.get_encoding("gpt2")
    prepare_books("pg19.npz", a.books, min_tokens=a.seq + 2)
    books = Books("pg19.npz", 1)
    rng = np.random.default_rng(7)
    args = argparse.Namespace(lr=a.lr, steps=a.steps, micro=a.batch, batch=a.batch, seq=a.seq, read_cost=0.01)   # один мікро-батч
    for name in a.models:
        torch.manual_seed(0)
        tr = fastloop.Trainer(name, "AB", args)              # як в інших тестах: оновлення ваг у зворотному проході
        t0 = time.time()
        for step in range(1, a.steps + 1):
            if a.needles > 1:                         # швидкий шлях: усе на процесорі, одна пересилка на крок
                xn = books.window_np(a.batch, a.seq)
                span = None
                if a.curriculum:                     # відстань голки росте: від половини вікна до 90% тексту
                    span = a.near_off + min(1.0, step / (a.steps * 0.6)) * (0.9 * a.seq - a.near_off)
                for r in range(len(xn)):
                    if rng.random() < a.needle_frac:
                        xn[r] = make_needles_np(xn[r], rng, enc, a.needles, span)
                loss = tr.step(torch.from_numpy(xn).pin_memory().to("cuda", non_blocking=True).long(), step - 1)
                if step % 250 == 0:
                    print(f"{name} крок {step} train {float(loss):.3f} {time.time() - t0:.0f}с", flush=True)
                continue
            xy = books.window(a.batch, a.seq)
            for r in range(len(xy)):                 # голки і в тренуванні: модель вчиться цьому завданню
                if rng.random() < a.needle_frac:
                    span = None
                    if a.curriculum:                 # відстань голки росте: від половини вікна до 90% тексту
                        prog = min(1.0, step / (a.steps * 0.6))
                        span = a.near_off + prog * (0.9 * a.seq - a.near_off)
                    xy[r] = (make_needles(xy[r], rng, enc, a.needles, span) if a.needles > 1
                             else make_needle(xy[r], rng, enc, answer=True)[0])
            loss = tr.step(xy, step - 1)
            if step % 250 == 0:
                print(f"{name} крок {step} train {float(loss):.3f} {time.time() - t0:.0f}с", flush=True)
        needle_eval(tr.model, books, a.eval_lens[:1], a.n_eval, enc, near=True, near_off=a.near_off)   # контроль: у вікні
        res = needle_eval(tr.model, books, a.eval_lens, a.n_eval, enc)
        print(f"RESULT {name} " + json.dumps(res), flush=True)
        del tr
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
    os._exit(0)          # datasets лишає фонові потоки — без цього процес не завершується
