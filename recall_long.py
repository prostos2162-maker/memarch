"""Пригадування з-за меж вікна: «Петро 3 Іван 7 …» → сміття довше за вікно → «Петро ?».

Вікно маленької моделі — 64 слова, тож факт можна дістати лише через пам'ять.
Fv2s — повна модель; Fv2s_nomem — та сама без пам'яті (мусить провалитись: задача чесна);
Tsmall — трансформер з повною увагою (стеля).

python recall_long.py --models Fv2s Fv2s_nomem Tsmall
"""
import argparse
import time

import torch
torch._dynamo.config.cache_size_limit = 64          # рівні й перевірка — різні форми; інакше після 8 перекомпіляцій крок без компіляції
import torch.nn.functional as F

from lm import TransformerLM, build

VOCAB = 50304
N_KEYS, N_VALS, FILL0 = 512, 64, 1000


def batch(B, pairs, filler, queries, g):
    keys = torch.rand(B, N_KEYS, generator=g, device="cuda").argsort(-1)[:, :pairs]
    vals = torch.randint(0, N_VALS, (B, pairs), generator=g, device="cuda") + N_KEYS
    kv = torch.stack([keys, vals], -1).view(B, -1)
    fill = torch.randint(FILL0, FILL0 + 20000, (B, filler), generator=g, device="cuda")
    # кожен ключ питаємо рівно раз (без повторів: інакше відповідь на повтор видно у вікні)
    qi = torch.rand(B, pairs, generator=g, device="cuda").argsort(-1)[:, :queries]
    qs = torch.stack([keys.gather(1, qi), vals.gather(1, qi)], -1).view(B, -1)
    seq = torch.cat([kv, fill, qs], 1)
    x, y = seq[:, :-1], torch.full_like(seq[:, 1:], -100)
    ans = 2 * pairs + filler + 2 * torch.arange(queries, device="cuda")      # вхід = ключ запиту
    y[:, ans] = seq[:, ans + 1]
    return x, y, ans


def loss_and_pred(m, x, y, ans):
    """Втрата лише на відповідях (інших міток у задачі немає)."""
    out, cost = m(x)
    hn = out[0] if isinstance(out, tuple) else None
    logits = m.head(hn[:, ans]) if hn is not None else out[:, ans]
    loss = F.cross_entropy(logits.float().reshape(-1, VOCAB), y[:, ans].reshape(-1))
    return loss + 0.01 * cost.float(), logits


@torch.no_grad()
def diagnose(m, a, name):
    """Де губляться факти: чи є картка ключа в пам'яті на момент питання, і чи знаходить її читання."""
    gv = torch.Generator(device="cuda").manual_seed(11)
    for filler in (256, 1024):
        x, y, ans = batch(64, a.pairs, filler, a.pairs, gv)
        with torch.autocast("cuda", torch.bfloat16):
            _, logits = loss_and_pred(m, x, y, ans)
        ok = (logits.argmax(-1) == y[:, ans])
        snaps, _ = m.last_mem
        seg = m.seg
        k = (2 * a.pairs + filler) // seg                       # шматок, де стоять питання
        faces, contents, pos, valid = snaps[min(k, len(snaps) - 1)]
        key_pos = 2 * torch.arange(a.pairs, device="cuda")      # позиції ключів у тексті
        in_mem = ((pos[:, :, None] == key_pos) | (pos[:, :, None] == key_pos + 1)) & valid[:, :, None]
        has = in_mem.any(1).float()                             # (B, pairs): картка ключа або значення є
        # які ключі питали в якому порядку
        q_keys = x[:, ans]
        k_idx = (q_keys[:, :, None] == x[:, key_pos][:, None]).float().argmax(-1)
        has_q = has.gather(1, k_idx)
        pairs_cards = (valid & (pos < 2 * a.pairs)).sum(1).float().mean().item()
        print(f"DIAG {name:10s} seed {a.seed} сміття {filler:4d}: точність {ok.float().mean().item():.0%} | "
              f"картка факту є в пам'яті {has_q.mean().item():.0%} | "
              f"точність, коли є {ok[has_q.bool()].float().mean().item() if has_q.any() else 0:.0%}, "
              f"коли нема {ok[~has_q.bool()].float().mean().item() if (~has_q.bool()).any() else 0:.0%} | "
              f"карток з ділянки фактів {pairs_cards:.1f} з {valid.sum(1).float().mean().item():.0f}", flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--models", nargs="+", default=["Fv2s", "Fv2s_big"])
    p.add_argument("--steps", type=int, default=10000)
    p.add_argument("--pairs", type=int, default=16)
    p.add_argument("--filler", type=int, default=256)
    p.add_argument("--batch", type=int, default=128)
    p.add_argument("--accum", type=int, default=1)        # батч частинами (ПК): той самий крок, що й батч цілком
    p.add_argument("--seed", type=int, default=0)        # ініціалізація: перевірка стабільності навчання    # з 32 «стрибок» навчання не настає навіть у трансформера
    a = p.parse_args()
    for name in a.models:
        torch.manual_seed(a.seed)
        m = (TransformerLM(VOCAB, d=256, layers=3, heads=4) if name == "Tsmall" else build(name, VOCAB)).cuda()
        opt = torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=0.0)
        g = torch.Generator(device="cuda").manual_seed(1)
        t0 = time.time()
        # поступове ускладнення «за успіхом»: рівень росте, лише коли поточний розв'язано на 90%+
        levels = [(p, f) for f in (16, 32, 64, 128, 256) for p in (4, 8, a.pairs)]
        lvl, hist = 0, []
        for step in range(1, a.steps + 1):
            pairs, filler = levels[lvl]
            opt.zero_grad(set_to_none=True)
            acc = 0.0
            for _ in range(a.accum):
                x, y, ans = batch(a.batch // a.accum, pairs, filler, pairs, g)
                with torch.autocast("cuda", torch.bfloat16):
                    loss, logits = loss_and_pred(m, x, y, ans)
                (loss / a.accum).backward()
                acc += (logits.argmax(-1) == y[:, ans]).float().mean().item() / a.accum
            torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
            opt.step()
            hist.append(acc)
            if len(hist) >= 50 and sum(hist[-50:]) / 50 > 0.9 and lvl < len(levels) - 1:
                lvl, hist = lvl + 1, []
            if step % 500 == 0:
                print(f"{name} крок {step} пар {pairs} сміття {filler} loss {loss.item():.3f} "
                      f"точність {sum(hist[-50:]) / max(1, len(hist[-50:])):.0%} {time.time() - t0:.0f}с", flush=True)
        m.eval()
        gv = torch.Generator(device="cuda").manual_seed(7)
        res = []
        for filler in (32, 128, 256, 512, 1024):
            ok = tot = 0
            with torch.no_grad(), torch.autocast("cuda", torch.bfloat16):
                for _ in range(4):
                    x, y, ans = batch(32, a.pairs, filler, a.pairs, gv)
                    _, logits = loss_and_pred(m, x, y, ans)
                    pred = logits.argmax(-1)
                    ok += (pred == y[:, ans]).sum().item()
                    tot += pred.numel()
            res.append(f"сміття {filler:4d}: {ok / tot:.0%}")
        print(f"RECALL {name:10s} seed {a.seed} " + " | ".join(res), flush=True)
        if hasattr(m, "last_mem"):
            diagnose(m, a, name)


if __name__ == "__main__":
    main()
