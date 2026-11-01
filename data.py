"""Associative recall: "Петро 3 Іван 7 ... Петро ?" -> 3.

Послідовність: N пар (ключ, значення), потім M запитів (ключ, значення).
Ключі в межах послідовності унікальні. Loss рахується лише на значеннях у запитах.
"""
import torch

N_KEYS = 512      # "імена"
N_VALUES = 64     # "номери будинків"
VOCAB = N_KEYS + N_VALUES
IGNORE = -100


def make_batch(batch, n_pairs, n_queries, device="cpu", generator=None):
    keys = torch.rand(batch, N_KEYS, generator=generator).argsort(-1)[:, :n_pairs]
    vals = torch.randint(0, N_VALUES, (batch, n_pairs), generator=generator) + N_KEYS

    pairs = torch.stack([keys, vals], -1).reshape(batch, 2 * n_pairs)

    q_idx = torch.randint(0, n_pairs, (batch, n_queries), generator=generator)
    q_keys = keys.gather(1, q_idx)
    q_vals = vals.gather(1, q_idx)
    queries = torch.stack([q_keys, q_vals], -1).reshape(batch, 2 * n_queries)

    seq = torch.cat([pairs, queries], 1)
    x = seq[:, :-1]
    y = torch.full_like(seq[:, 1:], IGNORE)
    # позиції, де вхід = ключ запиту, а ціль = його значення
    ans_pos = 2 * n_pairs + 2 * torch.arange(n_queries)
    y[:, ans_pos] = seq[:, ans_pos + 1]
    return x.to(device), y.to(device)
