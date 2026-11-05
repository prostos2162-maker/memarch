"""Спільне для test_memory / test_reread: torch-еталон (mem2.Memory, full2.Reread, mempass-цикл, _reread_sep)
на CPU, перенесення ваг у JAX, порівняння. Розриви рівних у torch topk/argsort замінено стабільними
(«менший індекс першим») — канонічний вибір порту (docs/decisions.md)."""
import os
import sys
import types
from contextlib import contextmanager

os.environ.setdefault("DEVICE", "cpu")
os.environ.setdefault("NO_COMPILE", "1")
os.environ.setdefault("ATTN_IMPL", "block")
os.environ.setdefault("REREAD_SEP", "1")
os.environ.setdefault("REREAD_ANCH", "0.25")
# точне округлення bf16 після кожної операції, як у eager torch (XLA інакше може пропускати проміжні округлення)
os.environ["XLA_FLAGS"] = (os.environ.get("XLA_FLAGS", "") + " --xla_allow_excess_precision=false").strip()
for k in ("MEM_BF16", "BF16_ALL", "RR_BISECT", "NO_MEM", "NO_REREAD"):
    os.environ.pop(k, None)
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")))  # repo root: torch reference code
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import jax
import jax.numpy as jnp

jax.config.update("jax_default_matmul_precision", "highest")

import mem2      # noqa: E402
import full2     # noqa: E402

_orig_topk, _orig_argsort = torch.Tensor.topk, torch.Tensor.argsort


def _stable_topk(self, k, dim=-1, largest=True, sorted=True):
    r = torch.sort(self, dim=dim, descending=largest, stable=True)
    return _TopK(r.values.narrow(dim, 0, k), r.indices.narrow(dim, 0, k))


class _TopK(tuple):
    def __new__(cls, v, i):
        return super().__new__(cls, (v, i))

    values = property(lambda s: s[0])
    indices = property(lambda s: s[1])


def _stable_argsort(self, dim=-1, descending=False, stable=False):
    return torch.sort(self, dim=dim, descending=descending, stable=True).indices


@contextmanager
def stable_torch():
    torch.Tensor.topk, torch.Tensor.argsort = _stable_topk, _stable_argsort
    try:
        yield
    finally:
        torch.Tensor.topk, torch.Tensor.argsort = _orig_topk, _orig_argsort


def t2j(sd):
    return {k: jnp.asarray(v.detach().float().numpy()) for k, v in sd.items()}


def j2n(x):
    return np.asarray(jnp.asarray(x, jnp.float32)) if jnp.issubdtype(jnp.asarray(x).dtype, jnp.floating) else np.asarray(x)


def tn(x):
    return x.detach().float().numpy() if x.is_floating_point() else x.detach().numpy()


def diff(a, b):
    """(max abs, max rel = max|a-b| / max|b|)."""
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    m = np.abs(a - b).max() if a.size else 0.0
    return float(m), float(m / max(np.abs(b).max(), 1e-30))


# ---- torch-еталон ----
def torch_memory_pass(mem, h, seg, bf16):
    """Точна копія hyperop/mempass._memory_pass_ingraph (use_memory=True)."""
    B, T, _ = h.shape
    st = mem.init_state(B, h.device, torch.float32)
    reads, snaps = [], []
    offs = torch.arange(0, T, seg, device=h.device, dtype=torch.float32)
    for j, s in enumerate(range(0, T, seg)):
        hs = h[:, s:s + seg].float()
        with torch.autocast("cpu", dtype=torch.bfloat16, enabled=bf16):
            r, w = mem.read(hs, st)
            snap = (st["faces"], mem._contents(st), st["pos"], st["valid"])
            qpos = offs[j] + torch.arange(hs.shape[1], device=hs.device).expand(hs.shape[0], -1)
            st = mem.write(hs + r, qpos, st, now=offs[j] + hs.shape[1], w_read=w)
        reads.append(r)
        snaps.append(snap)
    return torch.cat(reads, 1), snaps, st


def torch_reread_sep(rr, h, ids, table, snaps, seg, s_, bf16):
    """FullLM2._reread_sep на фальшивому self (лише reread + use_memory)."""
    full2.MEM_BF16 = bf16
    fake = types.SimpleNamespace(reread=rr, use_memory=True)
    fake._reread_core = lambda *a: full2.FullLM2._reread_core(fake, *a)
    return full2.FullLM2._reread_sep(fake, h, (ids, table), snaps, seg, s_)


def structured_h(B, T, d, seed, seg=64, width=3, noise=0.1, amp=4.0):
    """Стани з повторами «тем», що повільно дрейфують (шматок j бачить теми j//2 .. j//2+width-1): кожне слово ~
    одна базова + шум. Так працюють усі гілки: нові/опускання, оновлення з дублікатами, підйом, прибирання."""
    g = torch.Generator().manual_seed(seed)
    n = (T + seg - 1) // seg
    basis = torch.randn(n // 2 + width + 1, d, generator=g)
    base = (torch.arange(T) // seg) // 2
    idx = base[None] + torch.randint(0, width, (B, T), generator=g)
    sc = 0.5 + torch.rand(B, T, 1, generator=g)
    return (amp * (basis[idx] * sc + noise * torch.randn(B, T, d, generator=g))).contiguous()
