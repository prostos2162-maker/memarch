"""Важкий шар MoE (BankMoE ← BudgetMoE, MOE_DISP=onehot) і стек 14 важких шарів — порт з full2.py.

Усе — чисті функції над pytree параметрів (dict), форми статичні (XLA). Форми локальні для одного чипа:
роутинг, місткість, групи, черги, відкидання й aux рахуються на власних N словах чипа (design §0, spec_1 A6);
між чипами ходить лише робота експертів — через змінний «обмін» (exchange): тотожність при W=1,
lax.all_to_all усередині shard_map при W>1.

Параметри одного шару (імена як у torch state_dict):
    {'norm': (d,), 'router': (E, d), 'up': (El, d, h2), 'down': (El, h2, d)}
Стек важких шарів — ті самі ключі з провідною віссю шару (L, ...) для lax.scan.

Чого немає (свідомо, див. docs/decisions.md): ErrProbe (буфер err ніхто не читає), rebank/wave/hint
(мертві при WAVE_WARM=1e6, hint_w=0), budget-гілка (BankMoE: budget=False).
"""
import os
from dataclasses import dataclass, field
from functools import partial

import jax
import jax.numpy as jnp
from jax import lax
from jax.ad_checkpoint import checkpoint_name

EPS_F32 = 1.1920929e-07          # nn.RMSNorm(eps=None) → машинний eps fp32 (spec_1 A1)


# ---------- локальні примітиви (мінімальні копії; спільні — у prims.py іншого агента) ----------

def rmsnorm(x, w, eps=EPS_F32):
    """RMSNorm як у torch (nn.RMSNorm, opmath): нормування І множення на вагу у fp32, один раз назад у тип x
    (так збігається з torch CPU bf16 побітово, 1 розбіжність на 65k; див. docs/decisions.md)."""
    xf = x.astype(jnp.float32)
    y = xf * lax.rsqrt(jnp.mean(xf * xf, -1, keepdims=True) + eps) * w.astype(jnp.float32)
    return y.astype(x.dtype)


def _cpu_up(*ts):
    """CPU-XLA не вміє bf16×bf16→f32: там піднімаємо входи до f32 (добутки bf16 точні в f32 — та сама математика)."""
    if jax.default_backend() == "cpu":
        return tuple(t.astype(jnp.float32) if t.dtype == jnp.bfloat16 else t for t in ts)
    return ts


def einsum32(spec, a, b):
    """einsum з накопиченням fp32 (вихід fp32)."""
    a, b = _cpu_up(a, b)
    return jnp.einsum(spec, a, b, preferred_element_type=jnp.float32)


def linear(x, W):
    """x @ W^T (W у формі torch (out, in)): накопичення fp32, вихід у типі x (емуляція autocast)."""
    xa, Wa = _cpu_up(x, W.astype(x.dtype))
    y = lax.dot_general(xa, Wa, (((x.ndim - 1,), (1,)), ((), ())),
                        preferred_element_type=jnp.float32)
    return y.astype(x.dtype)


def bmm(a, b):
    """Пакетне множення (n,i,j)@(n,j,k): накопичення fp32, вихід у типі a."""
    aa, ba = _cpu_up(a, b.astype(a.dtype))
    y = lax.dot_general(aa, ba, (((2,), (1,)), ((0,), (0,))),
                        preferred_element_type=jnp.float32)
    return y.astype(a.dtype)


# ---------- конфігурація шару ----------

@dataclass(frozen=True)
class MoECfg:
    E: int = 40                 # експертів (глобально)
    k: int = 4                  # MOE_K: top-k = k_avg = k_max
    group: int = 512            # MOE_GROUP
    cap_align: int = 256        # MOE_CAP_ALIGN (TPU 256; CPU-torch за замовчуванням 0)
    fill: float = 0.2 / 0.25    # frac / cap, як FullLM2 ставить b.fill
    cf: float = 1.25            # запас місткості (budget=False)
    # --- швидкісні прапори (математика та сама; старий шлях — fast=False для A/B, див. docs/decisions.md) ---
    fast: bool = True           # one-hot einsum одразу в bf16, без транспозицій довкола a2a, конвеєр шматками
    a2a_chunks: int = 1         # moe_a2a_chunks: на скільки шматків по групах G різати dispatch→a2a→експерти→a2a→combine.
                                # 1 = побітово як старий шлях; ≥2 — dW експертів = сума K часткових (≤1 bf16-ulp), див. decisions
    # --- прапори профілю prof_b4_msh (усі — та сама математика; вмикаються env, бо config/bench — чужі файли) ---
    # unroll: шари важкого стеку python-циклом замість lax.scan → збережені залишки (a2a_in/a2a_out) — окремі
    # буфери, без DUS-стеку у fwd, DS + copy розкладки у bwd (moe.py:286, ≈20 ms). Ціна: довша компіляція.
    unroll: bool = field(default_factory=lambda: os.environ.get("MOE_UNROLL", "0") == "1")
    # up_mid: розкладка проміжку експерта: 'w' — (W,El,c,h) як зараз; 'e' — (El,W,c,h) (експерт-мажор, як старий
    # bmm, де up-матмул ішов на 84%); down повертає (W,El,c,d) для a2a. Ті самі добутки/згортки.
    up_mid: str = field(default_factory=lambda: os.environ.get("MOE_UP_MID", "w"))


def capacity(N, cfg):
    """Місткість експерта C і групування (g, G, Np, Cg, Ct) — python, точно як torch (banker's round)."""
    C = int(cfg.cf * cfg.fill * N * cfg.k / cfg.E) + 1
    if cfg.cap_align:
        C = max(cfg.cap_align, round(C / cfg.cap_align) * cfg.cap_align)
    g = min(cfg.group, N) if cfg.group > 0 else N
    G = -(-N // g)
    Np = G * g
    Cg = -(-C // G)
    return dict(C=C, g=g, G=G, Np=Np, Cg=Cg, Ct=G * Cg)


# ---------- обмін між чипами (EP) ----------

@dataclass(frozen=True)
class Exchange:
    """Як місця (E, Ct, d) їдуть до власників експертів і назад.
    send: (E, Ct, d) → (El, W·Ct, d) ; recv: (El, W·Ct, d) → (E, Ct, d)  — старий шлях (з транспозиціями).
    swap: (W, El, c, d) → (W, El, c, d) — сирий обмін блоками (блок w іде до чипа w; отриманий блок s — від чипа s).
    Він симетричний: той самий «туди» і «назад». Новий шлях (MoECfg.fast) бере лише swap і W."""
    send: callable
    recv: callable
    swap: callable = None
    W: int = 1


IDENTITY = Exchange(send=lambda xe: xe, recv=lambda y: y, swap=lambda x: x, W=1)   # W=1: усі експерти свої


def _swap_ppermute(x, axis_name, W):
    """Той самий обмін, що all_to_all(tiled=False), але кільцем з W-1 collective-permute.
    На TPU collective-permute асинхронний — планувальник XLA може сховати його під матмули. Лише переміщення байтів."""
    r = lax.axis_index(axis_name)
    out = x                                                                   # свій блок лишається: out[r] = x[r]
    for s_ in range(1, W):
        blk = lax.dynamic_index_in_dim(x, (r + s_) % W, 0, keepdims=True)      # шлю чипу r+s
        got = lax.ppermute(blk, axis_name, [(i, (i + s_) % W) for i in range(W)])   # отримую від r-s
        out = lax.dynamic_update_index_in_dim(out, got, (r - s_) % W, 0)
    return out


def a2a_exchange(axis_name, W, impl="a2a"):
    """EP усередині shard_map: чип r тримає експертів [r·El, (r+1)·El) (як all_to_all_single у torch:
    dim0 ділиться на W суцільних шматків по El експертів; отриманий блок s — від чипа s).
    impl ('a2a' | 'ppermute') стосується лише swap (нового шляху)."""
    def send(xe):
        E, Ct, d = xe.shape
        El = E // W
        r = lax.all_to_all(xe.reshape(W, El, Ct, d), axis_name, 0, 0, tiled=False)   # (W src, El, Ct, d)
        return r.transpose(1, 0, 2, 3).reshape(El, W * Ct, d)

    def recv(y):
        El, WC, d = y.shape
        Ct = WC // W
        y = y.reshape(El, W, Ct, d).transpose(1, 0, 2, 3)                              # (W dst, El, Ct, d)
        return lax.all_to_all(y, axis_name, 0, 0, tiled=False).reshape(W * El, Ct, d)

    if impl == "ppermute":
        swap = partial(_swap_ppermute, axis_name=axis_name, W=W)
    else:
        def swap(x):
            return lax.all_to_all(x, axis_name, 0, 0, tiled=False)
    return Exchange(send=send, recv=recv, swap=swap, W=W)


# ---------- шар ----------

def ffn_relu2(r, up, down):
    """Експерт relu(r·up)²·down: обидва множення fp32-накопичення з виходом у типі r; relu і квадрат — у типі r."""
    a = bmm(r, up)
    return bmm(jnp.square(jax.nn.relu(a)), down)


def _dot_we(a, b):
    """(W,El,c,i)@(El,i,j) → (W,El,c,j): пакет по експерту без транспозиції; fp32-накопичення, вихід у типі a.
    Ті самі добутки й та сама згортка по i, що bmm над (El, W·c, i) — лише інша розкладка."""
    aa, ba = _cpu_up(a, b.astype(a.dtype))
    y = jnp.einsum('wecd,edh->wech', aa, ba, preferred_element_type=jnp.float32)
    return y.astype(a.dtype)


def _ein(spec, a, b):
    """einsum з fp32-накопиченням і виходом у типі a (як _dot_we, але довільна розкладка)."""
    aa, ba = _cpu_up(a, b.astype(a.dtype))
    return jnp.einsum(spec, aa, ba, preferred_element_type=jnp.float32).astype(a.dtype)


def ffn_relu2_we(r, up, down, mid="w"):
    """ffn_relu2 над розкладкою (W src, El, c, d) — без transpose(1,0,2,3) до/після a2a.
    mid='e': проміжок (El,W,c,h) — up-матмул пише експерт-мажор (розкладка старого bmm), down читає її
    й пише назад (W,El,c,d). Ті самі добутки й згортки по d/h → побітово те саме."""
    if mid == "e":
        a = _ein('wecd,edh->ewch', r, up)
        return _ein('ewch,ehd->wecd', jnp.square(jax.nn.relu(a)), down)
    a = _dot_we(r, up)
    return _dot_we(jnp.square(jax.nn.relu(a)), down)


def _onehot_mm(spec, a, b, out_dtype):
    """One-hot einsum з виходом одразу в out_dtype (без f32-проміжку в HBM).
    TPU: fp32-накопичення в MXU і одне округлення → ті самі біти, що einsum32(...).astype(out_dtype).
    CPU: явний fp32 + astype (CPU-XLA не вміє bf16×bf16→f32) — буквально старий шлях."""
    if jax.default_backend() == "cpu" or out_dtype == jnp.float32:
        return einsum32(spec, a, b).astype(out_dtype)
    return jnp.einsum(spec, a, b, preferred_element_type=out_dtype)


def route(p, x, sa, cfg):
    """Роутер (_route): xf (N,d), keep (N,k), w (N,k) fp32, top_e (N,k), oh (N,k,E) int32, probs, aux."""
    B, A, d = x.shape
    N, E, k = B * A, cfg.E, cfg.k
    xf = rmsnorm(x, p['norm']).reshape(N, d)
    probs = jax.nn.softmax(linear(xf, p['router']).astype(jnp.float32), -1)
    real = sa.reshape(N) >= 0 if sa is not None else jnp.ones((N,), bool)
    top_p, top_e = lax.top_k(probs, k)
    keep = jnp.broadcast_to(real[:, None], (N, k))               # k_tok = k·real, k_max = k
    tk = top_p * keep
    w = tk / jnp.maximum(tk.sum(-1, keepdims=True), 1e-9)
    e_flat = jnp.where(keep, top_e, E)
    oh = jax.nn.one_hot(e_flat, E + 1, dtype=jnp.int32)[..., :E]
    frac = oh.sum((0, 1)).astype(jnp.float32)
    realf = real.astype(jnp.float32)
    pm = (probs * realf[:, None]).sum(0) / jnp.maximum(realf.sum(), 1.0)
    aux = E * (frac / jnp.maximum(frac.sum(), 1.0) * pm).sum()
    return xf, keep, w, top_e, oh, probs, aux


def dispatch_onehot(oh, keep, w, cfg, dtype):
    """Групова one-hot розсилка (GShard): D (G,g,E,Cg) у dtype, вага місць wtab (E·Ct,), dropped, place."""
    N, k, E = oh.shape
    c = capacity(N, cfg)
    G, g, Np, Cg = c['G'], c['g'], c['Np'], c['Cg']
    ohp = jnp.pad(oh, ((0, Np - N), (0, 0), (0, 0))).reshape(G, g * k, E)
    rank = (jnp.cumsum(ohp, 1) - 1).reshape(G, g, k, E)                       # номер у черзі експерта в групі
    place = ohp.reshape(G, g, k, E).astype(bool) & (rank < Cg)
    dropped = 1.0 - place.sum().astype(jnp.float32) / jnp.maximum(keep.sum(), 1).astype(jnp.float32)
    pe = place.any(2)                                                         # (G,g,E)
    re = (rank * place).sum(2)                                                # (G,g,E)
    D = ((re[..., None] == jnp.arange(Cg, dtype=re.dtype)) & pe[..., None]).astype(dtype)
    D = lax.stop_gradient(D)
    wp = jnp.pad(w, ((0, Np - N), (0, 0))).reshape(G, g, k, 1)
    we = (wp * place.astype(wp.dtype)).sum(2)                                 # (G,g,E) fp32
    wtab = (D * we[..., None].astype(dtype)).sum(1).transpose(1, 0, 2).reshape(E * G * Cg)
    return D, wtab, dropped, place, c


def bank_moe(p, x, sa, cfg, exchange=IDENTITY, with_fill=False):
    """Один важкий шар: x (B,A,d), sa (B,A) fp32 (<0 — порожнє місце) → (x + out, aux, dropped[, fill]).
    fill — частка непорожніх місць місткості (E·Ct) на чипі (для рішення про ragged_all_to_all), лише лог."""
    B, A, d = x.shape
    E = cfg.E
    xf, keep, w, top_e, oh, probs, aux = route(p, x, sa, cfg)
    N = xf.shape[0]
    D, wtab, dropped, place, c = dispatch_onehot(oh, keep, w, cfg, xf.dtype)
    G, g, Np, Cg, Ct = c['G'], c['g'], c['Np'], c['Cg'], c['Ct']
    xg = jnp.pad(xf, ((0, Np - N), (0, 0))).reshape(G, g, d)
    if cfg.fast and exchange.swap is not None:
        out = _moe_fast(p, xg, D, wtab, c, cfg, exchange, xf.dtype)
    else:                                                                     # старий шлях (A/B)
        xe = einsum32('gsec,gsd->egcd', D, xg).astype(xf.dtype)
        xe = checkpoint_name(xe.reshape(E, Ct, d), "moe_disp")                # у кожному виході — один доданок
        r = checkpoint_name(exchange.send(xe), "moe_a2a_in")
        ye = checkpoint_name(exchange.recv(ffn_relu2(r, p['up'], p['down'])), "moe_a2a_out")   # (E,Ct,d)
        yw = ye * wtab.reshape(E, Ct, 1).astype(ye.dtype)
        out = einsum32('gsec,egcd->gsd', D.astype(ye.dtype), yw.reshape(E, G, Cg, d)).astype(ye.dtype)
    out = out.reshape(Np, d)[:N].astype(xf.dtype)
    res = (x + out.reshape(B, A, d), aux, dropped)
    if with_fill:
        res = res + (lax.stop_gradient(place.sum().astype(jnp.float32) / float(E * Ct)),)
    return res


def _moe_fast(p, xg, D, wtab, c, cfg, exchange, dtype):
    """Новий шлях: K незалежних ланцюжків по групах G (dispatch → a2a → експерти → a2a → combine).
    Між ланцюжками нема залежностей даних, тож XLA може пересилати шматок i+1, поки йдуть матмули шматка i.
    Кожне місце — окремий рядок (експерт рахує рядки незалежно), тож розбиття по G не міняє жодного числа.
    Розкладка (E,G,Cg,d) → шматок (E,Gk,Cg,d) = (W,El,Gk·Cg,d): reshape без копій, транспозицій немає."""
    G, Cg = c['G'], c['Cg']
    E, d, W = cfg.E, xg.shape[-1], exchange.W
    El = E // W
    K = max(1, min(cfg.a2a_chunks, G))
    while G % K:                                                              # K має ділити G
        K -= 1
    Gk = G // K
    wt = wtab.reshape(E, G, Cg, 1)
    outs = []
    for i in range(K):
        sl = slice(i * Gk, (i + 1) * Gk)
        Di = D[sl]
        xe = _onehot_mm('gsec,gsd->egcd', Di, xg[sl], dtype)                  # один ненульовий доданок → точно
        xe = checkpoint_name(xe.reshape(W, El, Gk * Cg, d), "moe_disp")
        r = checkpoint_name(exchange.swap(xe), "moe_a2a_in")                  # (W src, El, c, d)
        y = ffn_relu2_we(r, p['up'], p['down'], cfg.up_mid)
        ye = checkpoint_name(exchange.swap(y), "moe_a2a_out")                 # (W власник, El, c, d) = (E, c, d)
        yw = ye.reshape(E, Gk, Cg, d) * wt[:, sl].astype(ye.dtype)
        outs.append(_onehot_mm('gsec,egcd->gsd', Di.astype(ye.dtype), yw, ye.dtype))
    return outs[0] if K == 1 else jnp.concatenate(outs, 0)


# ---------- стек важких шарів (без уваги: HEAVY_ATTN вимкнено) ----------

def heavy_stack(ps, h, idx, valid, s_, cfg, exchange=IDENTITY, remat=False, stats=None):
    """Важка частина FullLM2 (full2.py:1688-1708): якорі → 14 × bank_moe (lax.scan) → scatter_add назад.
    ps — стек параметрів з віссю шару; h (B,T,d); idx (B,A) int; valid (B,A) bool; s_ (B,T) fp32.
    Повертає h, mean(aux) (у loss іде 0.01·це), mean(dropped) (moe_drop, лише лог).
    stats — необов'язковий dict: сюди кладеться 'moe_fill' (середня частка заповнених місць, лише лог)."""
    B, T, d = h.shape
    a_in = jnp.take_along_axis(h, idx[..., None], 1)                          # (B,A,d)
    sa = jnp.clip(jnp.take_along_axis(s_, idx, 1), 0, 50)
    sa = jnp.where(valid, sa, -1.0)

    def body(a, p):
        a, aux, dr, fl = bank_moe(p, a, sa, cfg, exchange, with_fill=True)
        return a, (aux, dr, fl)
    if remat:
        body = jax.checkpoint(body, policy=None if remat is True else remat)   # remat: True або policy
    if cfg.unroll:
        # python-цикл: шар i бере свій зріз ваг через lax.split (транспозиція split — concatenate, тож градієнти
        # стеку ваг збираються одним concat без нулів+DUS); залишки шарів — окремі буфери (без scan-стеку).
        L = jax.tree.leaves(ps)[0].shape[0]
        sp = jax.tree.map(lambda t: [u[0] for u in lax.split(t, (1,) * L, 0)], ps)
        a, ys = a_in, []
        for i in range(L):
            a, y = body(a, jax.tree.map(lambda l: l[i], sp, is_leaf=lambda v: isinstance(v, list)))
            ys.append(y)
        auxs, drs, fls = (jnp.stack(t) for t in zip(*ys))
    else:
        a, (auxs, drs, fls) = lax.scan(body, a_in, ps)
    if stats is not None:
        stats["moe_fill"] = fls.mean()
    delta = (a - a_in) * valid[..., None].astype(a.dtype)
    bi = jnp.arange(B)[:, None]
    h = h.at[bi, idx].add(delta)                                              # idx у рядку різні
    return h, auxs.mean(), drs.mean()
