"""INT8_MM=1: множення матриць у int8 (прямий прохід і градієнт за входом), градієнт ваги — у bf16.

Навіщо: матричний блок v6e рахує int8 удвічі швидше за bf16. Тут — та сама математика у «фальшивому» вигляді для
перевірки якості на будь-якому пристрої: тензори квантуються симетрично (масштаб = max|·| / 127 на рядок входу й на
вихідний канал ваги), цілі значення (|q| ≤ 127 — точно представні в bf16) множаться звичайним bf16-множенням з
fp32-накопиченням, результат множиться на масштаби. Це числово те саме, що int8-множення з int32-накопиченням
(різниця — лише одне округлення суми до bf16 перед масштабом).

Схема (як SwitchBack / Jetfire):
    y  = Q_row(x) · Q_row(W)ᵀ                  — уперед, int8
    dx = Q_row(g) · Q_col(W)                   — градієнт за входом, int8 (масштаб W — по осі входу)
    dW = gᵀ · x                                — градієнт ваги, bf16 (найчутливіший — сума по всіх словах)
"""
import os

import torch
import torch.nn.functional as F

INT8 = os.environ.get("INT8_MM") == "1"
MIN_DIM = int(os.environ.get("INT8_MIN_DIM", 256))     # дрібні множення (маршрутизатор, гейти) лишаються bf16
# які множення в int8 (коли INT8_MM=1): голова на словник, щільні шари (nn.Linear / F.linear), експерти (bmm)
HEAD = os.environ.get("INT8_HEAD", "1") == "1"
LIN = os.environ.get("INT8_LIN", "1") == "1"
EXP = os.environ.get("INT8_EXP", "1") == "1"


def fq(x, dim):
    """Симетрична квантизація по осі dim → (цілі значення в bf16, масштаб fp32 з тією ж формою, що й max)."""
    xf = x.float()
    s = xf.abs().amax(dim=dim, keepdim=True).clamp(min=1e-12) / 127.0
    q = torch.round(xf / s).clamp(-127, 127)
    return q.to(torch.bfloat16), s


class _I8Linear(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, w):
        K = x.shape[-1]
        x2 = x.reshape(-1, K)
        if on_real(x):
            y = _jops()["lin_fwd"](x2.to(torch.bfloat16), w.to(torch.bfloat16))
        else:
            qx, sx = fq(x2, 1)                   # (M,K), (M,1)
            qw, sw = fq(w, 1)                    # (N,K), (N,1)
            y = (qx @ qw.t()).float() * sx * sw.t()
        ctx.save_for_backward(x2, w)
        ctx.shape, ctx.dt = x.shape, x.dtype
        return y.to(x.dtype).reshape(*x.shape[:-1], w.shape[0])

    @staticmethod
    def backward(ctx, g):
        x2, w = ctx.saved_tensors
        g2 = g.reshape(-1, w.shape[0])
        if on_real(g):
            dx = _jops()["lin_dgrad"](g2.to(torch.bfloat16), w.to(torch.bfloat16)).to(ctx.dt).reshape(ctx.shape)
        else:
            qg, sg = fq(g2, 1)                   # (M,N), (M,1)
            qw, sw = fq(w, 0)                    # (N,K), (1,K) — масштаб по осі входу
            dx = ((qg @ qw).float() * sg * sw).to(ctx.dt).reshape(ctx.shape)
        dw = (g2.to(torch.bfloat16).t() @ x2.to(torch.bfloat16)).to(w.dtype)
        return dx, dw


class _I8Bmm(torch.autograd.Function):
    """(E,C,K) @ (E,K,N) — експерти: масштаб на рядок входу й на вихідний стовпець ваги кожного експерта."""

    @staticmethod
    def forward(ctx, x, w):
        if on_real(x):
            y = _jops()["bmm_fwd"](x.to(torch.bfloat16), w.to(torch.bfloat16))
        else:
            qx, sx = fq(x, 2)                    # (E,C,K), (E,C,1)
            qw, sw = fq(w, 1)                    # (E,K,N), (E,1,N)
            y = torch.bmm(qx, qw).float() * sx * sw
        ctx.save_for_backward(x, w)
        return y.to(x.dtype)

    @staticmethod
    def backward(ctx, g):
        x, w = ctx.saved_tensors
        if on_real(g):
            dx = _jops()["bmm_dgrad"](g.to(torch.bfloat16), w.to(torch.bfloat16)).to(x.dtype)
        else:
            qg, sg = fq(g, 2)                    # (E,C,N), (E,C,1)
            qw, sw = fq(w, 2)                    # (E,K,N), (E,K,1) — масштаб по осі входу K
            dx = (torch.bmm(qg, qw.transpose(1, 2)).float() * sg * sw.transpose(1, 2)).to(x.dtype)
        dw = torch.bmm(x.to(torch.bfloat16).transpose(1, 2), g.to(torch.bfloat16)).to(w.dtype)
        return dx, dw


def bmm(x, w):
    if INT8 and EXP and min(x.shape[-1], w.shape[-1]) >= MIN_DIM and x.is_floating_point():
        return _I8Bmm.apply(x, w)
    return torch.bmm(x, w)


_orig_linear = F.linear


def _linear(x, w, b=None):
    if INT8 and LIN and w.dim() == 2 and min(w.shape) >= MIN_DIM and x.is_floating_point() and x.shape[-1] == w.shape[1]:
        y = _I8Linear.apply(x, w)                # типи вирівнює сама квантизація (усе — у цілі bf16)
        return y + b.to(y.dtype) if b is not None else y
    return _orig_linear(x, w, b)


if INT8:
    F.linear = _linear
    torch.nn.functional.linear = _linear
    if os.environ.get("I8_WQ") == "1":          # int8-копії ваг з ядра Instinct (int8mm8.py) — підміняє _I8Linear/_I8Bmm
        import int8mm8  # noqa: F401


# ------------------------------------------------------------------------------------------------- справжній int8 на TPU
# torch_tpu не має aten::_int_mm (перевірено пробою 56), а JAX-множення int8×int8→int32 на v6e — 1302 проти 790 TFLOP/с
# bf16. Тому на TPU ті самі квантизація й множення — однією JAX-функцією через міст torch.tpu.pallas.jax_op
# (XLA зливає квантизацію з множенням). INT8_REAL=0 — фальшивий шлях і на TPU.
REAL = os.environ.get("INT8_REAL", "1") == "1"
FUSED = os.environ.get("I8_FUSED", "1") == "1"
_JOPS = {}


def _jops():
    if _JOPS:
        return _JOPS
    import jax
    import jax.numpy as jnp
    import torch_tpu  # noqa: F401  (реєструє torch.tpu — модуль може імпортуватися раніше за решту)
    from torch.tpu import pallas as tpp

    def q(x, axis):
        xf = x.astype(jnp.float32)
        s = jnp.maximum(jnp.max(jnp.abs(xf), axis=axis, keepdims=True), 1e-12) / 127.0
        return jnp.clip(jnp.round(xf / s), -127, 127).astype(jnp.int8), s

    def mm_rows(a, b, out_f32):
        """a (M,K) · b (N,K)ᵀ, масштаб на рядок a і на рядок b."""
        qa, sa = q(a, 1)
        qb, sb = q(b, 1)
        y = jax.lax.dot_general(qa, qb, (((1,), (1,)), ((), ())), preferred_element_type=jnp.int32)
        y = y.astype(jnp.float32) * sa * sb.T
        return y if out_f32 else y.astype(a.dtype)

    def lin_fwd(x: jax.Array, w: jax.Array) -> jax.Array:
        return mm_rows(x, w, False)

    def lin_fwd_f32(x: jax.Array, w: jax.Array) -> jax.Array:
        return mm_rows(x, w, True)

    def lin_dgrad(g: jax.Array, w: jax.Array) -> jax.Array:
        """g (M,N) · w (N,K): масштаб на рядок g і на стовпець w (вісь входу K)."""
        qg, sg = q(g, 1)
        qw, sw = q(w, 0)
        y = jax.lax.dot_general(qg, qw, (((1,), (0,)), ((), ())), preferred_element_type=jnp.int32)
        return (y.astype(jnp.float32) * sg * sw).astype(g.dtype)

    def lin_dgrad_f32(g: jax.Array, w: jax.Array) -> jax.Array:
        qg, sg = q(g, 1)
        qw, sw = q(w, 0)
        y = jax.lax.dot_general(qg, qw, (((1,), (0,)), ((), ())), preferred_element_type=jnp.int32)
        return y.astype(jnp.float32) * sg * sw

    def bmm_fwd(x: jax.Array, w: jax.Array) -> jax.Array:
        """(E,C,K)·(E,K,N): масштаб на рядок x і на вихідний стовпець w."""
        qx, sx = q(x, 2)
        qw, sw = q(w, 1)
        y = jax.lax.dot_general(qx, qw, (((2,), (1,)), ((0,), (0,))), preferred_element_type=jnp.int32)
        return (y.astype(jnp.float32) * sx * sw).astype(x.dtype)

    def bmm_dgrad(g: jax.Array, w: jax.Array) -> jax.Array:
        """(E,C,N)·(E,K,N)ᵀ: масштаб на рядок g і на вхідний рядок w (вісь K)."""
        qg, sg = q(g, 2)
        qw, sw = q(w, 2)
        y = jax.lax.dot_general(qg, qw, (((2,), (2,)), ((0,), (0,))), preferred_element_type=jnp.int32)
        return (y.astype(jnp.float32) * sg * jnp.swapaxes(sw, 1, 2)).astype(g.dtype)

    if FUSED:                                    # I8_FUSED=1: квантизація активації — всередині Pallas-ядра (pallas_i8.py)
        impl = os.environ.get("I8_IMPL", "a")       # a: pallas_i8; b: pallas_i8b (великі плитки, тайлинг K);
        if impl == "c":                              # c: pallas_i8c (вага теж квантується в ядрі, без проходів у XLA)
            import pallas_i8c as P
        elif impl == "b":
            import pallas_i8b as P
        else:
            import pallas_i8 as P

        def lin_fwd(x: jax.Array, w: jax.Array) -> jax.Array:
            return P.lin_fwd(x, w, x.dtype)

        def lin_fwd_f32(x: jax.Array, w: jax.Array) -> jax.Array:
            return P.lin_fwd(x, w, jnp.float32)

        def lin_dgrad(g: jax.Array, w: jax.Array) -> jax.Array:
            return P.lin_dgrad(g, w, g.dtype)

        def lin_dgrad_f32(g: jax.Array, w: jax.Array) -> jax.Array:
            return P.lin_dgrad(g, w, jnp.float32)

        def bmm_fwd(x: jax.Array, w: jax.Array) -> jax.Array:
            return P.bmm_fwd(x, w)

        def bmm_dgrad(g: jax.Array, w: jax.Array) -> jax.Array:
            return P.bmm_dgrad(g, w)

    for fn in (lin_fwd, lin_fwd_f32, lin_dgrad, lin_dgrad_f32, bmm_fwd, bmm_dgrad):
        _JOPS[fn.__name__] = tpp.jax_op(f"memarch::i8{'f' if FUSED else ''}_{fn.__name__}", fn)
    return _JOPS


def on_real(t):
    return INT8 and REAL and t.device.type == "tpu"


# операції моста треба створити ДО torch.compile: jax_op, викликаний уперше під трасуванням dynamo, падає на побудові
# схеми (KeyError: 'x', проба 57) — тож на TPU створюємо всі одразу під час імпорту
if INT8 and REAL and os.environ.get("DEVICE") == "tpu":
    _jops()
