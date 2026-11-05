"""Пристрій тренування: DEVICE=cuda (за замовчуванням) | cpu | tpu (torch_tpu — рідний бекенд, як cuda) | xla (torch_xla).

Імпортувати ДО lm / full2: на XLA тут же вмикаються NO_COMPILE (граф кроку збирає сама XLA, torch.compile
не потрібен) і ATTN_IMPL=block (вікно уваги без FlexAttention — сталі форми)."""
import os

import torch

DEVICE = os.environ.get("DEVICE", "cuda")
XLA = DEVICE == "xla"
CUDA = DEVICE == "cuda"
TPU = DEVICE == "tpu"
TYPE = DEVICE                                   # тип пристрою для torch.autocast
if XLA:
    os.environ.setdefault("NO_COMPILE", "1")
    os.environ.setdefault("ATTN_IMPL", "block")
    import torch_xla
    import torch_xla.core.xla_model as xm
if TPU:                                         # torch.compile працює; FlexAttention і Triton — ні
    os.environ.setdefault("ATTN_IMPL", "block")
    import torch_tpu  # noqa: F401  (реєструє пристрій 'tpu' і модуль torch.tpu)

_dev = None


def device():
    global _dev
    if _dev is None:
        _dev = torch_xla.device() if XLA else torch.device(DEVICE)
    return _dev


def sync():
    """Дочекатися відеокарти / TPU (лише для замірів часу й оцінки — не щокроку)."""
    if XLA:
        xm.wait_device_ops()
    elif CUDA:
        torch.cuda.synchronize()
    elif TPU:                                   # внутрішні torch.tpu.* закриті — лише загальний torch.accelerator
        torch.accelerator.synchronize()


def step_end():
    """Кінець кроку: на XLA — запустити зібраний граф кроку (без чекання на результат)."""
    if XLA:
        xm.mark_step()


def generator(seed):
    """Генератор на пристрої; XLA власних генераторів не має — тоді None і спільний посів пристрою."""
    if XLA:
        xm.set_rng_state(seed, device())
        return None
    try:
        return torch.Generator(device=device()).manual_seed(seed)
    except Exception:                           # бекенд без власних генераторів — спільний посів пристрою
        torch.manual_seed(seed)
        return None


def empty_cache():
    if CUDA:
        torch.cuda.empty_cache()


def reset_peak():
    if CUDA:
        torch.cuda.reset_peak_memory_stats()
    elif TPU:
        try:
            torch.accelerator.reset_peak_memory_stats()
        except Exception:
            pass


def max_mem_gb():
    if CUDA:
        return torch.cuda.max_memory_allocated() / 2**30
    if XLA:
        info = xm.get_memory_info(device())
        return info.get("peak_bytes_used", info.get("bytes_used", 0)) / 2**30
    if TPU:
        try:
            return torch.accelerator.max_memory_allocated() / 2**30
        except Exception:
            return 0.0
    return 0.0


# RMS_IMPL=mm: RMSNorm зі своїм зворотним проходом — градієнт ваги (сума по всіх словах) як множення ones @ (g·x̂)
# на MXU, а не згортка по 8192 рядках у векторному блоці (на v6e злиття «назад» RMSNorm — ~3 мс на шар);
# обчислення у fp32 лише для 1/rms, решта в типі входу. Однаково для всіх nn.RMSNorm (стан моделі той самий).
import os as _os
if _os.environ.get("RMS_IMPL", "mm" if TYPE in ("tpu", "xla") else "") == "mm":
    import torch as _t

    class _RMS(_t.autograd.Function):
        @staticmethod
        def forward(ctx, x, w, eps):
            r = _t.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + eps)          # (..., 1) fp32
            xh = (x.float() * r).to(x.dtype)
            ctx.save_for_backward(x, w, r)
            return xh * w.to(x.dtype) if w is not None else xh

        @staticmethod
        def backward(ctx, g):
            x, w, r = ctx.saved_tensors
            d = x.shape[-1]
            xh = (x.float() * r)
            gw = g.float() * w.float() if w is not None else g.float()
            # d x = r · (gw − x̂ · mean(gw · x̂))
            dx = (r * (gw - xh * (gw * xh).mean(-1, keepdim=True))).to(x.dtype)
            dw = None
            if w is not None:
                gx = (g * xh.to(g.dtype)).reshape(-1, d)
                dw = (_t.ones(1, gx.shape[0], device=gx.device, dtype=gx.dtype) @ gx).float().reshape(d).to(w.dtype)
            return dx, dw, None

    def _rms_forward(self, x):
        eps = self.eps if self.eps is not None else _t.finfo(x.dtype).eps
        return _RMS.apply(x, self.weight, eps)

    _t.nn.RMSNorm.forward = _rms_forward
