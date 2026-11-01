"""Власні злиті кернели (Triton) для гіпероперації: вихід на словник + похибка, SwiGLU.

Порівнюються з Liger Kernel (`bench_fused.py`). Ідея та сама — менше проходів по пам'яті відеокарти:
  * linear_ce: логіти рахуються шматками рядків; один кернел одразу дає похибку і замінює логіти шматка
    на їхній градієнт (softmax − one-hot), тож великі логіти не зберігаються й не рахуються вдруге
    (зараз checkpoint рахує їх двічі). Градієнти входу й ваг рахуються в прямому проході.
  * swiglu: silu(g) * u і зворотний прохід — по одному кернелу замість кількох поелементних операцій.
"""
import torch
import triton
import triton.language as tl


@triton.jit
def _ce_grad_kernel(logits_ptr, y_ptr, loss_ptr, stride, V, BLOCK: tl.constexpr):
    # one row: loss = logsumexp - x[y]; logits are overwritten in place with softmax - onehot
    row = tl.program_id(0)
    ptr = logits_ptr + row * stride
    y = tl.load(y_ptr + row)
    m = -float("inf")
    s = 0.0
    for off in range(0, V, BLOCK):
        cols = off + tl.arange(0, BLOCK)
        x = tl.load(ptr + cols, mask=cols < V, other=-float("inf")).to(tl.float32)
        m_new = tl.maximum(m, tl.max(x, 0))
        s = s * tl.exp(m - m_new) + tl.sum(tl.exp(x - m_new), 0)
        m = m_new
    lse = m + tl.log(s)
    xy = tl.load(ptr + y).to(tl.float32)
    tl.store(loss_ptr + row, lse - xy)
    for off in range(0, V, BLOCK):
        cols = off + tl.arange(0, BLOCK)
        mask = cols < V
        x = tl.load(ptr + cols, mask=mask, other=0.0).to(tl.float32)
        g = tl.exp(x - lse) - (cols == y).to(tl.float32)
        tl.store(ptr + cols, g.to(logits_ptr.dtype.element_ty), mask=mask)


class _LinearCE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, h, w, y, chunk):
        """h (N, d), w (V, d), y (N,) → сума похибок. Градієнти рахуються тут же."""
        N, d = h.shape
        V = w.shape[0]
        hd = h.to(torch.bfloat16)
        wd = w.to(torch.bfloat16)
        gh = torch.empty_like(hd)
        gw = torch.zeros(V, d, device=h.device, dtype=torch.float32)
        loss = torch.empty(N, device=h.device, dtype=torch.float32)
        for i in range(0, N, chunk):
            hc = hd[i:i + chunk]
            logits = hc @ wd.T                                        # (n, V) bf16
            _ce_grad_kernel[(hc.shape[0],)](logits, y[i:i + chunk], loss[i:i + chunk], logits.stride(0), V, BLOCK=4096)
            gh[i:i + chunk] = logits @ wd
            gw += (logits.T @ hc).float()                             # накопичення в fp32
        ctx.save_for_backward(gh, gw)
        ctx.dtypes = (h.dtype, w.dtype)
        return loss.sum()

    @staticmethod
    def backward(ctx, go):
        gh, gw = ctx.saved_tensors
        return (gh * go).to(ctx.dtypes[0]), (gw * go).to(ctx.dtypes[1]), None, None


class _LinearCESparse(torch.autograd.Function):
    """Те саме, що _LinearCE, але до логітів додається розріджений зсув: logits[n, cand[n, k]] += b[n, k]
    («слова на язиці» — продовження з розмовника). Повертає градієнт і для b."""

    @staticmethod
    def forward(ctx, h, w, y, cand, b, chunk):
        N, d = h.shape
        V = w.shape[0]
        hd = h.to(torch.bfloat16)
        wd = w.to(torch.bfloat16)
        gh = torch.empty_like(hd)
        gw = torch.zeros(V, d, device=h.device, dtype=torch.float32)
        gb = torch.empty(b.shape, device=h.device, dtype=torch.float32)
        loss = torch.empty(N, device=h.device, dtype=torch.float32)
        for i in range(0, N, chunk):
            hc = hd[i:i + chunk]
            logits = hc @ wd.T
            ci = cand[i:i + chunk]
            logits.scatter_add_(1, ci, b[i:i + chunk].to(logits.dtype))
            _ce_grad_kernel[(hc.shape[0],)](logits, y[i:i + chunk], loss[i:i + chunk], logits.stride(0), V, BLOCK=4096)
            gb[i:i + chunk] = logits.gather(1, ci).float()            # градієнт логіта = градієнт зсуву
            gh[i:i + chunk] = logits @ wd
            gw += (logits.T @ hc).float()
        ctx.save_for_backward(gh, gw, gb)
        ctx.dtypes = (h.dtype, w.dtype, b.dtype)
        return loss.sum()

    @staticmethod
    def backward(ctx, go):
        gh, gw, gb = ctx.saved_tensors
        return (gh * go).to(ctx.dtypes[0]), (gw * go).to(ctx.dtypes[1]), None, None, (gb * go).to(ctx.dtypes[2]), None


def linear_ce_sparse(h, w, y, cand, b, chunk=4096):
    return _LinearCESparse.apply(h.reshape(-1, h.shape[-1]), w, y.reshape(-1), cand.reshape(-1, cand.shape[-1]).long(),
                                 b.reshape(-1, b.shape[-1]), chunk)


@triton.jit
def _ce_mix_kernel(logits_ptr, y_ptr, g_ptr, hit_ptr, loss_ptr, dg_ptr, stride, V, BLOCK: tl.constexpr):
    # «легка голова перебиває» (Стефан): P = (1-g)*softmax(x)[y] + g*hit; loss = -log P;
    # логіти перезаписуються градієнтом w*(softmax - onehot), w = (1-g)*p_y/P; dL/dg = (p_y - hit)/P
    row = tl.program_id(0)
    ptr = logits_ptr + row * stride
    y = tl.load(y_ptr + row)
    gm = tl.load(g_ptr + row)
    hit = tl.load(hit_ptr + row)
    m = -float("inf")
    s = 0.0
    for off in range(0, V, BLOCK):
        cols = off + tl.arange(0, BLOCK)
        x = tl.load(ptr + cols, mask=cols < V, other=-float("inf")).to(tl.float32)
        m_new = tl.maximum(m, tl.max(x, 0))
        s = s * tl.exp(m - m_new) + tl.sum(tl.exp(x - m_new), 0)
        m = m_new
    lse = m + tl.log(s)
    xy = tl.load(ptr + y).to(tl.float32)
    py = tl.exp(xy - lse)
    P = (1 - gm) * py + gm * hit
    P = tl.maximum(P, 1e-30)
    tl.store(loss_ptr + row, -tl.log(P))
    tl.store(dg_ptr + row, (py - hit) / P)
    wgt = (1 - gm) * py / P
    for off in range(0, V, BLOCK):
        cols = off + tl.arange(0, BLOCK)
        mask = cols < V
        x = tl.load(ptr + cols, mask=mask, other=0.0).to(tl.float32)
        g = wgt * (tl.exp(x - lse) - (cols == y).to(tl.float32))
        tl.store(ptr + cols, g.to(logits_ptr.dtype.element_ty), mask=mask)


class _LinearCEMix(torch.autograd.Function):
    @staticmethod
    def forward(ctx, h, w, y, gate, hit, chunk):
        N, d = h.shape
        V = w.shape[0]
        hd, wd = h.to(torch.bfloat16), w.to(torch.bfloat16)
        gh = torch.empty_like(hd)
        gw = torch.zeros(V, d, device=h.device, dtype=torch.float32)
        loss = torch.empty(N, device=h.device, dtype=torch.float32)
        dg = torch.empty(N, device=h.device, dtype=torch.float32)
        gf, hf = gate.float().contiguous(), hit.float().contiguous()
        for i in range(0, N, chunk):
            hc = hd[i:i + chunk]
            logits = hc @ wd.T
            _ce_mix_kernel[(hc.shape[0],)](logits, y[i:i + chunk], gf[i:i + chunk], hf[i:i + chunk], loss[i:i + chunk],
                                           dg[i:i + chunk], logits.stride(0), V, BLOCK=4096)
            gh[i:i + chunk] = logits @ wd
            gw += (logits.T @ hc).float()
        ctx.save_for_backward(gh, gw, dg)
        ctx.dtypes = (h.dtype, w.dtype, gate.dtype)
        return loss.sum()

    @staticmethod
    def backward(ctx, go):
        gh, gw, dg = ctx.saved_tensors
        return (gh * go).to(ctx.dtypes[0]), (gw * go).to(ctx.dtypes[1]), None, (dg * go).to(ctx.dtypes[2]), None, None


@torch.compiler.disable                       # поза графом: компілятор не вміє вбудувати цей кернел
def linear_ce_mix(h, w, y, gate, hit, chunk=4096):
    return _LinearCEMix.apply(h.reshape(-1, h.shape[-1]), w, y.reshape(-1), gate.reshape(-1), hit.reshape(-1), chunk)


def linear_ce(h, w, y, chunk=4096):
    """Сума крос-ентропії для logits = h @ w.T, без збереження логітів."""
    return _LinearCE.apply(h.reshape(-1, h.shape[-1]), w, y.reshape(-1), chunk)


@triton.jit
def _swiglu_fwd(gu_ptr, out_ptr, H, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    col = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    mask = col < H
    g = tl.load(gu_ptr + row * 2 * H + col, mask=mask, other=0.0).to(tl.float32)
    u = tl.load(gu_ptr + row * 2 * H + H + col, mask=mask, other=0.0).to(tl.float32)
    tl.store(out_ptr + row * H + col, (g * tl.sigmoid(g) * u).to(out_ptr.dtype.element_ty), mask=mask)


@triton.jit
def _swiglu_bwd(gu_ptr, go_ptr, dgu_ptr, H, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    col = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    mask = col < H
    g = tl.load(gu_ptr + row * 2 * H + col, mask=mask, other=0.0).to(tl.float32)
    u = tl.load(gu_ptr + row * 2 * H + H + col, mask=mask, other=0.0).to(tl.float32)
    go = tl.load(go_ptr + row * H + col, mask=mask, other=0.0).to(tl.float32)
    sg = tl.sigmoid(g)
    silu = g * sg
    dg = go * u * (sg + silu * (1 - sg))
    du = go * silu
    tl.store(dgu_ptr + row * 2 * H + col, dg.to(dgu_ptr.dtype.element_ty), mask=mask)
    tl.store(dgu_ptr + row * 2 * H + H + col, du.to(dgu_ptr.dtype.element_ty), mask=mask)


class _SwiGLU(torch.autograd.Function):
    @staticmethod
    def forward(ctx, gu):
        gu = gu.contiguous()
        shp = gu.shape
        gu2 = gu.view(-1, shp[-1])
        H = shp[-1] // 2
        out = torch.empty(gu2.shape[0], H, device=gu.device, dtype=gu.dtype)
        grid = (gu2.shape[0], triton.cdiv(H, 1024))
        _swiglu_fwd[grid](gu2, out, H, BLOCK=1024)
        ctx.save_for_backward(gu2)
        ctx.shp = shp
        return out.view(*shp[:-1], H)

    @staticmethod
    def backward(ctx, go):
        (gu2,) = ctx.saved_tensors
        H = gu2.shape[1] // 2
        go2 = go.contiguous().view(-1, H)
        dgu = torch.empty_like(gu2)
        grid = (gu2.shape[0], triton.cdiv(H, 1024))
        _swiglu_bwd[grid](gu2, go2, dgu, H, BLOCK=1024)
        return dgu.view(ctx.shp)


def swiglu(gu):
    """silu(g) * u для gu = [g, u] по останньому виміру — одним кернелом уперед і одним назад."""
    return _SwiGLU.apply(gu)
