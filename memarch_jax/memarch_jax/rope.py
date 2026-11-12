"""Маршрутизована увага RoPE з якорями по здивуванню (чистий XLA: jnp, lax).

Пошук якорів по ланцюжках (links) — точний (exact) або ієрархічний (hier) по блоках BM запитів.
Читання — спільний softmax у f32 над локальним вікном win та зібраними словами відкритих якорів."""
from dataclasses import dataclass
import math
from typing import Optional

import jax
from jax import lax
import jax.numpy as jnp


@dataclass(frozen=True)
class RopeCfg:
    K: int = 4
    links: int = 8
    win: int = 64
    span_cap: int = 256
    gather_cap: Optional[int] = None   # M, max gathered words per query; None -> K*span_cap (exact, cannot overflow)
    route: str = "hier"                # "hier" | "exact"
    G: int = 64
    NB: int = 16
    BM: int = 64                       # hier: anchors/group, groups per query block, queries per block
    excl_win: bool = True
    route_rot: bool = True
    rgrad: bool = True
    rgrad_scale: float = 1.0
    chunk: int = 256                   # queries per attention chunk (lax.map + jax.checkpoint)
    # Прапори швидкості (типово вимкнені; семантика та сама, див. tests/test_rope.py::test_fast_flags):
    route_lazy: bool = False           # hier: позиція ланки лише для top-K кандидатів, а не для всіх (менше скалярних gather)
    pallas: bool = False               # ядра rope_pallas (лише route="exact", route_rot=True); gather_cap не обрізає
    route_bf16: bool = False           # пошук: bf16-операнди з f32-накопиченням замість явного f32 (ті самі добутки, MXU-шлях)
    dense_read: bool = False           # читання без gather/scatter: q·k по всьому T + маска спанів; gather_cap не обрізає
    read: str = "v3"                   # pallas: "v1" — rope_pallas.read; "v3" — rope_read_v3.read (та сама математика, швидше)
    mass_pos: bool = False             # stats["_mass_pos"] (B,T): частка троса по позиціях (нагадування); False — як було
    mgrad: bool = False                # mass диференційовна (штраф --remind_use_w); False — під stop_gradient, як було


def anchors_from_surprise(s, tau, A):
    """s: (B, T) f32, s[:, 0] вже 1e4. Повертає own (B, T), apos (B, A), avalid (B, A)."""
    B, T = s.shape
    anchor = (s > tau).at[:, 0].set(True)
    keep = anchor & (jnp.cumsum(anchor.astype(jnp.int32), axis=1) <= A)
    own = jnp.cumsum(keep.astype(jnp.int32), axis=1) - 1
    pos = jnp.arange(T, dtype=jnp.int32)[None, :]
    key = (~keep).astype(jnp.int32) * T + pos
    order = jnp.argsort(key, axis=-1)
    if A <= T:
        apos = order[:, :A].astype(jnp.int32)
        avalid = jnp.take_along_axis(keep, apos, axis=1)
    else:
        apos0 = order.astype(jnp.int32)
        avalid0 = jnp.take_along_axis(keep, apos0, axis=1)
        apos = jnp.pad(apos0, ((0, 0), (0, A - T)), constant_values=0)
        avalid = jnp.pad(avalid0, ((0, 0), (0, A - T)), constant_values=False)
    return own, apos, avalid


def next_anchor(apos, avalid, T):
    """Позиція наступного дійсного якоря або T для останнього / недійсних: (B, A)."""
    nv = jnp.concatenate([avalid[:, 1:], jnp.zeros_like(avalid[:, :1])], axis=1)
    na = jnp.concatenate([apos[:, 1:], apos[:, :1]], axis=1)
    return jnp.where(nv, na, jnp.full_like(apos, T))


def _links(kf, own, apos, avalid, L):
    """Таблиця ключів-ланок (B, H, A, L, hd), маска валідності (B, A, L) та позиції (B, A, L)."""
    B, H, T, hd = kf.shape
    A = apos.shape[1]
    lp = apos[:, :, None] + jnp.arange(L, dtype=jnp.int32)[None, None, :]
    in_bounds = lp < T
    lp = jnp.clip(lp, 0, T - 1)
    own_lp = jnp.take_along_axis(own, lp.reshape(B, A * L), axis=1).reshape(B, A, L)
    a_idx = jnp.arange(A, dtype=jnp.int32)[None, :, None]
    lval = in_bounds & avalid[:, :, None] & (own_lp == a_idx)
    kl = jnp.take_along_axis(kf, lp.reshape(B, 1, A * L, 1), axis=2).reshape(B, H, A, L, hd)
    return kl, lval, lp


def _route_nxt(nxt, apos, cfg):
    """Поріг відкриття якоря для запиту i: max(nxt, apos + win) при excl_win, інакше nxt."""
    return jnp.maximum(nxt, apos + cfg.win) if cfg.excl_win else nxt


def _route_dot(spec, a, b, cfg):
    if cfg.route_bf16:
        return jnp.einsum(spec, a, b, precision=lax.Precision.HIGHEST, preferred_element_type=jnp.float32)
    return jnp.einsum(spec, a.astype(jnp.float32), b.astype(jnp.float32), precision=lax.Precision.HIGHEST)


def _route_exact(q0, k0, own, apos, avalid, cfg):
    B, H, T, hd = q0.shape
    A = apos.shape[1]
    kk = min(cfg.K, A)
    L = min(cfg.links, cfg.span_cap)
    BM = cfg.BM
    sc = 1.0 / math.sqrt(hd)

    kl, lval, lp = _links(k0, own, apos, avalid, L)
    nxt = next_anchor(apos, avalid, T)
    nx = _route_nxt(nxt, apos, cfg)

    NMB = (T + BM - 1) // BM
    T_pad = NMB * BM
    q_pad = jnp.pad(q0, ((0, 0), (0, 0), (0, T_pad - T), (0, 0)))
    q_blk = jnp.moveaxis(q_pad.reshape(B, H, NMB, BM, hd), 2, 0)
    b_ids = jnp.arange(NMB, dtype=jnp.int32)

    def _blk_fn(elem):
        m_blk, qb = elem
        qi = m_blk * BM + jnp.arange(BM, dtype=jnp.int32)
        s = _route_dot("bhmd,bhald->bhmal", qb, kl, cfg) * sc
        s = jnp.where(lval[:, None, None, :, :], s, -jnp.inf)
        best_l = jnp.argmax(s, axis=-1).astype(jnp.int32)
        s_a = jnp.max(s, axis=-1)
        if not cfg.route_lazy:
            lp_a = jnp.take_along_axis(lp[:, None, None, :, :], best_l[..., None], axis=-1)[..., 0]
        ok = avalid[:, None, :] & (nx[:, None, :] <= qi[None, :, None])
        s_a = jnp.where(ok[:, None, :, :], s_a, -jnp.inf)
        top_v, top_i = lax.top_k(s_a, kk)
        tval_b = top_v > -jnp.inf
        tidx_b = jnp.where(tval_b, top_i.astype(jnp.int32), 0)
        if cfg.route_lazy:                     # лише K рядків на запит замість усіх A
            flat = top_i * L + jnp.take_along_axis(best_l, top_i, axis=-1)
            tpos_raw = jnp.take_along_axis(lp.reshape(B, 1, 1, -1), flat, axis=-1)
        else:
            tpos_raw = jnp.take_along_axis(lp_a, top_i, axis=-1)
        tpos_b = jnp.where(tval_b, tpos_raw.astype(jnp.int32), 0)
        return tidx_b, tval_b, tpos_b

    tidx_m, tval_m, tpos_m = lax.map(_blk_fn, (b_ids, q_blk))
    tidx = jnp.moveaxis(tidx_m, 0, 2).reshape(B, H, T_pad, kk)[:, :, :T, :]
    tval = jnp.moveaxis(tval_m, 0, 2).reshape(B, H, T_pad, kk)[:, :, :T, :]
    tpos = jnp.moveaxis(tpos_m, 0, 2).reshape(B, H, T_pad, kk)[:, :, :T, :]
    return tidx, tval, tpos


def _route_hier(q0, k0, qg, kg, own, apos, avalid, cfg):
    B, H, T, hd = q0.shape
    A = apos.shape[1]
    kk = min(cfg.K, A)
    L = min(cfg.links, cfg.span_cap)
    G, NB, BM = cfg.G, cfg.NB, cfg.BM
    sc = 1.0 / math.sqrt(hd)

    AP0 = ((A + 15) // 16) * 16
    NG = (AP0 + G - 1) // G
    AP = NG * G
    pad_A = AP - A

    kl, lval, lp = _links(k0, own, apos, avalid, L)
    klg, _, _ = _links(kg, own, apos, avalid, L)
    nxt = next_anchor(apos, avalid, T)
    nx = _route_nxt(nxt, apos, cfg)

    klg_p = jnp.pad(klg, ((0, 0), (0, 0), (0, pad_A), (0, 0), (0, 0)))
    lval_p = jnp.pad(lval, ((0, 0), (0, pad_A), (0, 0)), constant_values=False)

    w = lval_p.reshape(B, 1, NG, G, L, 1).astype(jnp.float32)
    s1 = (klg_p.reshape(B, H, NG, G, L, hd).astype(jnp.float32) * w).sum(axis=(3, 4))
    s1 = s1 / jnp.maximum(w.sum(axis=(3, 4)), 1.0)

    NMB = (T + BM - 1) // BM
    T_pad = NMB * BM
    m0s = jnp.arange(NMB, dtype=jnp.int32) * BM
    ends = jnp.minimum(m0s + BM, T) - 1

    row_valid = (m0s[:, None] + jnp.arange(BM, dtype=jnp.int32)[None, :]) < T
    n_real = jnp.maximum(row_valid.sum(axis=-1).astype(jnp.float32), 1.0)
    qg_pad = jnp.pad(qg, ((0, 0), (0, 0), (0, T_pad - T), (0, 0)))
    qg_blk = qg_pad.reshape(B, H, NMB, BM, hd)
    dots = jnp.einsum("bhnrd,bhgd->bhnrg", qg_blk.astype(jnp.float32), s1,
                      precision=lax.Precision.HIGHEST)
    gsc = jnp.where(row_valid[None, None, :, :, None], dots, 0.0).sum(axis=3) / n_real[None, None, :, None]
    # причинність: групи для блоку n вибирають запити блоку n-1 (усі вони в минулому кожного запиту блоку n).
    # Оцінка самим блоком n залежала б від майбутніх запитів у блоці. Блок 0 повних груп не має (nfg=0) — нічого не губить.
    gsc = jnp.concatenate([jnp.zeros_like(gsc[:, :, :1]), gsc[:, :, :-1]], axis=2)

    c0 = (avalid[:, None, :] & (nxt[:, None, :] <= m0s[None, :, None])).sum(axis=-1)
    nfg = c0 // G
    g_idx = jnp.arange(NG, dtype=jnp.int32)[None, None, :]
    gok = g_idx < nfg[:, :, None]
    if cfg.excl_win:
        cw = (avalid[:, None, :] & (apos[:, None, :] + cfg.win <= ends[None, :, None])).sum(axis=-1)
        gok = gok & (g_idx * G < cw[:, :, None])
    gsc = jnp.where(gok[:, None, :, :], gsc, -jnp.inf)

    if NG < NB:
        gsc = jnp.pad(gsc, ((0, 0), (0, 0), (0, 0), (0, NB - NG)), constant_values=-jnp.inf)
    top_gv, top_gi = lax.top_k(gsc, NB)
    sel = jnp.where(top_gv > -jnp.inf, top_gi.astype(jnp.int32), -1)

    grp_a = sel[:, :, :, :, None] * G + jnp.arange(G, dtype=jnp.int32)[None, None, None, None, :]
    grp_valid = (sel[:, :, :, :, None] >= 0) & (grp_a < A)
    grp_a = grp_a.reshape(B, H, NMB, NB * G)
    grp_valid = grp_valid.reshape(B, H, NMB, NB * G)

    tail = G + BM
    tail_a = nfg[:, None, :, None] * G + jnp.arange(tail, dtype=jnp.int32)[None, None, None, :]
    tail_a = jnp.broadcast_to(tail_a, (B, H, NMB, tail))
    tail_valid = (tail_a >= 0) & (tail_a < A)

    cand_a = jnp.concatenate([grp_a, tail_a], axis=-1)
    cand_valid = jnp.concatenate([grp_valid, tail_valid], axis=-1)
    cand_a = jnp.where(cand_valid, cand_a, 0)

    q_pad = jnp.pad(q0, ((0, 0), (0, 0), (0, T_pad - T), (0, 0)))
    q_blk = jnp.moveaxis(q_pad.reshape(B, H, NMB, BM, hd), 2, 0)
    ca_blk = jnp.moveaxis(cand_a, 2, 0)
    cv_blk = jnp.moveaxis(cand_valid, 2, 0)
    b_ids = jnp.arange(NMB, dtype=jnp.int32)

    def _blk_fn(elem):
        m_blk, qb, ca, cv = elem
        qi = m_blk * BM + jnp.arange(BM, dtype=jnp.int32)
        kl_c = jnp.take_along_axis(kl, ca[:, :, :, None, None], axis=2)
        lval_c = jnp.take_along_axis(lval[:, None, :, :], ca[:, :, :, None], axis=2)
        lp_c = jnp.take_along_axis(lp[:, None, :, :], ca[:, :, :, None], axis=2)
        av_c = jnp.take_along_axis(avalid[:, None, :], ca, axis=2)
        nx_c = jnp.take_along_axis(nx[:, None, :], ca, axis=2)

        s = _route_dot("bhmd,bhcld->bhmcl", qb, kl_c, cfg) * sc
        s = jnp.where(lval_c[:, :, None, :, :], s, -jnp.inf)
        best_l = jnp.argmax(s, axis=-1).astype(jnp.int32)
        s_c = jnp.max(s, axis=-1)
        if not cfg.route_lazy:
            lp_best = jnp.take_along_axis(lp_c[:, :, None, :, :], best_l[..., None], axis=-1)[..., 0]

        ok = cv[:, :, None, :] & av_c[:, :, None, :] & (nx_c[:, :, None, :] <= qi[None, None, :, None])
        s_c = jnp.where(ok, s_c, -jnp.inf)
        top_v, top_c = lax.top_k(s_c, kk)
        tval_b = top_v > -jnp.inf
        tidx_raw = jnp.take_along_axis(ca[:, :, None, :], top_c, axis=-1)
        if cfg.route_lazy:                     # лише K рядків на запит замість усіх кандидатів
            flat = top_c * L + jnp.take_along_axis(best_l, top_c, axis=-1)
            tpos_raw = jnp.take_along_axis(lp_c.reshape(B, H, 1, -1), flat, axis=-1)
        else:
            tpos_raw = jnp.take_along_axis(lp_best, top_c, axis=-1)
        tidx_b = jnp.where(tval_b, tidx_raw.astype(jnp.int32), 0)
        tpos_b = jnp.where(tval_b, tpos_raw.astype(jnp.int32), 0)
        return tidx_b, tval_b, tpos_b

    tidx_m, tval_m, tpos_m = lax.map(_blk_fn, (b_ids, q_blk, ca_blk, cv_blk))
    tidx = jnp.moveaxis(tidx_m, 0, 2).reshape(B, H, T_pad, kk)[:, :, :T, :]
    tval = jnp.moveaxis(tval_m, 0, 2).reshape(B, H, T_pad, kk)[:, :, :T, :]
    tpos = jnp.moveaxis(tpos_m, 0, 2).reshape(B, H, T_pad, kk)[:, :, :T, :]
    return tidx, tval, tpos


def route(q0, k0, qg, kg, own, apos, avalid, cfg):
    """Маршрутизація якорів під stop_gradient. Повертає (tidx, tval, tpos) форми (B, H, T, kk)."""
    q0 = lax.stop_gradient(q0)
    k0 = lax.stop_gradient(k0)
    qg = q0 if qg is None else lax.stop_gradient(qg)
    kg = k0 if kg is None else lax.stop_gradient(kg)
    own = lax.stop_gradient(own)
    apos = lax.stop_gradient(apos)
    avalid = lax.stop_gradient(avalid)
    if cfg.route == "exact":
        tidx, tval, tpos = _route_exact(q0, k0, own, apos, avalid, cfg)
    elif cfg.route == "hier":
        tidx, tval, tpos = _route_hier(q0, k0, qg, kg, own, apos, avalid, cfg)
    else:
        raise ValueError(f"Unknown route mode: {cfg.route}")
    return lax.stop_gradient(tidx), lax.stop_gradient(tval), lax.stop_gradient(tpos)


def _qmask_tval(tval, qmask, docs=None, tidx=None, apos=None):
    """ROPE_QFRAC: tval (B,H,T,K) & qmask (B,T) — слова поза часткою не відкривають жодного якоря (лише вікно).
    docs (B,T) int (cfg.rope_doc_eos): якір з іншого документа, ніж запит, недійсний (без змін у ядрах; лише маска)."""
    if qmask is not None:
        tval = tval & qmask[:, None, :, None]
    if docs is not None:
        B, H, T, K = tidx.shape
        doc_a = jnp.take_along_axis(docs, apos, axis=1)                          # (B,A) документ кожного якоря
        da = jnp.take_along_axis(doc_a, tidx.reshape(B, -1), axis=1).reshape(B, H, T, K)
        tval = tval & (da == docs[:, None, :, None])
    return tval


def _rope_attention_pallas(qr, kr, v, own, apos, avalid, cfg, routed=None, qmask=None, docs=None):
    """Те саме, що rope_attention(route="exact", route_rot=True), через ядра rope_pallas. Пошук під stop_gradient."""
    from . import rope_pallas as rp
    assert cfg.route == "exact" and cfg.route_rot, "pallas: лише exact route і route_rot"
    B, H, T, hd = qr.shape
    A = apos.shape[1]
    kk = min(cfg.K, A)
    C = cfg.chunk
    assert T % C == 0 and T % cfg.BM == 0, "pallas: T має ділитися на chunk і BM"
    L = min(cfg.links, cfg.span_cap)
    q0, k0 = lax.stop_gradient(qr), lax.stop_gradient(kr)
    sg_own, sg_apos, sg_av = (lax.stop_gradient(x) for x in (own, apos, avalid))
    nxt = next_anchor(sg_apos, sg_av, T)
    pp = rp.P(H, T, hd, kk, cfg.win, C, min(2 * C, T), 1.0 / math.sqrt(hd), cfg.rgrad_scale / math.sqrt(hd))
    if routed is None:
        kl, lval, lp = _links(k0, sg_own, sg_apos, sg_av, L)
        tidx, tval, tpos, tsc, st, nx_t = rp.route_exact(q0, kl, lval, lp, sg_av, _route_nxt(nxt, sg_apos, cfg), sg_apos, nxt, kk, cfg.BM)
    else:
        tidx, tval, tpos, tsc, st, nx_t = routed
        if cfg.rgrad:
            tsc = rp.bias_forward(q0, k0, tpos, tval, pp)        # оцінка цього шару в позиції вибору іншого шару
    tval = _qmask_tval(tval, qmask, docs, tidx, apos)            # ROPE_QFRAC / межа документа: порожні спани
    qi = jnp.arange(T, dtype=jnp.int32)[None, None, :, None]
    en = jnp.minimum(jnp.minimum(nx_t, st + cfg.span_cap), qi - cfg.win + 1)
    empty = (~tval) | (en <= st)
    st_r = jnp.where(empty, T, st)
    en_r = jnp.where(empty, 0, en)
    gb = jnp.where(tval, tsc * cfg.rgrad_scale, 0.0) if cfg.rgrad else jnp.zeros_like(tsc)
    if cfg.read == "v3":
        from . import rope_read_v3 as r3
        BQ3, BK3 = r3.sizes(T)
        p3 = r3.P3(T, hd, kk, cfg.win, BQ3, BK3, pp.sc, pp.sc_bias, mgrad=cfg.mgrad)
        kid = r3.key_ids(sg_own, sg_apos, cfg.span_cap)
        qid3 = jnp.where(empty, -1, tidx)                        # порожній проліт не відкриває якоря
        out, mass = r3.read(qr, kr, v, kid, lax.stop_gradient(qid3), lax.stop_gradient(gb), lax.stop_gradient(tpos),
                            p3, rp._interp(), cfg.rgrad)
        share = mass.sum() / float(B * H * T)
        st3 = {"rope_overflow": jnp.zeros((), jnp.float32), "_route": (tidx, tval, tpos, tsc, st, nx_t)}
        if cfg.mass_pos:
            mp = mass.reshape(B, H, T).mean(1)
            st3["_mass_pos"] = mp if cfg.mgrad else lax.stop_gradient(mp)
        return out, share, st3
    out, mass = rp.read(qr, kr, v, lax.stop_gradient(st_r), lax.stop_gradient(en_r), lax.stop_gradient(gb),
                        lax.stop_gradient(tpos), pp, rp._interp(), cfg.rgrad)
    share = mass.sum() / float(B * H * T)
    stats = {"rope_overflow": jnp.zeros((), jnp.float32), "_route": (tidx, tval, tpos, tsc, st, nx_t)}
    assert not cfg.mgrad, "--remind_use_w з pallas: лише --rope_read v3 (v1 не має похідної mass)"
    if cfg.mass_pos and mass.size == B * H * T:
        stats["_mass_pos"] = lax.stop_gradient(mass.reshape(B, H, T).mean(1))
    return out, share, stats


def rope_attention(qr, kr, qu, ku, v, own, apos, avalid, cfg, routed=None, qmask=None, docs=None):
    """Увага над локальним вікном win та зібраними словами top-K якорів. Повертає (out, share, stats).
    routed — готовий (tidx, tval, tpos) від іншого шару (спільний пошук); тоді власний пошук не виконується.
    Вибір цього шару — у stats["_route"]."""
    if cfg.pallas:
        return _rope_attention_pallas(qr, kr, v, own, apos, avalid, cfg, routed, qmask, docs)
    B, H, T, hd = qr.shape
    A = apos.shape[1]
    kk = min(cfg.K, A)
    W = cfg.win
    C = cfg.chunk
    M = (cfg.K * cfg.span_cap) if cfg.gather_cap is None else cfg.gather_cap
    sc = 1.0 / math.sqrt(hd)

    q0, k0 = (qr, kr) if cfg.route_rot else (qu, ku)
    tidx, tval, tpos = route(q0, k0, qu, ku, own, apos, avalid, cfg) if routed is None else routed
    tval = _qmask_tval(tval, qmask, docs, tidx, apos)            # ROPE_QFRAC / межа документа
    nxt = lax.stop_gradient(next_anchor(apos, avalid, T))
    apos_sg = lax.stop_gradient(apos)

    NC = (T + C - 1) // C
    T_pad = NC * C
    pad_T = T_pad - T

    qr_pad = jnp.pad(qr, ((0, 0), (0, 0), (0, pad_T), (0, 0)))
    tidx_pad = jnp.pad(tidx, ((0, 0), (0, 0), (0, pad_T), (0, 0)))
    tval_pad = jnp.pad(tval, ((0, 0), (0, 0), (0, pad_T), (0, 0)), constant_values=False)
    tpos_pad = jnp.pad(tpos, ((0, 0), (0, 0), (0, pad_T), (0, 0)))

    kr_win_pad = jnp.pad(kr, ((0, 0), (0, 0), (W - 1, pad_T), (0, 0)))
    v_win_pad = jnp.pad(v, ((0, 0), (0, 0), (W - 1, pad_T), (0, 0)))

    qr_blk = jnp.moveaxis(qr_pad.reshape(B, H, NC, C, hd), 2, 0)
    tidx_blk = jnp.moveaxis(tidx_pad.reshape(B, H, NC, C, kk), 2, 0)
    tval_blk = jnp.moveaxis(tval_pad.reshape(B, H, NC, C, kk), 2, 0)
    tpos_blk = jnp.moveaxis(tpos_pad.reshape(B, H, NC, C, kk), 2, 0)
    q0_blk = jnp.moveaxis(jnp.pad(q0, ((0, 0), (0, 0), (0, pad_T), (0, 0))).reshape(B, H, NC, C, hd), 2, 0)
    c_ids = jnp.arange(NC, dtype=jnp.int32)
    CW = C + W - 1

    def _dense_tail(kr, v, k0, kr_win_pad, v_win_pad, c_idx, qr_c, q0_c, tval_c, tpos_c, st, en, gl, tot_c):
        """Те саме читання без збирання рядків: спани — маска над усіма T (спани різних якорів не перетинаються,
        тож кожне слово входить рівно раз, як і в зібраному списку). Обрізання gather_cap тут немає (lost = 0)."""
        q_start = c_idx * C
        qi = q_start + jnp.arange(C, dtype=jnp.int32)
        q_valid = qi < T
        t = jnp.arange(T, dtype=jnp.int32)[None, None, None, None, :]
        in_k = lax.stop_gradient((st[..., None] <= t) & (t < en[..., None]) & (gl[..., None] > 0))   # (B,H,C,kk,T)
        mask = in_k.any(axis=-2)
        sg = jnp.einsum("bhcd,bhtd->bhct", qr_c, kr, precision=lax.Precision.HIGHEST,
                        preferred_element_type=jnp.float32) * sc
        if cfg.rgrad:
            kb = jnp.take_along_axis(k0, tpos_c.reshape(B, H, C * kk, 1), axis=2).reshape(B, H, C, kk, hd)
            rsel = jnp.einsum("bhcd,bhckd->bhck", q0_c, kb, precision=lax.Precision.HIGHEST,
                              preferred_element_type=jnp.float32) * sc
            gb_c = jnp.where(tval_c, rsel * cfg.rgrad_scale, 0.0)
            sg = sg + jnp.einsum("bhck,bhckt->bhct", gb_c, in_k.astype(jnp.float32))
        sg = jnp.where(mask, sg, -jnp.inf)

        kw_blk = lax.dynamic_slice(kr_win_pad, (0, 0, q_start, 0), (B, H, CW, hd))
        vw_blk = lax.dynamic_slice(v_win_pad, (0, 0, q_start, 0), (B, H, CW, hd))
        sw = jnp.einsum("bhcd,bhwd->bhcw", qr_c, kw_blk, precision=lax.Precision.HIGHEST,
                        preferred_element_type=jnp.float32) * sc
        kj = q_start - (W - 1) + jnp.arange(CW, dtype=jnp.int32)
        w_ok = (kj[None, :] >= 0) & (kj[None, :] <= qi[:, None]) & (qi[:, None] - kj[None, :] < W)
        sw = jnp.where(w_ok[None, None, :, :], sw, -jnp.inf)

        p_all = jax.nn.softmax(jnp.concatenate([sw, sg], axis=-1), axis=-1)
        pw, pg = p_all[..., :CW], p_all[..., CW:]
        out_w = jnp.einsum("bhcw,bhwd->bhcd", pw, vw_blk.astype(jnp.float32), precision=lax.Precision.HIGHEST)
        out_g = jnp.einsum("bhct,bhtd->bhcd", pg, v.astype(jnp.float32), precision=lax.Precision.HIGHEST)
        out_c = (out_w + out_g).astype(v.dtype)
        out_c = jnp.where(q_valid[None, None, :, None], out_c, jnp.zeros_like(out_c))
        share_c = lax.stop_gradient(jnp.where(q_valid[None, None, :], pg.sum(axis=-1), 0.0).sum())
        if cfg.mass_pos:
            mp = pg.sum(axis=-1).mean(1)
            return out_c, share_c, jnp.zeros((), jnp.float32), tot_c, (mp if cfg.mgrad else lax.stop_gradient(mp))
        return out_c, share_c, jnp.zeros((), jnp.float32), tot_c

    # kr, v, вікна — явні аргументи: диференціювання крізь jax.checkpoint не покладається на замикання
    @jax.checkpoint
    def _chunk_fn(kr, v, k0, kr_win_pad, v_win_pad, elem):
        c_idx, qr_c, q0_c, tidx_c, tval_c, tpos_c = elem
        q_start = c_idx * C
        qi = q_start + jnp.arange(C, dtype=jnp.int32)
        q_valid = qi < T

        st = jnp.take_along_axis(apos_sg, tidx_c.reshape(B, -1), axis=1).reshape(B, H, C, kk)
        nx_t = jnp.take_along_axis(nxt, tidx_c.reshape(B, -1), axis=1).reshape(B, H, C, kk)
        en = jnp.minimum(jnp.minimum(nx_t, st + cfg.span_cap), qi[None, None, :, None] - W + 1)
        gl = jnp.maximum(en - st, 0) * (tval_c & q_valid[None, None, :, None]).astype(jnp.int32)

        c_end = jnp.cumsum(gl, axis=-1)
        c_start = c_end - gl
        tot = c_end[..., -1]
        lost_c = lax.stop_gradient(jnp.maximum(tot - M, 0).sum().astype(jnp.float32))
        tot_c = lax.stop_gradient(tot.sum().astype(jnp.float32))

        if cfg.dense_read:
            return _dense_tail(kr, v, k0, kr_win_pad, v_win_pad, c_idx, qr_c, q0_c, tval_c, tpos_c, st, en, gl, tot_c)

        m = jnp.arange(M, dtype=jnp.int32)[None, None, None, :]
        slot = jnp.minimum((m[..., None] >= c_end[..., None, :]).sum(axis=-1), kk - 1).astype(jnp.int32)
        st_m = jnp.take_along_axis(st, slot, axis=-1)
        cs_m = jnp.take_along_axis(c_start, slot, axis=-1)
        g_valid = m < tot[..., None]
        pos = jnp.clip(jnp.where(g_valid, st_m + (m - cs_m), 0), 0, T - 1)
        pos = lax.stop_gradient(pos)
        slot = lax.stop_gradient(slot)
        g_valid = lax.stop_gradient(g_valid)

        if cfg.rgrad:
            kb = jnp.take_along_axis(k0, tpos_c.reshape(B, H, C * kk, 1), axis=2).reshape(B, H, C, kk, hd)  # у векторах пошуку (ROUTE_ROT)
            rsel = jnp.einsum("bhcd,bhckd->bhck", q0_c, kb, precision=lax.Precision.HIGHEST,
                              preferred_element_type=jnp.float32) * sc
            gb_c = jnp.where(tval_c, rsel * cfg.rgrad_scale, 0.0)
            bias_m = jnp.take_along_axis(gb_c, slot, axis=-1)
        else:
            bias_m = 0.0

        kg_m = jnp.take_along_axis(kr, pos.reshape(B, H, C * M, 1), axis=2).reshape(B, H, C, M, hd)
        vg_m = jnp.take_along_axis(v, pos.reshape(B, H, C * M, 1), axis=2).reshape(B, H, C, M, hd)
        # без .astype(f32) на зібраних (C, M, hd): добутки bf16 точні у f32, накопичення — f32 (preferred_element_type)
        sg = jnp.einsum("bhcd,bhcmd->bhcm", qr_c, kg_m, precision=lax.Precision.HIGHEST,
                        preferred_element_type=jnp.float32) * sc + bias_m
        sg = jnp.where(g_valid, sg, -jnp.inf)

        kw_blk = lax.dynamic_slice(kr_win_pad, (0, 0, q_start, 0), (B, H, CW, hd))
        vw_blk = lax.dynamic_slice(v_win_pad, (0, 0, q_start, 0), (B, H, CW, hd))
        sw = jnp.einsum("bhcd,bhwd->bhcw", qr_c, kw_blk, precision=lax.Precision.HIGHEST,
                        preferred_element_type=jnp.float32) * sc
        kj = q_start - (W - 1) + jnp.arange(CW, dtype=jnp.int32)
        w_ok = (kj[None, :] >= 0) & (kj[None, :] <= qi[:, None]) & (qi[:, None] - kj[None, :] < W)
        sw = jnp.where(w_ok[None, None, :, :], sw, -jnp.inf)

        s_all = jnp.concatenate([sw, sg], axis=-1)
        p_all = jax.nn.softmax(s_all, axis=-1)
        pw = p_all[..., :CW]
        pg = p_all[..., CW:]

        out_w = jnp.einsum("bhcw,bhwd->bhcd", pw, vw_blk.astype(jnp.float32), precision=lax.Precision.HIGHEST)
        out_g = jnp.einsum("bhcm,bhcmd->bhcd", pg, vg_m.astype(jnp.float32), precision=lax.Precision.HIGHEST)
        out_c = (out_w + out_g).astype(v.dtype)
        out_c = jnp.where(q_valid[None, None, :, None], out_c, jnp.zeros_like(out_c))
        share_c = lax.stop_gradient(jnp.where(q_valid[None, None, :], pg.sum(axis=-1), 0.0).sum())
        if cfg.mass_pos:
            mp = pg.sum(axis=-1).mean(1)                         # (B,C) середнє по головах
            return out_c, share_c, lost_c, tot_c, (mp if cfg.mgrad else lax.stop_gradient(mp))
        return out_c, share_c, lost_c, tot_c

    res_m = lax.map(lambda e: _chunk_fn(kr, v, k0, kr_win_pad, v_win_pad, e), (c_ids, qr_blk, q0_blk, tidx_blk, tval_blk, tpos_blk))
    out_m, share_m, lost_m, tot_m = res_m[:4]
    out = jnp.moveaxis(out_m, 0, 2).reshape(B, H, T_pad, hd)[:, :, :T, :]
    share = lax.stop_gradient(share_m.sum() / float(B * H * T))
    overflow = lax.stop_gradient(lost_m.sum() / jnp.maximum(tot_m.sum(), 1.0))
    stats = {"rope_overflow": overflow, "_route": (tidx, tval, tpos)}
    if cfg.mass_pos:
        stats["_mass_pos"] = jnp.moveaxis(res_m[4], 0, 1).reshape(B, T_pad)[:, :T]
    return out, share, stats
