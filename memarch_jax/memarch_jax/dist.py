"""M9: одна синхронна модель на W чипах (design §2, §10) — Mesh з віссю "x", shard_map train_loss, крок Instinct.

Розкладка параметрів (param_specs):
  * emb, head, l_head1, book — рядки словника P('x', None): V/W рядків на чип (V має ділитися на W);
      збирання рядків — VocabShard.take (маскований gather своїх рядків + psum: одна ненульова частина → точно);
      вихід на словник — VocabShard.ce (як shard.short_logp_vocab: all_gather h, chunk_ce по своїх рядках,
      lse = logsumexp(all_gather(lse своїх рядків)), логіт слова — psum у власника рядка).
  * heavy.up / heavy.down — EP P(None, 'x', ...): E/W експертів на чип, обмін — moe.a2a_exchange (all_to_all).
  * усе інше (light, memory, reread, router, норми, скаляри) — репліковане P() (design §9.6, рекомендація (б)).

Градієнти беремо ЗОВНІ shard_map: loss = mean_r(loss_r) — середнє власних loss чипів (кожен loss_r нормований
на свої B·T), jax.grad транспонує shard_map сам:
  * репліковані ваги → середнє локальних градієнтів по чипах (= pmean);
  * EP- і vocab-шарди → свої рядки/експерти; транспозиція all_to_all / all_gather / psum уже сумує внески
    всіх чипів, а множник 1/W з mean дає ту саму семантику, що torch (reduce-scatter SUM/W).
Тому градієнти виходять глобальними jax.Array з тим самим шардуванням, що й параметри.

Синхронний стан (§10):
  * tau — pmean квантиля (anchors.quantile_hook): tau_new однаковий на всіх чипах;
  * loss для Instinct.prepare — mean по чипах (саме те, що повертає loss_fn);
  * ситуація — справжнє середнє вектора слів по батчу ВСІХ чипів (situation: psum своїх рядків / (W·B·T));
  * rel / норми Instinct — глобальні: instinct.step виконується поза shard_map над глобальними масивами під jit
    (XLA сам вставляє колективи для сум по шардованих осях), тож норма — по всьому тензору, а не по шматку.

Що НЕ синхронне (свідомо, design §0 / spec_1 A6): MoE-роутинг, місткість, черги, відкидання й aux — на власних
N словах чипа (місткість рахується від локального N); якорі — локальні (маска від спільного старого tau).
"""
from dataclasses import dataclass
from functools import partial

import numpy as np
import jax
import jax.numpy as jnp
from jax import lax
from jax.ad_checkpoint import checkpoint_name
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

from . import ce, model, moe, instinct

AXIS = "x"
F32 = jnp.float32
VOCAB_KEYS = ("emb", "head", "l_head1", "book")          # рядки словника — шардовані


try:                                                     # jax ≥0.6: jax.shard_map(check_vma=...)
    _shard_map = partial(jax.shard_map, check_vma=False)
except AttributeError:                                   # старіші: experimental, check_rep
    from jax.experimental.shard_map import shard_map as _sm
    _shard_map = partial(_sm, check_rep=False)


# ---------------------------------------------------------------- mesh і розкладка

def make_mesh(W=None):
    devs = jax.devices()
    W = len(devs) if W is None else W
    return Mesh(np.asarray(devs[:W]), (AXIS,))


def param_specs(params):
    """PartitionSpec для кожного листка (pytree тієї ж структури)."""
    def spec(path, x):
        top = instinct.path_str(path).split(".")[0]
        if top in VOCAB_KEYS:
            return P(AXIS, *([None] * (x.ndim - 1)))
        name = instinct.path_str(path)
        if name in ("heavy.up", "heavy.down"):                    # (L, E, ., .): вісь експертів
            return P(None, AXIS, None, None)
        return P()
    return jax.tree_util.tree_map_with_path(spec, params)


def put(tree, mesh, specs):
    """Розкласти pytree на чипи за specs (specs — pytree PartitionSpec або один PartitionSpec)."""
    if isinstance(specs, P):
        return jax.tree_util.tree_map(lambda x: jax.device_put(x, NamedSharding(mesh, specs)), tree)
    return jax.tree_util.tree_map(lambda x, s: jax.device_put(x, NamedSharding(mesh, s)), tree, specs,
                                  is_leaf=lambda s: isinstance(s, P))


# ---------------------------------------------------------------- шардований словник

@dataclass(frozen=True)
class VocabShard:
    """Таблиці (V, D), порізані рядками між W чипами: чип r тримає [r·rows, (r+1)·rows). Усередині shard_map."""
    V: int
    W: int
    axis: str = AXIS
    take_mode: str = "ag"          # "ag" — all_gather таблиці у fwd (новий, типовий); "masked" — старий RS-шлях
    take_bwd: str = "sort"         # bwd take: "sort" — stable sort + segment_sum (sorted); "scatter" — старий scatter-add
    ce_a2a: bool = True            # lse/логіт слова у ce: all_to_all/psum_scatter лише свого шматка замість AG/psum усього

    def _lo(self, rows):
        return lax.axis_index(self.axis) * rows

    def _part(self, Wl, ids):
        rows = Wl.shape[0]
        loc = ids - self._lo(rows)
        inside = (loc >= 0) & (loc < rows)
        return Wl[jnp.clip(loc, 0, rows - 1)] * inside[..., None].astype(Wl.dtype)

    def take(self, Wl, ids):
        """Wl (rows, D) свої рядки, ids (...) ВЛАСНІ номери чипа (на кожному чипі різні) → (..., D) повні рядки.
        take_mode="ag" (типово): fwd — all_gather таблиці (V,D) + локальний gather лише своїх N рядків (замість
        gather W·N рядків і reduce_scatter 1.34 GB, з яких 7/8 нулі); bwd — РІВНО транспозиція старого шляху
        (all_gather котангенса + маскований scatter-add у свої рядки), тож градієнт побітово той самий.
        fwd теж побітово: рядки просто копіюються (сума «одна ненульова частина + нулі» була точною).
        take_mode="masked" — старий шлях для A/B."""
        if self.take_mode == "masked":
            return self._take_masked(Wl, ids)
        return _take_ag(self, Wl, ids)

    def _take_masked(self, Wl, ids):
        """Старий шлях: all_gather номерів → кожен чип дає свої рядки для слів усіх чипів → psum_scatter."""
        e = self._part(Wl, lax.all_gather(ids, self.axis))                 # (W, ..., D)
        return lax.psum_scatter(e, self.axis, scatter_dimension=0, tiled=False)

    def take_same(self, Wl, ids):
        """Те саме для номерів, ОДНАКОВИХ на всіх чипах (короткий словник): psum без збирання номерів."""
        return lax.psum(self._part(Wl, ids), self.axis)

    def ce(self, h, Wl, b, y, ch):
        """(lse, логіт слова) для СВОЇХ слів h (N,d) по ВСЬОМУ словнику; Wl (rows,d) свої рядки, b (V,) fp32 зсув
        (повний, сталий), y (N,) глобальні номери. Те саме, що ce.chunk_ce(h, W, b, y, ch) на одному чипі."""
        rows, N = Wl.shape[0], h.shape[0]
        r = lax.axis_index(self.axis)
        lo = r * rows
        hg = lax.all_gather(h, self.axis, tiled=True)                      # (W·N, d) — місця всіх чипів
        hg = checkpoint_name(hg, "ce_h_gather")                            # політика remat може зберегти, щоб bwd не збирав знову
        yg = lax.all_gather(y, self.axis, tiled=True)
        bl = lax.dynamic_slice_in_dim(jnp.asarray(b, F32), lo, rows)
        yl = yg - lo
        own = (yl >= 0) & (yl < rows)
        lse_l, ly_l = ce.chunk_ce(hg, Wl, bl, jnp.where(own, yl, 0), min(ch, rows))
        ly_l = jnp.where(own, ly_l, 0.0)                                    # не свій рядок — 0 (і градієнт 0)
        if self.ce_a2a:
            # кожен чип отримує лише lse своїх N слів від усіх W чипів: (W·N,) → (W, N) → all_to_all → (W, N);
            # logsumexp по осі чипів — ті самі W чисел у тому самому порядку → побітово як старий шлях.
            W = self.W
            lse_w = lax.all_to_all(lse_l.reshape(W, N), self.axis, 0, 0, tiled=True)
            lse_own = jax.nn.logsumexp(lse_w, 0)
            # логіт слова: одна ненульова частина + нулі → сума точна за будь-якого порядку (psum_scatter = psum).
            ly_own = lax.psum_scatter(ly_l.reshape(W, N), self.axis, scatter_dimension=0, tiled=False)
            return lse_own, ly_own
        lse = jax.nn.logsumexp(lax.all_gather(lse_l, self.axis), 0)       # (W, W·N) → (W·N,)
        ly = lax.psum(ly_l, self.axis)
        return lax.dynamic_slice_in_dim(lse, r * N, N), lax.dynamic_slice_in_dim(ly, r * N, N)


@partial(jax.custom_vjp, nondiff_argnums=(0,))
def _take_ag(vs, Wl, ids):
    """fwd: all_gather таблиці (V, D) + локальний gather ids (вихід ідентичний _take_masked)."""
    full = lax.all_gather(Wl, vs.axis, tiled=True)                         # (V, D)
    return full[jnp.clip(ids, 0, vs.V - 1)]


def _take_ag_fwd(vs, Wl, ids):
    return _take_ag(vs, Wl, ids), (ids, jax.ShapeDtypeStruct(Wl.shape, Wl.dtype))


def _take_ag_bwd(vs, res, g):
    ids, wsd = res
    if vs.take_bwd == "sort":
        return _take_bwd_sorted(vs, ids, g, wsd), None
    # транспозиція СТАРОГО лінійного шляху: all_gather g → маскований scatter-add у свої рядки (побітово як було)
    (dW,) = jax.linear_transpose(lambda w: vs._take_masked(w, ids), wsd)(g)
    return dW, None


def _take_bwd_sorted(vs, ids, g, wsd):
    """bwd take без «рваного» scatter (профіль: 10 і 1 GB/s): all_gather котангенса й номерів (як раніше), потім
    stable argsort локальних номерів (чужі → rows, відкидаються) і segment_sum з indices_are_sorted=True.
    Порядок доданків у кожному рядку — той самий, що в старому scatter (порядок (W, ...) зберігає stable sort);
    накопичення — у dtype градієнта, як і в старому шляху."""
    rows, D = wsd.shape
    gg = lax.all_gather(g, vs.axis).reshape(-1, D)                         # (W·N, D)
    loc = (lax.all_gather(ids, vs.axis) - vs._lo(rows)).reshape(-1)
    loc = jnp.where((loc >= 0) & (loc < rows), loc, rows)                  # чужий рядок → сегмент rows (за межею)
    order = jnp.argsort(loc, stable=True)
    seg = loc[order]
    dW = jax.ops.segment_sum(gg[order].astype(wsd.dtype), seg, num_segments=rows,
                             indices_are_sorted=True, mode="drop")
    return dW


_take_ag.defvjp(_take_ag_fwd, _take_ag_bwd)


@dataclass(frozen=True)
class Dist:
    """Що model.forward/loss_terms бере при W>1 (усередині shard_map)."""
    vocab: VocabShard
    exchange: moe.Exchange
    quantile_hook: callable


def make_dist(cfg, W, axis=AXIS):
    assert cfg.V % W == 0, f"V={cfg.V} має ділитися на W={W}"
    assert cfg.E % W == 0, f"E={cfg.E} має ділитися на W={W}"
    vs = VocabShard(cfg.V, W, axis, take_mode=getattr(cfg, "vocab_take", "ag"),
                    take_bwd=getattr(cfg, "vocab_take_bwd", "sort"),
                    ce_a2a=getattr(cfg, "vocab_ce_a2a", True))
    return Dist(vocab=vs, exchange=moe.a2a_exchange(axis, W, impl=getattr(cfg, "moe_a2a_impl", "a2a")),
                quantile_hook=lambda q: lax.pmean(q, axis))


def situation_local(emb_l, x, W, axis=AXIS):
    """Справжнє середнє вектора слів по батчу всіх чипів (§10): psum(своїх рядків своїх і чужих слів) / (W·B·T).
    emb_l (rows, d) свої рядки, x (B,T) свої слова. Кожне слово всіх чипів враховане рівно раз."""
    xg = lax.all_gather(x, axis, tiled=True)                               # (W·B, T)
    rows = emb_l.shape[0]
    loc = xg - lax.axis_index(axis) * rows
    inside = (loc >= 0) & (loc < rows)
    e = emb_l[jnp.clip(loc, 0, rows - 1)].astype(F32) * inside[..., None]
    return lax.psum(e.sum((0, 1)), axis) / xg.size


# ---------------------------------------------------------------- loss і градієнти

def make_loss_fn(cfg, consts, mesh, params_like, remat=False):
    """→ loss_fn(params, xy, prep, tau) -> (loss, (tau_new, terms, stats, per_chip, situation)) для jax.value_and_grad.
    xy (W·B, T+1) і prep (W·B, ...) розкладаються по чипах P('x'); tau — скаляр (P())."""
    W = mesh.shape[AXIS]
    dist = make_dist(cfg, W)
    pspecs = param_specs(params_like)
    pm = lambda t: jax.tree_util.tree_map(lambda v: lax.pmean(v, AXIS), t)

    def local(params, xy, prep, tau):
        total, (tau_n, terms, stats) = model.train_loss(params, consts, xy, prep, tau, cfg, remat, dist)
        sit = situation_local(lax.stop_gradient(params["emb"]), xy[:, :-1], W)
        return total[None], tau_n, pm(terms), pm(stats), sit

    prep_spec = {k: P(AXIS) for k in ("cand", "on", "bk", "src")}
    sm = _shard_map(local, mesh=mesh, in_specs=(pspecs, P(AXIS), prep_spec, P()),
                    out_specs=(P(AXIS), P(), P(), P(), P()))

    def loss_fn(params, xy, prep, tau):
        per, tau_n, terms, stats, sit = sm(params, xy, prep, tau)
        return per.mean(), (tau_n, terms, stats, per, sit)
    return loss_fn


def make_grad_fn(cfg, consts, mesh, params_like, remat=False):
    """jit(value_and_grad(loss_fn)): (loss, aux), grads — grads у шардуванні параметрів."""
    return jax.jit(jax.value_and_grad(make_loss_fn(cfg, consts, mesh, params_like, remat), has_aux=True))


def make_train_step(cfg, consts, mesh, params_like, ocfg, spec, remat=False):
    """Повний крок (§10): grads (глобальні) → instinct.step зі спільним loss (mean по чипах) і справжньою ситуацією;
    rel/норми — по глобальних тензорах (step поза shard_map). → (params, opt, tau_new, loss, terms, stats)."""
    loss_fn = make_loss_fn(cfg, consts, mesh, params_like, remat)

    @jax.jit
    def train_step(params, opt, xy, prep, tau, lr):
        (loss, (tau_n, terms, stats, _, sit)), grads = jax.value_and_grad(loss_fn, has_aux=True)(
            params, xy, prep, tau)
        params, opt = instinct.step(params, grads, opt, lr=lr, situation=sit, loss=loss, spec=spec, cfg=ocfg)
        return params, opt, tau_n, loss, terms, stats
    return train_step


# ---------------------------------------------------------------- M11: повний крок step.py на сітці (одна jit-програма)

def offload_fns(m_sh, kind="pinned_host"):
    """(to_dev, to_host) для step.make_local_step(offload=...): m_sh — дерево NamedSharding листків opt["m"].
    Усередині jit: device_put на те саме шардування з іншим memory_kind (MaxText-стиль offload оптимізатора)."""
    dev = jax.tree_util.tree_map(lambda s: s.with_memory_kind("device"), m_sh)
    host = jax.tree_util.tree_map(lambda s: s.with_memory_kind(kind), m_sh)
    return (lambda m: jax.device_put(m, dev)), (lambda m: jax.device_put(m, host))


def offload_state_shardings(st_sh, kind="pinned_host"):
    """Шардування стану, де opt["m"] живе на хості (memory_kind=kind), решта — як було."""
    opt = dict(st_sh["opt"], m=jax.tree_util.tree_map(lambda s: s.with_memory_kind(kind), st_sh["opt"]["m"]))
    return dict(st_sh, opt=opt)


def make_step(cfg, consts, mesh, params, spec, icfg, *, remat="none", donate=True, tau=7.5, offload_opt=False):
    """step.make_train_step з loss_fn = shard_map(train_loss) цього модуля: fwd+bwd (shard_map, колективи
    всередині) + Instinct (поза shard_map, глобальні масиви) — одна jit-програма з donate стану.
    Розкладка стану фіксується in/out_shardings: вихідний state має ТІ САМІ шардування, що й вхідний, тож
    крок 2+ бере кеш (без перекомпіляції). → (train_step, state, place_batch).
    state будується тут: params розкладаються за param_specs, opt — instinct.init під jit (шардування
    успадковуються від params), tau/step — репліковані."""
    from . import step as S
    lf = make_loss_fn(cfg, consts, mesh, params, remat)

    def loss_fn(p, xy, prep, t):
        loss, (tau_n, terms, stats, per, sit) = lf(p, xy, prep, t)
        return loss, (tau_n, terms, stats, {"situation": sit, "per_chip": per})

    ps = put(params, mesh, param_specs(params))
    rep = NamedSharding(mesh, P())
    opt = jax.jit(lambda p: instinct.init(p, params["emb"].shape[1], icfg, spec))(ps)
    # листки, що не залежать від params (скаляри, cs-сталі), jit кладе на ОДИН пристрій → реплікуємо на сітку
    opt = jax.tree_util.tree_map(
        lambda x: x if len(x.sharding.device_set) == mesh.size else jax.device_put(x, NamedSharding(mesh, P())), opt)
    # jit без out_shardings НЕ гарантує успадкування: на TPU m експертів/словника виходив реплікованим
    # (10 ГБ на чип замість 1.3). Листок з формою параметра → шардування цього параметра.
    by_shape = {}
    for x in jax.tree_util.tree_leaves(ps):
        by_shape.setdefault((x.shape, x.dtype), x.sharding)
        by_shape.setdefault(x.shape, x.sharding)
    def _fix(x):
        s = by_shape.get((x.shape, x.dtype), by_shape.get(x.shape))
        return jax.device_put(x, s) if s is not None and not x.sharding.is_equivalent_to(s, x.ndim) else x
    opt = jax.tree_util.tree_map(_fix, opt)
    state = dict(params=ps, opt=opt, tau=jax.device_put(jnp.asarray(tau, F32), rep),
                 step=jax.device_put(jnp.zeros((), jnp.int32), rep))
    st_sh = jax.tree_util.tree_map(lambda x: x.sharding, state)
    off = None
    if offload_opt:                                   # --offload_opt: Instinct m (bf16, ≈3.4 GiB/чип) — на хості
        off = offload_fns(st_sh["opt"]["m"])
        st_sh = offload_state_shardings(st_sh)
        state = jax.device_put(state, st_sh)
    row = NamedSharding(mesh, P(AXIS))
    b_sh = dict(xy=row, prep={k: row for k in ("cand", "on", "bk", "src")}, lr=rep)
    lspec = {instinct.path_str(pth): sp for pth, sp in jax.tree_util.tree_flatten_with_path(
        param_specs(params), is_leaf=lambda v: isinstance(v, P))[0]}
    coll = S.Collectives(shard=(mesh, lspec))
    ts = S.make_train_step(cfg, consts, spec, icfg, remat=remat, loss_fn=loss_fn, donate=donate, coll=coll,
                           in_shardings=(st_sh, b_sh), out_shardings=(st_sh, rep), offload=off)

    def place_batch(b):
        return dict(xy=jax.device_put(b["xy"], row), prep={k: jax.device_put(v, row) for k, v in b["prep"].items()},
                    lr=jax.device_put(b["lr"], rep))
    return ts, state, place_batch
