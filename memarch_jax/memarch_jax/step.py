"""M10: повний крок тренування в ОДНІЙ jit-функції: value_and_grad(train_loss) → situation → Instinct prepare →
(синхронізація градієнтів) → Instinct update → finish. Порядок як у torch step.py (prepare за loss і ситуацією
ДО оновлення ваг; ситуація — за старою таблицею emb).

Стан — dict: {"params", "opt", "tau" (f32 скаляр), "step" (int32)}; batch — dict: {"xy" (B,T+1) int32,
"prep" (induct_raw з хоста), "lr" (f32 скаляр, рахується на хості через instinct.lr_at)}.

Інтерфейс для dist.py (W>1, shard_map) — без знання внутрішностей:
  local = make_local_step(..., coll=Collectives(axis_name="x", loss_mean=..., situation=..., grad_sync=...,
                          quantile_hook=...), loss_fn=...)
  train_step = make_train_step(..., wrap=lambda f: shard_map(f, mesh, in_specs, out_specs))
wrap обгортає local(state, batch) → (state, metrics) до jit; donate — на state (params+opt+tau+step).
loss_fn(params, xy, prep, tau) → (loss, (tau_new, terms, stats)) — за замовчуванням model.train_loss з remat."""
import dataclasses
import functools
from typing import Callable, Optional

import jax
import jax.numpy as jnp

from . import model, instinct

F32 = jnp.float32
_ident = lambda v: v
import os as _os
RB_DST = _os.environ.get("RB_DST", "1") != "0"
RB_PRE = _os.environ.get("RB_PRE", "1") != "0"   # 15.11: рядки фішок до кроку — до запису (без копії стеку)   # 15.11: відкатник — вибір hit у Pallas-ядрах (0 — старий jnp.where)


@dataclasses.dataclass(frozen=True)
class Collectives:
    """Хуки багаточиповості (W=1 — тотожні). design §10: одна синхронна модель.
    loss_mean(loss) — pmean втрати для Instinct.prepare; situation(params, xy) — справжнє середнє по батчу всіх
    чипів (None — локальне середнє emb[x]); grad_sync(grads) — psum/pmean градієнтів; спільний tau (pmean квантиля)
    — через власний loss_fn (anchors.quantile_hook); axis_name/rows_total — у instinct.update (hsS, норми)."""
    axis_name: Optional[str] = None
    rows_total: Optional[int] = None
    loss_mean: Callable = _ident
    situation: Optional[Callable] = None
    grad_sync: Callable = _ident
    shard: Optional[tuple] = None        # (mesh, {leaf: PartitionSpec}) — Pallas-ядра Instinct через shard_map (W>1)


def situation_full(params, xy):
    """full_situation_xy: середній вектор слів батчу (fp32) за нешардованою таблицею emb."""
    x = xy[:, :-1]
    return params["emb"][x].astype(F32).mean((0, 1))


def default_loss_fn(cfg, consts, remat="none"):
    """loss_fn(params, xy, prep, tau) → (loss, (tau_new, terms, stats))."""
    def loss_fn(params, xy, prep, tau):
        return model.train_loss(params, consts, xy, prep, tau, cfg, remat=remat)
    return loss_fn


def make_local_step(cfg, consts, spec, icfg, *, remat="none", loss_fn=None, coll=Collectives(), offload=None):
    """Чиста функція кроку (без jit): (state, batch) → (state, metrics).
    offload: None | (to_dev, to_host) — функції над деревом opt["m"]: стан Instinct m живе в пам'яті хоста
    (pinned_host), на чип приходить лише на час оновлення і повертається назад. Математика та сама."""
    lf = loss_fn if loss_fn is not None else default_loss_fn(cfg, consts, remat)
    sit_fn = coll.situation or situation_full

    def _rb_finish(params, opt, params_in, m_in, opt_in, hit, rep, shape_ok, ids_prev, ids_now, kf, xy, prep,
                   my_src, same, kc, lk, loss_g, done=frozenset(), pre=None):
        """Відкатник А після оновлення: hit → ствол = rb_p (jnp.where, in-place з donate), фішки — scatter лише рядків
        ids_now (назад до стану до цього кроку) і ids_prev (назад до rb_p); m так само; решта opt — стан до кроку.
        Далі новий знімок: ваги/m ДО цього кроку (фішки — рядки ids_now) і батч цього кроку."""
        if not shape_ok:
            # 15.11 латка: крок іншої форми (синтетика) сам відкату не робить і батч не зберігає, але робить знімок
            # ваг/m ДО себе — тоді наступний текстовий крок може відкотити оновлення синтетики (варіант Б, без повтору).
            lp_in0 = {instinct.path_str(p): x for p, x in jax.tree_util.tree_flatten_with_path(params_in)[0]}
            return params, dict(opt, rb_rep=opt_in["rb_rep"], rb_valid=jnp.ones((), bool),
                                rb_src=jnp.where(opt_in["rb_rep"], opt_in["rb_src"], jnp.asarray(my_src, jnp.int32)),
                                rb_p={k: instinct.rb_cut(k, x, icfg, ids_now) for k, x in lp_in0.items()},
                                rb_m={k: instinct.rb_cut(k, v, icfg, ids_now) for k, v in m_in.items()},
                                rb_ids=ids_now.reshape(opt_in["rb_ids"].shape) if kf > 0 else ids_prev)

        pre = pre or {}

        def back(key, new, cur_in, saved, pre_rows=None):
            if key in done:                              # ядро вже вибрало (hit → знімок) і писало в буфер знімка
                return new
            if not instinct.rb_is_fish(key, new, icfg):
                return jnp.where(hit, saved.astype(new.dtype), new)
            rin = instinct.rb_rows(cur_in, ids_now) if pre_rows is None else pre_rows
            r_now = jnp.where(hit, rin.astype(new.dtype), instinct.rb_rows(new, ids_now))
            v = instinct.rb_set_rows(new, ids_now, r_now)
            r_prev = jnp.where(hit, saved.astype(new.dtype), instinct.rb_rows(v, ids_prev))
            return instinct.rb_set_rows(v, ids_prev, r_prev)
        lp_in = {instinct.path_str(p): x for p, x in jax.tree_util.tree_flatten_with_path(params_in)[0]}
        skip = ("m", "rb_p", "rb_m", "rb_b", "rb_ids", "rb_lema", "rb_n", "rb_rep", "rb_src", "rb_valid", "rb_cool")
        # 15.11: саме jnp.where поверх оновлення (не lax.cond): XLA зливає вибір з фузіями оновлення і накладає
        # оновлення на хвіст bwd/зв'язок. lax.cond навколо всього оновлення (версія 14:22–15:36) коштував +8…19 мс/крок
        # на v6e-8 (main5_1115_1422/1536: 557–569 мс), тоді як ця форма — 549–551 мс (main5_1115_1313 ≈ 1517 без відкатника).
        lv, tdef = jax.tree_util.tree_flatten_with_path(params)
        params_out = jax.tree_util.tree_unflatten(tdef, [
            back(instinct.path_str(p), x, lp_in[instinct.path_str(p)], opt_in["rb_p"][instinct.path_str(p)],
                 pre.get("p", {}).get(instinct.path_str(p)))
            for p, x in lv])
        m_out = {k: back(k, v, m_in[k], opt_in["rb_m"][k], pre.get("m", {}).get(k)) for k, v in opt["m"].items()}
        small = {k: jax.tree_util.tree_map(lambda a, b: jnp.where(hit, a, b), opt_in[k], v)
                 for k, v in opt.items() if k not in skip and k in opt_in}
        pend_other = opt_in["rb_rep"] & (~same)          # повтор чекає крок свого виду — батч не чіпаємо
        rep_new = hit & same                             # 15.11: відкат після кроку іншого виду — без повтору (Б)
        keep_b = rep_new | pend_other
        rbb = opt_in["rb_b"]
        new_b = dict(xy=jnp.where(keep_b, rbb["xy"], xy.astype(rbb["xy"].dtype)),
                     prep={k: (jnp.where(keep_b, v, prep[k].astype(v.dtype)) if k in prep else v)
                           for k, v in rbb["prep"].items()})
        newl = jnp.where(lk > 0, icfg.rb_beta * lk + (1 - icfg.rb_beta) * jnp.minimum(loss_g, lk * icfg.rb_ratio),
                         loss_g)
        snap = ({k: (pre["p"][k] if k in pre.get("p", {}) else instinct.rb_cut(k, x, icfg, ids_now)) for k, x in lp_in.items()},
                {k: (pre["m"][k] if k in pre.get("m", {}) else instinct.rb_cut(k, v, icfg, ids_now)) for k, v in m_in.items()})
        opt = dict(opt, **small, m=m_out, rb_p=snap[0], rb_m=snap[1],
                   rb_ids=ids_now.reshape(opt_in["rb_ids"].shape) if kf > 0 else ids_prev,
                   rb_b=new_b, rb_rep=rep_new | pend_other,
                   rb_src=jnp.where(pend_other, opt_in["rb_src"], jnp.asarray(my_src, jnp.int32)),
                   rb_valid=~pend_other,
                   rb_lema=opt_in["rb_lema"].at[kc].set(newl), rb_n=opt_in["rb_n"] + hit.astype(jnp.int32))
        if "rb_cool" in opt_in:
            # вид kc: відхилений (hit) → наступний його крок без відкату; повтор чужого батча (rep) стану виду не міняє
            opt["rb_cool"] = jnp.where(rep, opt_in["rb_cool"], opt_in["rb_cool"].at[kc].set(hit))
        return params_out, opt

    def local(state, batch):
        params, opt, tau = state["params"], state["opt"], state["tau"]
        xy, prep, lr = batch["xy"], batch["prep"], batch["lr"]
        rb_on = bool(getattr(icfg, "rollback", False)) and "rb_p" in opt and "rb_b" in opt
        if rb_on:
            rbb = opt["rb_b"]
            rb_shape_ok = (rbb["xy"].shape == xy.shape and all(
                k in rbb["prep"] and rbb["prep"][k].shape == v.shape for k, v in prep.items()))
            if not rb_shape_ok:                         # інша форма батча (напр. фінальна фаза фішок) — без відкату
                print(f"[step] відкатник: форма батча ≠ збереженої — на цьому кроці вимкнено "
                      f"({ {k: v.shape for k, v in prep.items()} })", flush=True)
            my_src = 1 if getattr(cfg, "remind_only", False) else 0
            same = opt["rb_src"] == my_src
            rb_rep = opt["rb_rep"] & same & rb_shape_ok     # цей крок — повтор батча, що спричинив стрибок
            rb_ok = opt["rb_valid"] & (~(opt["rb_rep"] & same)) & rb_shape_ok  # повтор сам відкату не запускає; після кроку іншого виду — відкат без повтору
            if rb_shape_ok:
                _sel = lambda st, inc: jnp.where(rb_rep, st.astype(inc.dtype), inc)
                xy = _sel(rbb["xy"], xy)
                prep = {k: _sel(rbb["prep"][k], v) for k, v in prep.items()}
            lr = jnp.where(rb_rep, lr * icfg.rb_lr, lr)
        if getattr(cfg, "norm_dump_proj", False) and "U" in opt:   # fix14 F1: напрям «звалища» з попереднього кроку
            prep = dict(prep, stream_u=opt["U"])
        (loss, aux), grads = jax.value_and_grad(lf, has_aux=True)(params, xy, prep, tau)
        tau_new, terms, stats = aux[:3]
        extra = aux[3] if len(aux) > 3 else {}          # dist: {"situation", "per_chip"} з усередини shard_map
        loss_g = coll.loss_mean(loss)
        sit = extra["situation"] if "situation" in extra else sit_fn(params, xy)
        grads = coll.grad_sync(grads)
        if offload is not None:                         # m: хост → чип лише після bwd (пік fwd/bwd без m)
            opt = dict(opt, m=offload[0](opt["m"]))
        opt_pre = opt                                   # відкатник: стан Instinct ДО цього кроку
        if (getattr(icfg, "stream_proj", False) or getattr(icfg, "m_rate", False)) and "stream_m" in stats:
            # K1 (13.11): див. instinct.stream_project; fix15 без K1 — лише оновлення U (project=False)
            grads, opt = instinct.stream_project(grads, opt, stats["stream_m"],
                                                 update_u=not getattr(cfg, "remind_only", False),
                                                 project=getattr(icfg, "stream_proj", False))
        card = None
        if icfg.card_by_kind:
            if getattr(cfg, "remind_only", False):
                card = jnp.asarray(icfg.cards - 1, jnp.int32)
            elif "kind" in prep:
                card = prep["kind"].reshape(-1)[0].astype(jnp.int32)
        rb_hit = jnp.zeros((), bool)
        if rb_on:
            # 15.11 запобіжник-відкатник, варіант А (Стефан): стрибок loss → ваги/m назад на стан до попереднього
            # кроку, оновлення цього кроку не застосовується, наступний крок — повтор батча попереднього кроку ×rb_lr.
            # Без другого градієнта і без lax.cond: вибір jnp.where (ствол) / scatter 2 рядків (фішки) поверх оновлення.
            kc = card if card is not None else jnp.asarray(0, jnp.int32)
            lk = opt["rb_lema"][kc]
            rb_hit = (rb_ok & (lk > 0) & (loss_g > lk * icfg.rb_ratio) & (state["step"] >= icfg.rb_warm))
            if "rb_cool" in opt and not _os.environ.get("RB_NOCOOL"):
                # 15.11 (fable52f): відкатник не відхиляє той самий вид двічі поспіль. Було: вид, чия втрата піднялась
                # понад rb_ratio× звичної, відхилявся на КОЖНОМУ своєму кроці (його градієнт не застосовувався, а
                # відкочувався невинний попередній крок іншого виду) → вид не вчився саме тоді, коли зламаний
                # (стенд tr5: uk 8.0 → 11.0 → 12.2 з відкатником проти 7.2 → 5.9 → 7.5 без нього).
                rb_hit = rb_hit & (~opt["rb_cool"][kc])
            kf = int(getattr(icfg, "rb_fish_k", 0))
            ids_prev = opt["rb_ids"]
            if kf > 0 and "fish" in prep:
                ids_now = prep["fish"].reshape(-1, prep["fish"].shape[-1])[0][:kf].astype(jnp.int32)
            else:
                ids_now = ids_prev
            params_in, m_in, opt_in = params, opt_pre["m"], opt_pre
            # 15.11 швидкість: рядки ids_now фішок ДО кроку (для відкату й нового знімка) — читаються тут, а бар'єр
            # ставить запис оновлення на місці (лише активні рядки, instinct.fish_fast_ok) ПІСЛЯ цих читань; інакше
            # XLA копіює весь стек фішок (NF, ., .), бо вхідний буфер ще читається після запису
            rb_pre = {}
            if kf > 0 and RB_PRE:
                lpf = {instinct.path_str(p_): x for p_, x in jax.tree_util.tree_flatten_with_path(params)[0]}
                fk = [k for k, x in lpf.items() if instinct.rb_is_fish(k, x, icfg)]
                mk = [k for k in fk if k in opt["m"] and instinct.rb_is_fish(k, opt["m"][k], icfg)]
                if fk:
                    pre = ({k: instinct.rb_rows(lpf[k], ids_now) for k in fk},
                           {k: instinct.rb_rows(opt["m"][k], ids_now) for k in mk})
                    pre, tp, tm = jax.lax.optimization_barrier((pre, {k: lpf[k] for k in fk}, {k: opt["m"][k] for k in mk}))
                    rb_pre = dict(p=pre[0], m=pre[1])
                    lpf.update(tp)
                    lv0, td0 = jax.tree_util.tree_flatten_with_path(params)
                    params = jax.tree_util.tree_unflatten(td0, [lpf[instinct.path_str(p_)] for p_, _ in lv0])
                    opt = dict(opt, m=dict(opt["m"], **tm))
        p_before = params
        card = None
        if icfg.card_by_kind:                           # 13.11: картка Instinct = вид кроку; синтетика — остання картка
            if getattr(cfg, "remind_only", False):
                card = jnp.asarray(icfg.cards - 1, jnp.int32)
            elif "kind" in prep:
                card = prep["kind"].reshape(-1)[0].astype(jnp.int32)
        rb_dst = None
        if rb_on and rb_shape_ok and RB_DST and not getattr(icfg, "norm_anchor", False):
            # 15.11 швидкість відкатника: Pallas-ядра ствола пишуть нові m/w одразу в буфер знімка rb_p/rb_m з вибором
            # hit у ядрі → вхідні ваги/m (новий знімок) без копії перед ядром на місці і без окремого jnp.where.
            rb_dst = dict(p=opt["rb_p"], m=opt["rb_m"], hit=rb_hit.astype(jnp.float32), done=set())

        # 15.11 швидкість: активні фішки кроку — ті самі номери, що бере model.forward (prep["fish"], рядок 0)
        fish_ids = None
        if (getattr(cfg, "fish_n", 0) > 0 and isinstance(prep, dict) and "fish" in prep
                and not getattr(cfg, "fish_eval", False)):
            fish_ids = prep["fish"].reshape(-1, prep["fish"].shape[-1])[0].astype(jnp.int32)

        def _upd(params, opt):
            params, opt = instinct.step(params, grads, opt, lr=lr, situation=sit, loss=loss_g, spec=spec, cfg=icfg,
                                        axis_name=coll.axis_name, rows_total=coll.rows_total,
                                        shard=coll.shard, card=card, rb_dst=rb_dst, fish_ids=fish_ids)
            if getattr(icfg, "norm_anchor", False):        # K2 (13.11): 1D norm-ваги → 1
                params = instinct.norm_anchor(params, lr, icfg)
            return params, opt
        mrl_on = getattr(icfg, "m_rate", False) and "mrl_bank" in opt
        params, opt = _upd(params, opt)
        mrl_info = None
        if mrl_on:                                      # fix15: швидкість виходу m з відкладенням
            params, opt, mrl_info = instinct.m_rate(p_before, params, opt, icfg, grads=grads)
        if rb_on:
            params, opt = _rb_finish(params, opt, params_in, m_in, opt_in, rb_hit, rb_rep, rb_shape_ok, ids_prev,
                                     ids_now, kf, xy, prep, my_src, same, kc, lk, loss_g,
                                     done=frozenset(rb_dst["done"]) if rb_dst is not None else frozenset(), pre=rb_pre)
            tau_new = jax.tree_util.tree_map(lambda a, b: jnp.where(rb_hit, a, b), tau, tau_new)
        if offload is not None:
            opt = dict(opt, m=offload[1](opt["m"]))
        metrics = dict(loss=loss_g, loss_local=loss, terms=terms, stats=stats, tau=tau_new,
                       k=opt["k"], s=opt["s"], stake=opt["stake"])
        if "per_chip" in extra:
            metrics["per_chip"] = extra["per_chip"]
        if "snr_f" in opt:                              # --opt_snr: множник впевненості кроку на кожен тензор (P,)
            metrics["snr_f"] = opt["snr_f"]
        if "df" in opt:                                 # fix21 (--opt_damp): множник гасіння сплеску на кожен тензор (P,)
            metrics["df"] = opt["df"]
        if mrl_info is not None:
            metrics.update(mrl_info)
        if "rb_n" in opt:                               # 15.11 відкатник: чи спрацював на цьому кроці і скільки разів усього
            metrics["rb_hit"] = rb_hit.astype(jnp.float32)
            metrics["rb_rep"] = rb_rep.astype(jnp.float32) if rb_on else jnp.zeros((), F32)
            metrics["rb_lr"] = jnp.asarray(lr, F32)     # фактичний lr кроку (повтор — ×rb_lr)
            metrics["rb_n"] = opt["rb_n"]
        metrics["g"] = opt["g"]                         # 13.11: підсилювачі картки цього кроку (P,) — для журналу
        metrics["rel"] = opt["rel"]                     # |крок|/|w| по тензорах (P,; оновлюється раз на rel_every)
        return dict(params=params, opt=opt, tau=tau_new, step=state["step"] + 1), metrics

    return local


def make_train_step(cfg, consts, spec, icfg, *, remat="none", loss_fn=None, coll=Collectives(), wrap=None,
                    donate=True, jit=True, in_shardings=None, out_shardings=None, offload=None):
    """train_step(state, batch) → (state, metrics). Одна jit-функція; donate_argnums=(0,) — params/opt/tau/step
    перезаписуються на місці (після виклику старий state недійсний). wrap — напр. shard_map з dist.py.
    in/out_shardings — для W>1 (dist.make_step): фіксують розкладку стану, щоб крок 2+ не перекомпільовувався."""
    f = make_local_step(cfg, consts, spec, icfg, remat=remat, loss_fn=loss_fn, coll=coll, offload=offload)
    if wrap is not None:
        f = wrap(f)
    if not jit:
        return f
    kw = {}
    if in_shardings is not None:
        kw["in_shardings"] = in_shardings
    rb = bool(getattr(icfg, "rollback", False))
    # фішки з копією лише k рядків (rb_p має іншу форму, ніж params) — без перестановки: свій буфер на місці
    _cut = int(getattr(icfg, "rb_fish_n", 0)) > 0 and int(getattr(icfg, "rb_fish_k", 0)) > 0
    swap = functools.partial(rb_swap, keep=(lambda k: str(k).startswith("fish.")) if _cut else None)
    if out_shardings is not None:
        kw["out_shardings"] = (swap(out_shardings[0]), out_shardings[1]) if rb else out_shardings
    if not rb:
        return jax.jit(f, donate_argnums=(0,) if donate else (), **kw)
    # 15.11 відкатник А: новий знімок rb_p/rb_m = ваги/m ДО кроку (той самий буфер), нові ваги/m — у буфер старого
    # знімка. jax зіставляє donate-входи з виходами за порядком; jit віддає стан з переставленими params↔rb_p, m↔rb_m,
    # тож кожен вихід лягає в «чужий» вхідний буфер без копії; обгортка переставляє назад (лише python-дерева).
    jf = jax.jit(lambda st, b: (lambda o: (swap(o[0]), o[1]))(f(st, b)), donate_argnums=(0,) if donate else (), **kw)
    return _RbStep(jf, swap)


class _RbStep:
    """train_step для відкатника: jit віддає переставлений стан (rb_swap), тут — назад. lower/_cache_size — від jit."""

    def __init__(self, jf, swap=None):
        self.jf = jf
        self.swap = swap or rb_swap

    def __call__(self, state, batch):
        st, met = self.jf(state, batch)
        return self.swap(st), met

    def lower(self, *a, **k):
        return self.jf.lower(*a, **k)

    def _cache_size(self):
        return self.jf._cache_size()


def rb_swap(st, keep=None):
    """Інволюція: params (дерево) ↔ opt.rb_p (плоский {path_str}), opt.m ↔ opt.rb_m. Працює й на деревах шардувань.
    keep(k) → True: листок k не переставляється (фішки з копією лише k рядків — інша форма)."""
    if not isinstance(st, dict) or "opt" not in st or not isinstance(st["opt"], dict) or "rb_p" not in st["opt"]:
        return st
    opt = st["opt"]
    kp = keep or (lambda k: False)
    lv, tdef = jax.tree_util.tree_flatten_with_path(st["params"], is_leaf=lambda x: x is None)
    keys = [instinct.path_str(p) for p, _ in lv]
    new_params = jax.tree_util.tree_unflatten(tdef, [x if kp(k) else opt["rb_p"][k] for k, (_, x) in zip(keys, lv)])
    new_rbp = {k: (opt["rb_p"][k] if kp(k) else x) for k, (_, x) in zip(keys, lv)}
    m, rbm = opt["m"], opt["rb_m"]
    new_m = {k: (m[k] if kp(k) else rbm[k]) for k in rbm}
    new_rbm = {k: (rbm[k] if kp(k) else m[k]) for k in m}
    return dict(st, params=new_params, opt=dict(opt, m=new_m, rb_m=new_rbm, rb_p=new_rbp))


def init_state(params, spec, icfg, tau=7.5, d_sit=None):
    """Початковий стан: Instinct з d_sit = d (ширина emb)."""
    d_sit = d_sit or params["emb"].shape[1]
    return dict(params=params, opt=instinct.init(params, d_sit, icfg, spec), tau=jnp.asarray(tau, F32),
                step=jnp.zeros((), jnp.int32))


def make_batch(xy, cfg, consts, lr):
    """Хостовий батч: xy numpy (B,T+1) → dict з prep (induct_raw на CPU) і lr (f32)."""
    import numpy as np
    xy = np.asarray(xy)
    return dict(xy=jnp.asarray(xy, jnp.int32), prep=model.make_prep(xy[:, :-1], cfg, consts),
                lr=jnp.asarray(lr, F32))
