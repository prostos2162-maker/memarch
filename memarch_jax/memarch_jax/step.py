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
from typing import Callable, Optional

import jax
import jax.numpy as jnp

from . import model, instinct

F32 = jnp.float32
_ident = lambda v: v


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

    def local(state, batch):
        params, opt, tau = state["params"], state["opt"], state["tau"]
        xy, prep, lr = batch["xy"], batch["prep"], batch["lr"]
        (loss, aux), grads = jax.value_and_grad(lf, has_aux=True)(params, xy, prep, tau)
        tau_new, terms, stats = aux[:3]
        extra = aux[3] if len(aux) > 3 else {}          # dist: {"situation", "per_chip"} з усередини shard_map
        loss_g = coll.loss_mean(loss)
        sit = extra["situation"] if "situation" in extra else sit_fn(params, xy)
        grads = coll.grad_sync(grads)
        if offload is not None:                         # m: хост → чип лише після bwd (пік fwd/bwd без m)
            opt = dict(opt, m=offload[0](opt["m"]))
        params, opt = instinct.step(params, grads, opt, lr=lr, situation=sit, loss=loss_g, spec=spec, cfg=icfg,
                                    axis_name=coll.axis_name, rows_total=coll.rows_total,
                                    shard=coll.shard)
        if offload is not None:
            opt = dict(opt, m=offload[1](opt["m"]))
        metrics = dict(loss=loss_g, loss_local=loss, terms=terms, stats=stats, tau=tau_new,
                       k=opt["k"], s=opt["s"], stake=opt["stake"])
        if "per_chip" in extra:
            metrics["per_chip"] = extra["per_chip"]
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
    if out_shardings is not None:
        kw["out_shardings"] = out_shardings
    return jax.jit(f, donate_argnums=(0,) if donate else (), **kw)


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
