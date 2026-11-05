"""Швидкий тренувальний цикл: етапи конвеєра (ARCHITECTURE.md, «Тренувальний конвеєр»).

Етапи вмикаються літерами в --stage:
  base  — як lm_bench: дані з CPU, зупинка на хост щокроку, звичайний AdamW.
  A     — дані на GPU, жодних зупинок на хост щокроку, torch.compile кроку.
  B     — оновлення ваг прямо в зворотному проході: кожен параметр оновлюється,
          щойно готовий його градієнт (суворо: лише на останньому мікро-батчі).
  C     — перемежування мікро-батчів (див. Trainer.step_interleaved).
  G     — крок записаний графом (CUDA graphs); лише для моделей зі сталою формою.

python fastloop.py speed --models T300 S300_20 --stages base A AB
python fastloop.py train --models T300 S300_20 --stage AB --steps 4000 --lr 6e-4
"""
import argparse
import json
import math
import os
import sys
import time

import numpy as np

# кеш компіляції — на диску D:, а не в тимчасовій папці на C: (Стефан: не забивати C:)
_CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".cache")
os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", os.path.join(_CACHE, "inductor"))
os.environ.setdefault("TRITON_CACHE_DIR", os.path.join(_CACHE, "triton"))
import dev  # noqa: E402  DEVICE=cuda|cpu|xla — до lm/full2 (на XLA вмикає NO_COMPILE і ATTN_IMPL=block)
import torch
torch._dynamo.config.cache_size_limit = 64   # крок пам'яті й перевірка на інших довжинах — різні форми; інакше після 8 перекомпіляцій крок без компіляції
import torch.nn.functional as F
from torch.utils.flop_counter import FlopCounterMode

from lm import build
from lm_bench import VOCAB, prepare

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


class GPUData:
    """Усі токени лежать на відеокарті; батч збирається там само, без участі хоста."""

    def __init__(self, arr, seed):
        self.t = torch.from_numpy(np.asarray(arr, dtype=np.int32)).to(dev.device())
        self.g = dev.generator(seed)

    def batch(self, B, T):
        start = torch.randint(0, len(self.t) - T - 1, (B,), device=dev.device(), generator=self.g)
        return self.t[start[:, None] + torch.arange(T + 1, device=dev.device())].long()


def lr_at(step, args):
    warm = int(os.environ.get("WARMUP", 200))
    up = min(1, (step + 1) / warm)
    if os.environ.get("LR_SCHED") == "wsd":
        # WSD: пік до останніх 20% кроків, далі лінійно до 0.1 (зручно продовжувати з чекпойнта)
        d0 = (1 - float(os.environ.get("WSD_FRAC", 0.2))) * args.steps
        fl = float(os.environ.get("LR_FLOOR", 0.1))   # куди спускається lr у кінці (0 — «глибокий сон»)
        return args.lr * up * (1.0 if step < d0 else 1 - (1 - fl) * min(1, (step - d0) / (args.steps - d0)))
    return args.lr * up * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(1, step / args.steps))))


class _MuonCombo:
    """Muon (матриці всередині шарів) + AdamW (таблиці слів, норми): спільний lr; Muon — з lr ×mul (match_rms_adamw)."""

    def __init__(self, params, kw, mul):
        mat = [p for p in params if p.dim() == 2 and p.shape[0] < 10000]
        rest = [p for p in params if not (p.dim() == 2 and p.shape[0] < 10000)]
        self.mul = mul
        self.muon = torch.optim.Muon(mat, lr=kw["lr"] * mul, weight_decay=kw["weight_decay"], adjust_lr_fn="match_rms_adamw")
        self.adam = torch.optim.AdamW(rest, fused=True, **kw)
        self.param_groups = [dict(lr=kw["lr"])]

    def step(self):
        lr = self.param_groups[0]["lr"]
        self.muon.param_groups[0]["lr"], self.adam.param_groups[0]["lr"] = lr * self.mul, lr
        self.muon.step()
        self.adam.step()

    def zero_grad(self, set_to_none=True):
        self.muon.zero_grad(set_to_none=set_to_none)
        self.adam.zero_grad(set_to_none=set_to_none)


class _CpuPrep:
    """Копія фіксованих таблиць моделі на процесорі — лише для prep (усе, що залежить від слів, а не від ваг).
    Рахується в завантажувачі наперед: пристрій отримує готові номери і ніколи не чекає (TPU: пошуки, сортування,
    scatter на прискорювачі — дорогі, на процесорі — дрібниця)."""

    def __init__(self, m):
        import types
        from full2 import FullLM2
        for n, b in m.named_buffers(recurse=False):
            setattr(self, n, b.detach().cpu())
        for n in ("phr_V", "phr_hmask", "vocab"):
            if hasattr(m, n):
                setattr(self, n, getattr(m, n))
        for n in ("phr", "phr4", "ind_a", "lock_w", "sp_lse"):
            if hasattr(m, n) and not hasattr(self, n):
                setattr(self, n, True)                 # позначка «є в моделі» (prep перевіряє hasattr)
        for f in ("prep", "_phr_idx", "_phr4_idx", "_induct_raw", "_phr_surp"):
            setattr(self, f, types.MethodType(getattr(FullLM2, f), self))


PREP_PACK = os.environ.get("PREP_PACK") == "1"      # prep — однією передачею на тип даних (TPU: менше проміжків)


class CpuFeeder:
    """Фоновий потік: токени батча + prep кожного мікро-батча на процесорі, на depth батчів уперед."""

    def __init__(self, model, sample, B, T, micro, depth=3):
        import queue
        import threading
        self.cp, self.sample, self.B, self.T, self.micro = _CpuPrep(model), sample, B, T, micro
        self.q = queue.Queue(maxsize=depth)
        threading.Thread(target=self._run, daemon=True).start()

    def _run(self):
        torch.set_num_threads(max(1, (os.cpu_count() or 4) // 2))
        try:
            while True:
                xy = self.sample(self.B, self.T)
                with torch.no_grad():
                    preps = [self.cp.prep(mb[:, :-1]) for mb in xy.split(self.micro)]
                self.q.put((xy, self._pack(preps) if PREP_PACK else preps))
        except BaseException as e:                # помилка потоку — у головний (інакше крок вічно чекає черги)
            self.q.put(e)
            raise

    def get(self):
        item = self.q.get()
        if isinstance(item, BaseException):
            raise RuntimeError("CpuFeeder: підготовка на процесорі впала") from item
        xy, preps = item
        d = dev.device()
        if PREP_PACK:
            return xy.to(d, non_blocking=True), self._unpack(preps, d)
        return (xy.to(d, non_blocking=True),
                [{k: v.to(d, non_blocking=True) for k, v in p.items()} for p in preps])

    @staticmethod
    def _pack(preps):
        """Усі тензори prep — в один плаский буфер на тип (одна передача хост→пристрій замість десятка)."""
        groups, meta = {}, []
        for i, p in enumerate(preps):
            for k, v in p.items():
                g = groups.setdefault(v.dtype, [])
                off = sum(t.numel() for t in g)
                g.append(v.reshape(-1))
                meta.append((i, k, v.dtype, off, tuple(v.shape)))
        return {dt: torch.cat(g) for dt, g in groups.items()}, tuple(meta), len(preps)

    def _unpack(self, packed, d):
        flat, meta, n = packed
        flat_d = {dt: t.to(d, non_blocking=True) for dt, t in flat.items()}
        fn = getattr(self, "_unpack_fn", None)
        if fn is None:
            def split(fl, meta=meta):                 # суцільні копії (вигляд зі зсувом TPU-програма не приймає)
                return [fl[dt][off:off + math.prod(sh)].view(sh).clone() for _, _, dt, off, sh in meta]
            fn = self._unpack_fn = torch.compile(split, dynamic=False) if dev.TPU else split
        outs = fn(flat_d)
        preps = [{} for _ in range(n)]
        for (i, k, _, _, _), t in zip(meta, outs):
            preps[i][k] = t
        return preps


class _Ckpt(torch.nn.Module):
    """Обгортка шару: прямий прохід без збереження проміжних тензорів (перерахунок на зворотному)."""

    def __init__(self, m):
        super().__init__()
        self.m = m

    def __getattr__(self, name):                # атрибути шару (moe, aux, cur_sel…) — як у самого шару
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self._modules["m"], name)

    def forward(self, *a, **k):
        from torch.utils.checkpoint import checkpoint
        if not self.training:
            return self.m(*a, **k)
        if os.environ.get("ACT_CKPT") == "sel":
            # вибіркова: виходи множень і уваги зберігаються, перераховується лише дешеве (норми, активації, RoPE)
            return checkpoint(self.m, *a, use_reentrant=False, context_fn=_sel_ctx, **k)
        return checkpoint(self.m, *a, use_reentrant=False, **k)


def _sel_ctx():
    from torch.utils.checkpoint import CheckpointPolicy, create_selective_checkpoint_contexts
    aten = torch.ops.aten
    keep = {aten.mm.default, aten.bmm.default, aten.addmm.default, aten._grouped_mm.default,
            aten._scaled_dot_product_flash_attention.default, aten._scaled_dot_product_efficient_attention.default,
            aten._scaled_dot_product_cudnn_attention.default}
    def policy(ctx, op, *args, **kwargs):
        return CheckpointPolicy.MUST_SAVE if op in keep else CheckpointPolicy.PREFER_RECOMPUTE
    return create_selective_checkpoint_contexts(policy)


def dist_world():
    import torch.distributed as tdist
    return tdist.get_world_size() if tdist.is_initialized() else 1


def dist_init():
    """Кілька процесів (один на чип): група процесів «tpu» на TPU, nccl на GPU, gloo на процесорі."""
    import torch.distributed as tdist
    if tdist.is_initialized() or int(os.environ.get("WORLD_SIZE", 1)) == 1:
        return
    tdist.init_process_group("tpu" if dev.TPU else "nccl" if dev.CUDA else "gloo")
    if dev.CUDA:
        torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", 0)))


class Trainer:
    def __init__(self, name, stage, args):
        torch.manual_seed(0)
        self.stage, self.args = stage, args
        self.model = build(name, VOCAB).to(dev.device())
        m = self.model
        if os.environ.get("BF16_W") == "1" and "X" in stage:
            # ваги лінійних шарів у bf16 (точна fp32-копія — в оптимізаторі, як для експертів): без перетворення
            # fp32→bf16 на кожному мікро-батчі й з удвічі легшими градієнтами; таблиця слів і норми — fp32
            for nm, mod in m.named_modules():           # пам'ять рахується у fp32 навмисно (поза autocast)
                if isinstance(mod, torch.nn.Linear) and not nm.startswith(("memory", "reread")):
                    mod.weight = torch.nn.Parameter(mod.weight.detach().bfloat16())
        if os.environ.get("BF16_ALL") == "1":
            # Стефан 03.11: УСІ ваги в bf16 (таблиці слів, норми, пам'ять, перечитування, скаляри) — без fp32-копій;
            # Instinct пише оновлення стохастичним округленням (exp_opt._sr_bf16), щоб малі кроки не губились
            with torch.no_grad():
                for p in m.parameters():
                    if p.is_floating_point() and p.dtype != torch.bfloat16:
                        p.data = p.data.bfloat16()

        def loss_fn(xy, prep=None):
            x_in, y_t = xy[:, :-1], xy[:, 1:]
            if dev.TPU:                               # вигляди зі зсувом через межу графа TPU-програма не приймає (E0102)
                x_in, y_t = x_in.contiguous(), y_t.contiguous()
            out, cost = m(x_in, prep) if prep is not None else m(x_in)
            if hasattr(m, "train_loss"):
                loss = m.train_loss(out, y_t, True)
            else:
                loss = F.cross_entropy(out.float().view(-1, VOCAB), y_t.reshape(-1))
            return loss + args.read_cost * cost.float()

        # NO_COMPILE=1 — для локальної перевірки на Windows, де немає Triton
        compile_ok = "A" in stage and not os.environ.get("NO_COMPILE")
        # G: крок записується графом (CUDA graphs) — потрібні сталі розміри (якорі cap / quota)
        mode = "reduce-overhead" if "G" in stage else None
        if os.environ.get("ACT_CKPT"):
            # чекпойнт активацій по шарах: проміжні результати шару перераховуються на зворотному проході
            # (порядок як у torch_tpu: чекпойнт усередині шару, компіляція — зовні)
            for ch in m.children():
                if isinstance(ch, torch.nn.ModuleList):
                    for i in range(len(ch)):
                        ch[i] = _Ckpt(ch[i])
        if compile_ok and os.environ.get("COMPILE_LAYERS"):
            # TPU (як у прикладах torch_tpu): кожен шар — окремо, сталі форми; решта (вихід, маршрутизація) — eager
            # компіляція — ПІСЛЯ FSDP/EP (нижче) і на місці (module.compile): скомпільований шар бачить уже порізані ваги
            self._compile_layers = True
            self.loss_fn = loss_fn
        else:
            # COMPILE_BACKEND=aot_eager — перевірка трасування (FSDP/DTensor під компіляцією) на процесорі без C++
            bk = os.environ.get("COMPILE_BACKEND")
            self.loss_fn = (torch.compile(loss_fn, mode=mode, **({"backend": bk} if bk else {})) if compile_ok else loss_fn)

        if os.environ.get("FSDP") == "1":
            # FSDP2: кожен шар (і вся модель) — ваги порізані між чипами, збираються на час шару.
            # Оновлення — після зворотного проходу (OPT_ALL): FSDP сам кладе .grad після reduce-scatter
            from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard
            from torch.distributed.device_mesh import init_device_mesh
            mesh = init_device_mesh(dev.TYPE if dev.TPU or dev.CUDA else "cpu", (dist_world(),))
            # ваги зберігаються порізаними у fp32, на час шару збираються в bf16 (те, що поза FSDP дають
            # BF16_W / BF16_EXPERTS); градієнти ділить між чипами у fp32
            for p in m.parameters():
                if p.dtype == torch.bfloat16:
                    p.data = p.data.float()
            mp = MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.float32)
            # скаляри FSDP2 не ріже — вони в кожного свої, градієнт усереднюється вручну
            self._fsdp_ignored = [p for p in m.parameters() if p.dim() == 0]
            ign = set(self._fsdp_ignored)
            if os.environ.get("EP") == "1":
                # шардинг експертів: кожен чип — свої E/W експертів (не ріжуться FSDP і не усереднюються між чипами)
                import torch.distributed as tdist
                for mod in m.modules():
                    if hasattr(mod, "ep_shard"):
                        mod.ep_shard(tdist.get_rank(), dist_world())
                ign |= {p for p in m.parameters() if getattr(p, "_ep_local", False)}
            sub = lambda mod: {p for p in mod.parameters() if p in ign}
            from torch.distributed.fsdp import register_fsdp_forward_method
            for nm in ("memory", "reread"):              # пам'ять рахується у fp32 навмисно — без змішаної точності
                if hasattr(m, nm):
                    mod = getattr(m, nm)
                    fully_shard(mod, mesh=mesh, ignored_params=sub(mod))
                    for meth in ("read", "write", "_contents", "init_state"):   # викликаються повз forward
                        if hasattr(mod, meth):
                            register_fsdp_forward_method(mod, meth)
            for ch in m.children():
                if isinstance(ch, torch.nn.ModuleList):
                    for blk in ch:
                        fully_shard(blk, mesh=mesh, mp_policy=mp, ignored_params=sub(blk))
            fully_shard(m, mesh=mesh, mp_policy=mp, ignored_params=ign)
            if hasattr(m, "train_loss"):                 # вихід і похибка рахуються поза forward — теж зі зібраними вагами
                register_fsdp_forward_method(m, "train_loss")
            os.environ["OPT_ALL"] = "1"
        if getattr(self, "_compile_layers", False):
            for ch in m.children():
                if isinstance(ch, torch.nn.ModuleList):
                    for blk in ch:
                        blk.compile(dynamic=False)
        params = [p for p in m.parameters() if p.requires_grad]
        kw = dict(lr=args.lr, betas=(0.9, 0.95), weight_decay=0.1)
        if "B" in stage and "X" in stage:
            # Б+X: інстинкт без зупинок на хост, кожна вага оновлюється в хуку, щойно готовий її градієнт
            from exp_opt import InstinctOptB
            import json
            self.opt = InstinctOptB(params, **kw, **json.loads(os.environ.get("OPT_KW", "{}")))
            self.apply_now = False

            # TPU: p.grad не можна переносити між скомпільованими викликами (E0102) — свої сталі буфери накопичення
            self.acc = {} if (dev.TPU or os.environ.get("ACC_BUF")) and args.micro < args.batch else None

            def hook_x(p):
                if p.grad is None:
                    return
                if self.acc is not None:
                    a = self.acc.get(p)
                    if a is None:
                        a = self.acc[p] = torch.zeros_like(p.grad)
                    if not self.apply_now:
                        a.add_(p.grad)
                        p.grad = None
                        return
                    p.grad.add_(a)
                    a.zero_()
                if self.apply_now and not self.opt_all:
                    self.opt.update(p)
                    p.grad = None
            # OPT_ALL=1: оновлення не в хуках, а після зворотного проходу — усі ваги одним скомпільованим кроком
            self.opt_all = os.environ.get("OPT_ALL") == "1"
            if os.environ.get("FSDP") == "1":              # FSDP накопичує й ділить градієнти сам — без хуків
                self.acc = None
            else:
                for p in params:
                    p.register_post_accumulate_grad_hook(hook_x)
            self._hook = hook_x
        elif "B" in stage:
            # кожен параметр має свій оптимізатор і оновлюється в хуку, щойно градієнт готовий
            self.opts = {p: torch.optim.AdamW([p], fused=True, **kw) for p in params}
            self.apply_now = False

            def hook(p):
                if self.apply_now:
                    fr = getattr(p, "_frozen_rows", None)    # навчені експерти (хвилі): ваги не рухаються
                    keep = p.detach().clone() if fr is not None else None
                    self.opts[p].step()
                    self.opts[p].zero_grad(set_to_none=True)
                    if keep is not None:
                        with torch.no_grad():
                            p.copy_(torch.where(fr.view(-1, *[1] * (p.dim() - 1)), keep, p))
            for p in params:
                p.register_post_accumulate_grad_hook(hook)
            self._hook, self._kw = hook, kw
        elif "X" in stage:
            # X: оптимізатор з нуля з картками досвіду (exp_opt.py); потребує loss на хості щокроку
            from exp_opt import ExperienceOpt, InstinctOpt
            import json
            okw = json.loads(os.environ.get("OPT_KW", "{}"))                # варіанти для розбору
            cls = InstinctOpt if okw.pop("kind", None) == "instinct" else ExperienceOpt
            self.opt = cls(params, **kw, **okw)
        elif "S" in stage:
            # S: оптимізатор за несподіванкою (surprise_opt.py); потребує loss на хості щокроку
            from surprise_opt import SurpriseAdamW
            self.opt = SurpriseAdamW(params, **kw)
        elif os.environ.get("OPT") == "muon":
            # лише для заміру в гонці: Muon на прихованих матрицях, AdamW на таблицях слів і нормах
            self.opt = _MuonCombo(params, kw, float(os.environ.get("MUON_MUL", 1)))
        else:
            self.opt = torch.optim.AdamW(params, fused="A" in stage and dev.CUDA, foreach=not dev.CUDA or None, **kw)
        self.params = params

    def set_lr(self, step):
        lr = lr_at(step, self.args)
        bx = "B" in self.stage and "X" not in self.stage
        groups = [o.param_groups[0] for o in self.opts.values()] if bx else self.opt.param_groups
        for g in groups:
            g["lr"] = lr

    def step(self, xy, step, preps=None):
        """Один крок тренування. Нічого не повертає на хост. preps — готове наперед (лише від слів) на мікро-батч."""
        self.set_lr(step)
        if hasattr(self.model, "before_step"):
            self.model.before_step(step)
        mbs = xy.split(self.args.micro)
        if dev.TPU and len(mbs) > 1:         # шматок-вигляд зі зсувом у пам'яті скомпільована програма TPU не приймає (E0102)
            mbs = [m.clone() for m in mbs]
        if os.environ.get("SPARSE_ROWS") and "X" in self.stage and hasattr(self.model, "table_rows"):
            self.opt.rows = self.model.table_rows(xy[:, :-1])
        if "C" in self.stage:
            return self.step_interleaved(mbs, len(xy))
        total = torch.zeros((), device=dev.device())
        for i, mb in enumerate(mbs):
            if "B" in self.stage:
                self.apply_now = i == len(mbs) - 1     # оновлюємо ваги лише на останньому мікро-батчі
            if "G" in self.stage:
                torch.compiler.cudagraph_mark_step_begin()   # межа запуску графа
            with torch.autocast(dev.TYPE, torch.bfloat16):
                loss = self.loss_fn(mb, preps[i]) if preps is not None else self.loss_fn(mb)
            if "G" in self.stage:
                loss = loss.clone()                          # вихід графа перезапишеться наступним запуском
            if "B" in self.stage and "X" in self.stage and self.apply_now:
                # loss кроку вже відомий (прямий прохід позаду), а жодна вага ще не оновлена
                with torch.no_grad():
                    sit = self.model.emb(xy[:, :-1]).float().mean((0, 1))
                self.opt.prepare(total + loss.detach() * len(mb) / len(xy), sit)
            (loss * len(mb) / len(xy)).backward()
            total += loss.detach() * len(mb) / len(xy)
        if "B" in self.stage and "X" in self.stage:
            if getattr(self, "_fsdp_ignored", None):
                import torch.distributed as tdist
                for p in self._fsdp_ignored:
                    if p.grad is not None:
                        tdist.all_reduce(p.grad, op=tdist.ReduceOp.AVG)
            if self.opt_all:
                self.opt.update_all()
            self.opt.finish()
        if "B" in self.stage and hasattr(self.model, "rebank"):
            self._rebank(step)
        if "B" not in self.stage:
            # (у B глобальне обрізання градієнта неможливе: ваги оновлюються до того, як пораховано всі градієнти)
            torch.nn.utils.clip_grad_norm_(self.params, 1.0)
            if "X" in self.stage:          # ситуація батчу — за тим, що модель читає (ще до помилки)
                with torch.no_grad():
                    sit = self.model.emb(xy[:, :-1]).float().mean((0, 1))
                self.opt.step(total, sit)
            elif "S" in self.stage:
                self.opt.step(total)
            else:
                self.opt.step()
            self.opt.zero_grad(set_to_none=True)
        return total

    def _rebank(self, step):
        """Хвилі з банками: експерти переїхали між «вчаться» і «заморожені» — новий оптимізатор для нового
        параметра, інерція (exp_avg, exp_avg_sq) тих, хто лишився вчитись, переноситься по місцях."""
        self._apply_rebank(self.model.rebank(step))

    def _apply_rebank(self, changes):
        """Експерти переїхали між банками: оптимізатор переходить на нові тензори (інерція — по місцях)."""
        for old, new, slot_map in changes:
            if "X" in self.stage:                      # інстинкт: та сама роль тензора, інерція по місцях
                self.opt.swap(old, new, slot_map)
                new.register_post_accumulate_grad_hook(self._hook)
                continue
            opt_old = self.opts.pop(old, None)
            opt = torch.optim.AdamW([new], fused=True, **self._kw)
            opt.param_groups[0]["lr"] = opt_old.param_groups[0]["lr"] if opt_old else self._kw["lr"]
            st_old = opt_old.state.get(old) if opt_old else None
            if st_old and new.numel():
                st = {"step": st_old["step"].clone(),
                      "exp_avg": torch.zeros_like(new), "exp_avg_sq": torch.zeros_like(new)}
                for j, s in enumerate(slot_map):
                    if s is not None:
                        st["exp_avg"][j] = st_old["exp_avg"][s]
                        st["exp_avg_sq"][j] = st_old["exp_avg_sq"][s]
                opt.state[new] = st
            self.opts[new] = opt
            new.register_post_accumulate_grad_hook(self._hook)
        self.params = [p for p in self.model.parameters() if p.requires_grad]

    def step_interleaved(self, mbs, n):
        """C: мікро-батчі незалежні до оновлення ваг, тож кожен іде на своєму потоці
        відеокарти: поки важка частина одного думає над якорями, легка частина іншого
        вже читає. Зворотний прохід — один на всі; autograd сам розводить його по тих
        самих потоках."""
        if not hasattr(self, "streams"):
            self.streams = [torch.cuda.Stream() for _ in mbs]
        cur = torch.cuda.current_stream()
        losses = []
        for s, mb in zip(self.streams, mbs):
            s.wait_stream(cur)
            mb.record_stream(s)
            with torch.cuda.stream(s), torch.autocast("cuda", torch.bfloat16):
                losses.append(self.loss_fn(mb) * len(mb) / n)
        for s in self.streams:
            cur.wait_stream(s)
        total = torch.stack(losses).sum()
        if "B" in self.stage:
            self.apply_now = True
        total.backward()
        if "B" not in self.stage:
            torch.nn.utils.clip_grad_norm_(self.params, 1.0)
            self.opt.step()
            self.opt.zero_grad(set_to_none=True)
        return total.detach()


@torch.no_grad()
def val_loss(model, val, seq, n=20):
    from lm_bench import step_loss
    model.eval()
    # ті самі вікна щоразу; на XLA/CPU — генератор процесора (свого генератора XLA не має, а спільний посів зсунув би тренування)
    g = torch.Generator(device="cuda" if dev.CUDA else "cpu").manual_seed(0)
    t = val.t
    total = torch.zeros((), device=dev.device())
    for _ in range(n):
        start = torch.randint(0, len(t) - seq - 1, (8,), device=g.device, generator=g).to(dev.device())
        xy = t[start[:, None] + torch.arange(seq + 1, device=dev.device())].long()
        with torch.autocast(dev.TYPE, torch.bfloat16):
            total += step_loss(model, xy, train=False)[0]
        dev.step_end()
    model.train()
    return (total / n).item()


def matmul_peak():
    """Практична стеля заліза: великі bf16-множення (PEAK_TFLOPS — задати вручну, напр. паспорт чипа TPU)."""
    if os.environ.get("PEAK_TFLOPS"):
        return float(os.environ["PEAK_TFLOPS"]) * 1e12
    a = torch.randn(8192, 8192, device=dev.device(), dtype=torch.bfloat16)
    b = torch.randn(8192, 8192, device=dev.device(), dtype=torch.bfloat16)
    for _ in range(3):
        c = a @ b
        dev.step_end()
    dev.sync()
    t = time.time()
    for _ in range(20):
        c = a @ b
        dev.step_end()
    dev.sync()
    return 2 * 8192 ** 3 * 20 / (time.time() - t)


def step_flops(name, args):
    """Обчислення одного кроку (прямий + зворотний прохід), порахувані на звичайній моделі."""
    torch.manual_seed(0)
    import lm as lm_mod                                # лічильник рахує незлиту модель (обчислень стільки ж; злиті кернели
    fused_was, lm_mod.FUSED = lm_mod.FUSED, ""          # лічильник не розуміє) — MFU порівнянне між режимами
    try:
        return _step_flops(name, args)
    finally:
        lm_mod.FUSED = fused_was


def _step_flops(name, args):
    torch.manual_seed(0)
    # корисні обчислення: звичайна (не упакована — лічильник не знає FlexAttention) версія,
    # по одному тексту, тож доповнення порожнечею немає; результат множимо на батч
    if os.environ.get("STEP_FLOPS"):                   # задати вручну (напр. пораховане на GPU для тієї самої моделі)
        return float(os.environ["STEP_FLOPS"])
    m = build(name.replace("S300p_", "S300_"), VOCAB).to(dev.device())
    xy = torch.randint(0, int(os.environ.get("PHR_V", 50257)), (args.micro, args.seq + 1), device=dev.device())
    if hasattr(m, "tau"):
        with torch.no_grad(), torch.autocast(dev.TYPE, torch.bfloat16):
            for _ in range(60):                        # поріг підлаштовується до реальної частки якорів
                m(xy[:, :-1])
                dev.step_end()
    xy = xy[:1]

    def gmm_flops(a, b, *args, out_shape=None, **kw):
        # grouped_mm (лічильник отримує форми): (M,K)@(E,K,N) — x@w; (K,M)@(M,N) — градієнт ваг. Рахуються всі M
        # рядків (у FullLM2 всі місця якорів — справжні, «нікуди» лише з порожніх, яких тут немає)
        if len(b) == 3:
            return 2 * a[0] * a[-1] * b[-1]
        return 2 * a[0] * a[1] * b[-1]
    with FlopCounterMode(display=False, custom_mapping={torch.ops.aten._grouped_mm: gmm_flops}) as fc:
        with torch.autocast(dev.TYPE, torch.bfloat16):
            out, _ = m(xy[:, :-1])
            loss = m.train_loss(out, xy[:, 1:], True) if hasattr(m, "train_loss") else \
                F.cross_entropy(out.float().view(-1, VOCAB), xy[:, 1:].reshape(-1))
        loss.backward()
    dev.step_end()
    del m
    dev.empty_cache()
    return fc.get_total_flops() * args.batch


def speed(args):
    peak = matmul_peak()
    print(f"стеля множень: {peak / 1e12:.0f} TFLOP/s", flush=True)
    data = GPUData(np.random.randint(0, int(os.environ.get("PHR_V", 50257)), 20_000_000, dtype=np.int32), seed=1)
    rows = []
    for name in args.models:
        flops = step_flops(name, args)
        for stage in args.stages:
            dev.reset_peak()
            tr = Trainer(name, stage, args)
            frz = float(os.environ.get("SPEED_FROZEN", "0"))
            if frz and "B" in stage and hasattr(tr.model, "rebank"):
                # замір усталеного режиму: частка експертів уже в замороженому банку (як після ~2750 кроків тренування)
                with torch.no_grad():
                    for b in tr.model.heavy:
                        b.learned[:int(b.E * frz)] = True
                tr._rebank(1000)
            cpu = np.random.default_rng(1)
            arr = data.t.cpu().numpy()

            def get():
                if "A" in stage:
                    return data.batch(args.batch, args.seq)
                idx = cpu.integers(0, len(arr) - args.seq - 1, args.batch)   # як у lm_bench
                return torch.from_numpy(np.stack([arr[i:i + args.seq + 1] for i in idx]).astype(np.int64)).to(dev.device())

            feeder = None
            if os.environ.get("CPU_PREP", "1" if dev.TPU else "") and hasattr(tr.model, "prep"):
                rng = np.random.default_rng(1)          # prep (усе від слів) — процесор, наперед; пристрій не чекає

                def sample(B, T):
                    idx = rng.integers(0, len(arr) - T - 1, B)
                    return torch.from_numpy(np.stack([arr[i:i + T + 1] for i in idx]).astype(np.int64))
                feeder = CpuFeeder(tr.model, sample, args.batch, args.seq, args.micro)

            nxt = [None]

            def go(s):
                if feeder is None:
                    return tr.step(get(), s)
                xy, preps = nxt[0] if nxt[0] is not None else feeder.get()
                loss = tr.step(xy, s, preps)
                nxt[0] = feeder.get()                  # перенесення наступного батча — поки пристрій рахує цей
                return loss

            for s in range(args.warmup):
                tw = time.time()
                go(s)
                dev.step_end()
                if stage == "base" or os.environ.get("WARMUP_LOG"):
                    dev.sync()
                if os.environ.get("WARMUP_LOG"):          # чи це довга компіляція, чи зависання (у stderr)
                    print(f"розігрів {s}: {time.time() - tw:.1f} с", file=sys.stderr, flush=True)
            dev.sync()
            t = time.time()
            for s in range(args.warmup, args.warmup + args.n):
                loss = go(s)
                dev.step_end()
                if stage == "base":                     # base зупиняється на хост щокроку, як lm_bench
                    loss.item()
            dev.sync()
            dt = (time.time() - t) / args.n
            stats = {k: float(v) for k, v in getattr(tr.model, "stats", {}).items()}
            row = dict(model=name, stage=stage, ms=dt * 1000, tflops=flops / dt / 1e12, mfu=flops / dt / peak,
                       mem=dev.max_mem_gb(), loss=float(loss), **stats)
            rows.append(row)
            print(f"SPEED {name:8s} {stage:5s} {dt * 1000:7.1f} ms  {flops / dt / 1e12:6.1f} TFLOP/s  "
                  f"MFU {flops / dt / peak:5.1%}  mem {row['mem']:4.0f}GB  loss {float(loss):.3f}  {stats}", flush=True)
            del tr
            dev.empty_cache()
    os.makedirs("results", exist_ok=True)
    with open("results/speed.json", "w") as f:
        json.dump(rows, f, indent=1)


def _opt_state(opt):
    """Стан будь-якого нашого оптимізатора: тензори, списки тензорів і числа (без посилань на параметри)."""
    skip = {"plist", "idx", "param_groups", "rows"}
    out = {}
    for k, v in vars(opt).items():
        if k in skip:
            continue
        if torch.is_tensor(v):
            out[k] = v.detach().cpu().clone()
        elif isinstance(v, list) and all(t is None or torch.is_tensor(t) for t in v):
            out[k] = [None if t is None else t.detach().cpu().clone() for t in v]
        elif v is None or isinstance(v, (int, float, bool)):
            out[k] = v
    out["_lr"] = [g["lr"] for g in getattr(opt, "param_groups", [])]
    return out


def _opt_load(opt, st):
    for k, v in st.items():
        if k == "_lr":
            for g, lr in zip(getattr(opt, "param_groups", []), v):
                g["lr"] = lr
            continue
        cur = getattr(opt, k, None)
        if torch.is_tensor(v):
            if torch.is_tensor(cur) and cur.shape == v.shape:
                cur.copy_(v)
            else:
                setattr(opt, k, v.to(dev.device()))
        elif isinstance(v, list):
            if isinstance(cur, list) and len(cur) == len(v):
                for a, b in zip(cur, v):
                    if torch.is_tensor(a) and b is not None:
                        a.copy_(b)
            else:
                setattr(opt, k, [None if b is None else b.to(dev.device()) for b in v])
        else:
            setattr(opt, k, v)


class CorpusData:
    """Батчі з корпусу (data/loader.py): CORPUS='{"джерело": частка, ...}', CORPUS_ROOT (тека або gs://)."""

    def __init__(self, seq, seed=0, rank=0, world=1):
        import importlib.util
        spec = importlib.util.spec_from_file_location("loader", os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                                             "data", "loader.py"))
        L = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(L)
        self.c = L.Corpus(json.loads(os.environ["CORPUS"]), os.environ["CORPUS_ROOT"], seq=seq, seed=seed,
                          rank=rank, world=world, cache=os.environ.get("CORPUS_CACHE", "/tmp/corpus_cache"))
        self.seq = seq

    def batch(self, B, T):
        assert T == self.seq, "корпус видає вікна сталої довжини"
        return torch.from_numpy(self.c.batch(B)).to(dev.device(), non_blocking=True)


def _ckpt_paths():
    d = os.environ.get("CKPT_DIR")
    if not d:
        return None, None
    local = d if not d.startswith("gs://") else os.path.join("/tmp", "ckpt")
    os.makedirs(local, exist_ok=True)
    return d, local


def _ckpt_save(tr, step, log, data):
    import subprocess
    import threading
    d, local = _ckpt_paths()
    if not d:
        return
    st = {"step": step, "log": log,
          "banks": [list(b.Fz) for b in getattr(tr.model, "heavy", []) if hasattr(b, "Fz")], "model": {k: v.detach().cpu() for k, v in tr.model.state_dict().items()},
          "opt": _opt_state(tr.opt) if hasattr(tr, "opt") else None,
          "corpus": data.c.state_dict() if isinstance(data, CorpusData) else None}
    tmp = os.path.join(local, "latest.pt.tmp")
    torch.save(st, tmp)
    os.replace(tmp, os.path.join(local, "latest.pt"))
    if d.startswith("gs://"):                    # заливка в бакет — у фоні, тренування не чекає
        threading.Thread(target=lambda: subprocess.run(["gcloud", "storage", "cp", os.path.join(local, "latest.pt"),
                                                        d.rstrip("/") + "/latest.pt"], capture_output=True),
                         daemon=True).start()


def _ckpt_load(tr, data):
    """Продовжити з останнього чекпойнта, якщо він є. Повертає (наступний крок, журнал)."""
    import subprocess
    d, local = _ckpt_paths()
    if not d:
        return 1, []
    path = os.path.join(local, "latest.pt")
    if d.startswith("gs://") and not os.path.exists(path):
        subprocess.run(["gcloud", "storage", "cp", d.rstrip("/") + "/latest.pt", path], capture_output=True)
    if not os.path.exists(path):
        return 1, []
    st = torch.load(path, map_location="cpu", weights_only=False)
    banks = [b for b in getattr(tr.model, "heavy", []) if hasattr(b, "Fz")]
    changes = []
    for b, fz in zip(banks, st.get("banks", [])):  # та сама розкладка банків, що при збереженні — до завантаження ваг
        if fz:
            changes += b.rebank([e in fz for e in range(b.E)])
    if changes:
        tr._apply_rebank(changes)
    tr.model.load_state_dict(st["model"])
    if st.get("opt") is not None and hasattr(tr, "opt"):
        _opt_load(tr.opt, st["opt"])
    if st.get("corpus") is not None and isinstance(data, CorpusData):
        data.c.load_state_dict(st["corpus"])
    print(f"продовжую з кроку {st['step'] + 1} (чекпойнт {path})", flush=True)
    return st["step"] + 1, st.get("log", [])


def train(args):
    if os.environ.get("CORPUS"):                 # справжній корпус: шматки з масками, частки джерел
        data = CorpusData(args.seq, seed=int(os.environ.get("CORPUS_SEED", 0)),
                          rank=int(os.environ.get("RANK", 0)), world=int(os.environ.get("WORLD", 1)))
        if os.environ.get("VAL_NPY"):            # відкладені вікна, зібрані наперед з окремих шматків
            val = GPUData(np.load(os.environ["VAL_NPY"]), 0)
        else:                                    # інакше — окремий посів того самого корпусу (перетин малий, для нагляду)
            vc = CorpusData(args.seq, seed=10**6)
            val = GPUData(vc.c.batch(160).reshape(-1), 0)
            del vc
    else:
        dp = os.environ.get("DATA")              # DATA — інші одиниці (словосполучення); оцінка — той самий початок тексту
        if not dp:
            prepare("fineweb.npy", args.tokens)
        allt = np.load(dp or "fineweb.npy", mmap_mode="r")
        n_val = int(len(allt) * 2_000_000 / 40_000_000) if dp else 2_000_000
        val, data = GPUData(allt[:n_val], 0), GPUData(allt[n_val:], 1)
    bpu = float(os.environ.get("BYTES_PER_UNIT", 4.6361))   # байт тексту на одиницю (біти на байт — чесна мірка)
    for name in args.models:
        tr = Trainer(name, args.stage, args)
        log, train_time = [], 0.0
        # SEQ_SCHED="256:0.5" — «немовля бачить коротко»: перші 50% кроків шматки по 256 (токенів на крок стільки ж)
        # AUTO_SCHED="0.3:0.8" — автоматизація навички: перші 30% кроків важкі шари думають над 80% слів
        ss, au = os.environ.get("SEQ_SCHED"), os.environ.get("AUTO_SCHED")
        if au or os.environ.get("PANIC"):   # частка якорів / сила паніки міняються: дробові числа — сталі графа
            torch._dynamo.config.specialize_float = True
        s_short, s_frac = (int(ss.split(":")[0]), float(ss.split(":")[1])) if ss else (args.seq, 0.0)
        micro0 = args.micro

        def cfg(step):
            seq = s_short if step <= s_frac * args.steps else args.seq
            fr = cp = None
            if au and hasattr(tr.model, "frac"):
                a_end, a_frac = float(au.split(":")[0]), float(au.split(":")[1])
                fr, cp = (a_frac, min(1.0, a_frac * 1.25)) if step <= a_end * args.steps else (0.2, 0.25)
            return seq, fr, cp

        def apply(c):
            seq, fr, cp = c
            tr.args.micro = micro0 * (args.seq // seq)
            if fr is not None and tr.model.frac != fr:
                tr.model.frac, tr.model.cap = fr, cp
                for b in tr.model.heavy:
                    b.fill = fr / cp
            return seq

        if os.environ.get("WARM_COMPILE"):
            # як у суперника: компіляція всіх форм кроку — до таймера, потім ваги й стан оптимізатора повертаються
            import copy
            sd = copy.deepcopy(tr.model.state_dict())
            ost = {k: (v.clone() if torch.is_tensor(v) else [t.clone() if torch.is_tensor(t) else t for t in v]
                       if isinstance(v, list) else copy.copy(v)) for k, v in vars(tr.opt).items()
                   if v is None or torch.is_tensor(v) or isinstance(v, list) or isinstance(v, (int, float))} if hasattr(tr, "opt") else {}
            for c in sorted(set(cfg(st) for st in range(1, args.steps + 1)), key=str):
                seq = apply(c)
                for _ in range(2):
                    tr.step(data.batch(args.batch * (args.seq // seq), seq), 0)
                    dev.step_end()
            val_loss(tr.model, val, args.seq)
            with torch.no_grad():
                tr.model.load_state_dict(sd)
                for k, v in ost.items():
                    cur = getattr(tr.opt, k)
                    if torch.is_tensor(v) and torch.is_tensor(cur):
                        cur.copy_(v)
                    elif isinstance(v, list):
                        for a, b in zip(cur, v):
                            if torch.is_tensor(a):
                                a.copy_(b)
                    else:
                        setattr(tr.opt, k, v)
            if hasattr(data, "g"):
                data.g = dev.generator(1)
        # EMA_START=0.6 — «сон закріплює»: з 60% кроків — ковзне середнє ваг, оцінка на ньому
        ema_start = float(os.environ.get("EMA_START", 0) or 0)
        ema_beta = float(os.environ.get("EMA_BETA", 0.98))
        ema = None
        dev.sync()
        t0 = time.time()
        # PREFETCH=1 — «вода вже набрана»: наступний батч і все, що залежить лише від слів, — на окремому потоці
        pf = os.environ.get("PREFETCH") and hasattr(tr.model, "prep") and dev.CUDA   # бічний потік — лише CUDA
        side = torch.cuda.Stream() if pf else None

        def fetch(st):
            sq = cfg(st)[0]
            b = data.batch(args.batch * (args.seq // sq), sq)
            side.wait_stream(torch.cuda.current_stream())  # спершу батч має бути записаний основним потоком
            with torch.cuda.stream(side):
                pr = [tr.model.prep(mb[:, :-1]) for mb in b.split(micro0 * (args.seq // sq))]
                ev = torch.cuda.Event(); ev.record(side)
            return b, pr, ev

        start, log0 = _ckpt_load(tr, data)
        log += log0
        ck_every = int(os.environ.get("CKPT_EVERY", 0))
        nxt = fetch(start) if pf else None
        for step in range(start, args.steps + 1):
            seq = apply(cfg(step))
            pn = os.environ.get("PANIC")                       # "0.15:1.0" — останні 15% кроків, сила 1
            if pn:
                pf, pw = float(pn.split(":")[0]), float(pn.split(":")[1])
                want = pw if step > (1 - pf) * args.steps else 0
                if getattr(tr.model, "panic", 0) != want:
                    tr.model.panic = want
            if pf:
                xy_b, preps, ev = nxt
                torch.cuda.current_stream().wait_event(ev)
                for pr_ in preps:                          # тензори з бічного потоку живуть і в основному
                    for v in pr_.values():
                        v.record_stream(torch.cuda.current_stream())
                if step < args.steps:
                    nxt = fetch(step + 1)                  # наперед: поки цей крок рахує, готується наступний
                loss = tr.step(xy_b, step - 1, preps)
            else:
                loss = tr.step(data.batch(args.batch * (args.seq // seq), seq), step - 1)
            dev.step_end()
            ema_every = int(os.environ.get("EMA_EVERY", 1))     # сон періодичний: раз на N кроків, сила β^N
            if ema_start and step >= ema_start * args.steps and step % ema_every == 0:
                with torch.no_grad():
                    ps = [p for p in tr.model.parameters() if p.numel() < 10_000_000]   # без великих таблиць слів
                    if ema is None:
                        ema = [p.detach().float().clone() for p in ps]
                    else:
                        torch._foreach_lerp_(ema, [p.detach().float() for p in ps], 1 - ema_beta ** ema_every)
            if step % args.eval_every == 0 or step == args.steps:
                dev.sync()                              # єдина зупинка — на оцінці
                train_time += time.time() - t0
                if ema is not None:                       # оцінка на «виспаних» вагах, потім назад
                    with torch.no_grad():
                        ps = [p for p in tr.model.parameters() if p.numel() < 10_000_000]
                        cur = [p.detach().clone() for p in ps]
                        for p, e in zip(ps, ema):
                            p.copy_(e)
                    vl_raw = None
                vl = val_loss(tr.model, val, args.seq)
                if ema is not None:
                    with torch.no_grad():
                        for p, c in zip(ps, cur):
                            p.copy_(c)
                    del cur
                extra = "".join(f"  {k} {float(v):.2f}" for k, v in getattr(tr.model, "stats", {}).items())
                log.append({"step": step, "time": train_time, "val": vl})
                print(f"{name}[{args.stage}]  крок {step:5d}  val {vl:.3f}  bpb {vl / bpu / math.log(2):.4f}  train {float(loss):.3f}{extra}  "
                      f"{train_time / 60:.1f} хв", flush=True)
                dev.sync()
                t0 = time.time()
            if ck_every and (step % ck_every == 0 or step == args.steps):
                _ckpt_save(tr, step, log, data)
        if os.environ.get("DUMP_TOK"):                  # похибка кожного слова на тих самих вікнах, що й val_loss
            m = tr.model
            m.eval()
            g = torch.Generator(device="cuda").manual_seed(0)
            ces = []
            with torch.no_grad(), torch.autocast("cuda", torch.bfloat16):
                for _ in range(20):
                    start = torch.randint(0, len(val.t) - args.seq - 1, (8,), device="cuda", generator=g)
                    xy = val.t[start[:, None] + torch.arange(args.seq + 1, device="cuda")].long()
                    out, _ = m(xy[:, :-1])
                    ces.append(m._ce_tok(out[0], xy[:, 1:]).float() if hasattr(m, "_ce_tok") else
                               F.cross_entropy(out.float().view(-1, VOCAB), xy[:, 1:].reshape(-1), reduction="none"))
            torch.save(torch.cat(ces).cpu(), os.environ["DUMP_TOK"])
            m.train()
        os.makedirs("results", exist_ok=True)
        with open(f"results/{name}_{args.stage}.json", "w") as f:
            json.dump({"log": log}, f)
        del tr
        dev.empty_cache()


def main():
    dist_init()                                   # кілька процесів (FSDP) — якщо задано WORLD_SIZE > 1
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=["speed", "train"])
    p.add_argument("--models", nargs="+", default=["T300", "S300_20"])
    p.add_argument("--stages", nargs="+", default=["base", "A", "AB"])
    p.add_argument("--stage", default="AB")
    p.add_argument("--steps", type=int, default=4000)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--micro", type=int, default=16)
    p.add_argument("--seq", type=int, default=1024)
    p.add_argument("--lr", type=float, default=6e-4)
    p.add_argument("--read_cost", type=float, default=0.01)
    p.add_argument("--eval_every", type=int, default=250)
    p.add_argument("--tokens", type=int, default=140_000_000)
    p.add_argument("--warmup", type=int, default=15)
    p.add_argument("--n", type=int, default=30)
    args = p.parse_args()
    speed(args) if args.mode == "speed" else train(args)


if __name__ == "__main__":
    main()
