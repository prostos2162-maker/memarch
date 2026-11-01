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
import time

import numpy as np

# кеш компіляції — на диску D:, а не в тимчасовій папці на C: (Стефан: не забивати C:)
_CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".cache")
os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", os.path.join(_CACHE, "inductor"))
os.environ.setdefault("TRITON_CACHE_DIR", os.path.join(_CACHE, "triton"))
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
        self.t = torch.from_numpy(np.asarray(arr, dtype=np.int32)).cuda()
        self.g = torch.Generator(device="cuda").manual_seed(seed)

    def batch(self, B, T):
        start = torch.randint(0, len(self.t) - T - 1, (B,), device="cuda", generator=self.g)
        return self.t[start[:, None] + torch.arange(T + 1, device="cuda")].long()


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


class Trainer:
    def __init__(self, name, stage, args):
        torch.manual_seed(0)
        self.stage, self.args = stage, args
        self.model = build(name, VOCAB).cuda()
        m = self.model

        def loss_fn(xy, prep=None):
            out, cost = m(xy[:, :-1], prep) if prep is not None else m(xy[:, :-1])
            if hasattr(m, "train_loss"):
                loss = m.train_loss(out, xy[:, 1:], True)
            else:
                loss = F.cross_entropy(out.float().view(-1, VOCAB), xy[:, 1:].reshape(-1))
            return loss + args.read_cost * cost.float()

        # NO_COMPILE=1 — для локальної перевірки на Windows, де немає Triton
        compile_ok = "A" in stage and not os.environ.get("NO_COMPILE")
        # G: крок записується графом (CUDA graphs) — потрібні сталі розміри (якорі cap / quota)
        mode = "reduce-overhead" if "G" in stage else None
        self.loss_fn = torch.compile(loss_fn, mode=mode) if compile_ok else loss_fn

        params = [p for p in m.parameters() if p.requires_grad]
        kw = dict(lr=args.lr, betas=(0.9, 0.95), weight_decay=0.1)
        if "B" in stage and "X" in stage:
            # Б+X: інстинкт без зупинок на хост, кожна вага оновлюється в хуку, щойно готовий її градієнт
            from exp_opt import InstinctOptB
            import json
            self.opt = InstinctOptB(params, **kw, **json.loads(os.environ.get("OPT_KW", "{}")))
            self.apply_now = False

            def hook_x(p):
                if self.apply_now and p.grad is not None:
                    self.opt.update(p)
                    p.grad = None
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
            self.opt = torch.optim.AdamW(params, fused="A" in stage, **kw)
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
        if os.environ.get("SPARSE_ROWS") and "X" in self.stage and hasattr(self.model, "table_rows"):
            self.opt.rows = self.model.table_rows(xy[:, :-1])
        if "C" in self.stage:
            return self.step_interleaved(mbs, len(xy))
        total = torch.zeros((), device="cuda")
        for i, mb in enumerate(mbs):
            if "B" in self.stage:
                self.apply_now = i == len(mbs) - 1     # оновлюємо ваги лише на останньому мікро-батчі
            if "G" in self.stage:
                torch.compiler.cudagraph_mark_step_begin()   # межа запуску графа
            with torch.autocast("cuda", torch.bfloat16):
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
        for old, new, slot_map in self.model.rebank(step):
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
    g = torch.Generator(device="cuda").manual_seed(0)
    t = val.t
    total = torch.zeros((), device="cuda")
    for _ in range(n):
        start = torch.randint(0, len(t) - seq - 1, (8,), device="cuda", generator=g)
        xy = t[start[:, None] + torch.arange(seq + 1, device="cuda")].long()
        with torch.autocast("cuda", torch.bfloat16):
            total += step_loss(model, xy, train=False)[0]
    model.train()
    return (total / n).item()


def matmul_peak():
    """Практична стеля заліза: великі bf16-множення."""
    a = torch.randn(8192, 8192, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(8192, 8192, device="cuda", dtype=torch.bfloat16)
    for _ in range(3):
        a @ b
    torch.cuda.synchronize()
    t = time.time()
    for _ in range(20):
        a @ b
    torch.cuda.synchronize()
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
    m = build(name.replace("S300p_", "S300_"), VOCAB).cuda()
    xy = torch.randint(0, 50257, (args.micro, args.seq + 1), device="cuda")
    if hasattr(m, "tau"):
        with torch.no_grad(), torch.autocast("cuda", torch.bfloat16):
            for _ in range(60):                        # поріг підлаштовується до реальної частки якорів
                m(xy[:, :-1])
    xy = xy[:1]
    with FlopCounterMode(display=False) as fc:
        with torch.autocast("cuda", torch.bfloat16):
            out, _ = m(xy[:, :-1])
            loss = m.train_loss(out, xy[:, 1:], True) if hasattr(m, "train_loss") else \
                F.cross_entropy(out.float().view(-1, VOCAB), xy[:, 1:].reshape(-1))
        loss.backward()
    del m
    torch.cuda.empty_cache()
    return fc.get_total_flops() * args.batch


def speed(args):
    peak = matmul_peak()
    print(f"стеля множень: {peak / 1e12:.0f} TFLOP/s", flush=True)
    data = GPUData(np.random.randint(0, 50257, 20_000_000, dtype=np.int32), seed=1)
    rows = []
    for name in args.models:
        flops = step_flops(name, args)
        for stage in args.stages:
            torch.cuda.reset_peak_memory_stats()
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
                return torch.from_numpy(np.stack([arr[i:i + args.seq + 1] for i in idx]).astype(np.int64)).cuda()

            for s in range(args.warmup):
                tr.step(get(), s)
                if stage == "base":
                    torch.cuda.synchronize()
            torch.cuda.synchronize()
            t = time.time()
            for s in range(args.warmup, args.warmup + args.n):
                loss = tr.step(get(), s)
                if stage == "base":                     # base зупиняється на хост щокроку, як lm_bench
                    loss.item()
            torch.cuda.synchronize()
            dt = (time.time() - t) / args.n
            stats = {k: float(v) for k, v in getattr(tr.model, "stats", {}).items()}
            row = dict(model=name, stage=stage, ms=dt * 1000, tflops=flops / dt / 1e12, mfu=flops / dt / peak,
                       mem=torch.cuda.max_memory_allocated() / 2**30, loss=float(loss), **stats)
            rows.append(row)
            print(f"SPEED {name:8s} {stage:5s} {dt * 1000:7.1f} ms  {flops / dt / 1e12:6.1f} TFLOP/s  "
                  f"MFU {flops / dt / peak:5.1%}  mem {row['mem']:4.0f}GB  loss {float(loss):.3f}  {stats}", flush=True)
            del tr
            torch.cuda.empty_cache()
    os.makedirs("results", exist_ok=True)
    with open("results/speed.json", "w") as f:
        json.dump(rows, f, indent=1)


def train(args):
    dp = os.environ.get("DATA")                  # DATA — інші одиниці (словосполучення); оцінка — той самий початок тексту
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
            data.g.manual_seed(1)
        # EMA_START=0.6 — «сон закріплює»: з 60% кроків — ковзне середнє ваг, оцінка на ньому
        ema_start = float(os.environ.get("EMA_START", 0) or 0)
        ema_beta = float(os.environ.get("EMA_BETA", 0.98))
        ema = None
        torch.cuda.synchronize()
        t0 = time.time()
        # PREFETCH=1 — «вода вже набрана»: наступний батч і все, що залежить лише від слів, — на окремому потоці
        pf = os.environ.get("PREFETCH") and hasattr(tr.model, "prep")
        side = torch.cuda.Stream() if pf else None

        def fetch(st):
            sq = cfg(st)[0]
            b = data.batch(args.batch * (args.seq // sq), sq)
            side.wait_stream(torch.cuda.current_stream())  # спершу батч має бути записаний основним потоком
            with torch.cuda.stream(side):
                pr = [tr.model.prep(mb[:, :-1]) for mb in b.split(micro0 * (args.seq // sq))]
                ev = torch.cuda.Event(); ev.record(side)
            return b, pr, ev

        nxt = fetch(1) if pf else None
        for step in range(1, args.steps + 1):
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
            ema_every = int(os.environ.get("EMA_EVERY", 1))     # сон періодичний: раз на N кроків, сила β^N
            if ema_start and step >= ema_start * args.steps and step % ema_every == 0:
                with torch.no_grad():
                    ps = [p for p in tr.model.parameters() if p.numel() < 10_000_000]   # без великих таблиць слів
                    if ema is None:
                        ema = [p.detach().float().clone() for p in ps]
                    else:
                        torch._foreach_lerp_(ema, [p.detach().float() for p in ps], 1 - ema_beta ** ema_every)
            if step % args.eval_every == 0 or step == args.steps:
                torch.cuda.synchronize()                # єдина зупинка — на оцінці
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
                torch.cuda.synchronize()
                t0 = time.time()
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
        torch.cuda.empty_cache()


def main():
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
