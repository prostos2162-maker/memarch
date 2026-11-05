"""Оптимізатор з нуля під нашу модель (ARCHITECTURE.md, «Ідея 5»). Не надбудова над AdamW.

Для кожної ваги вирішується: куди (градієнт), наскільки (несподіванка) і що пам'ятати (досвід):
  * **картки досвіду**: K карток, у кожної — ключ ситуації (у просторі «що модель читає») і своя
    поправка-інерція для кожної ваги. Ситуацію батчу впізнають ДО помилки — за тим, що на вході;
    береться найсхожа картка; якщо ситуація зовсім нова, а вільна картка є — заводиться нова.
    Кілька разів зливаються в одне відчуття: інерція картки = ковзне середнє її градієнтів;
  * **несподіванка**: у кожної картки своє очікуване loss; s = (очікуване − поточне) / очікуване;
    хороша (s > 0) — крок ширший для всіх; погана (s < 0) — посилюються лише найвинніші ваги
    (з найбільшим градієнтом, ті, що діяли в мить помилки); без несподіванки — звичайний крок;
  * **масштаб кроку** — нормування на весь тензор (а не окремий другий стан на кожну вагу, як в AdamW):
    дешевше і менше пам'яті. Для таблиць слів (вбудовування, вихід) — окремо для кожного слова,
    бо якорі часто рідкісні слова, і з одним масштабом на таблицю вони б майже не вчились.
"""
import math

import torch
import torch.nn.functional as F


class ExperienceOpt(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3, weight_decay=0.1, cards=4, beta=0.9, d_sit=None,
                 new_card=0.5, good=4.0, bad=6.0, max_good=2.0, max_bad=3.0, top=0.1, ema=0.98,
                 m_dtype="bf16", surprise=True, norm="tensor", scale=0.2, **_):
        # перемикачі для розбору (за замовчуванням — як у першій версії): m_dtype — у чому зберігати інерцію;
        # surprise=False — без несподіванки; norm="row" — масштаб на кожен рядок матриці; scale — множник кроку
        self.m_dtype = torch.float32 if m_dtype == "fp32" else torch.bfloat16
        self.surprise, self.norm, self.scale = surprise, norm, scale
        super().__init__(params, dict(lr=lr, weight_decay=weight_decay))
        self.K, self.beta, self.new_card = cards, beta, new_card
        self.good, self.bad, self.max_good, self.max_bad, self.top, self.ema = good, bad, max_good, max_bad, top, ema
        self.keys = None                       # (K, d_sit) ключі ситуацій
        self.used = torch.zeros(cards, dtype=torch.bool)
        self.expect = [None] * cards           # очікуване loss кожної картки
        self.last = dict(card=0, s=0.0)

    @torch.no_grad()
    def _pick(self, sit):
        """Найсхожа картка для ситуації (або нова, якщо ситуація зовсім інша)."""
        sit = F.normalize(sit.float().flatten(), dim=0)
        if self.keys is None:
            self.keys = torch.zeros(self.K, sit.numel(), device=sit.device)
        if not self.used.any():
            k = 0
        else:
            sim = (self.keys @ sit).masked_fill(~self.used.to(sit.device), -2)
            best = int(sim.argmax())
            free = (~self.used).nonzero()
            k = int(free[0]) if float(sim[best]) < self.new_card and len(free) else best
        if not self.used[k]:
            self.keys[k] = sit
            self.used[k] = True
        else:
            self.keys[k] = F.normalize(0.95 * self.keys[k] + 0.05 * sit, dim=0)   # ключ підлаштовується
        return k

    @torch.no_grad()
    def step(self, loss, situation):
        k = self._pick(situation)
        loss = float(loss)
        if self.expect[k] is None:
            self.expect[k] = loss
        s = (self.expect[k] - loss) / max(self.expect[k], 1e-6)
        self.expect[k] = self.ema * self.expect[k] + (1 - self.ema) * loss
        self.last = dict(card=k, s=s)
        for g in self.param_groups:
            for p in g["params"]:
                if p.grad is None:
                    continue
                st = self.state[p]
                if "m" not in st:
                    st["m"] = torch.zeros(self.K, *p.shape, device=p.device, dtype=self.m_dtype)
                    st["seen"] = torch.zeros(self.K, dtype=torch.bool)
                grad = p.grad.float()
                if not st["seen"][k]:                      # нова картка стартує з досвіду найсхожої наявної
                    if st["seen"].any():
                        st["m"][k] = st["m"][int(st["seen"].nonzero()[0])]
                    st["seen"][k] = True
                m = st["m"][k].float().mul_(self.beta).add_(grad, alpha=1 - self.beta)
                st["m"][k] = m.to(self.m_dtype)
                if p.dim() == 2 and p.shape[0] >= 10000:
                    # таблиці слів (вбудовування, вихід, «книжка»): масштаб окремо для кожного слова, інакше
                    # рідкісні слова (а якорі часто саме вони) тонуть серед частих. Рядок без уроку не змінюється.
                    rms = m.pow(2).mean(1, keepdim=True).sqrt()
                    upd = torch.where(grad.abs().sum(1, keepdim=True) > 0, m / (rms + 1e-8), torch.zeros_like(m))
                elif self.norm == "row" and p.dim() >= 2:
                    upd = m / (m.pow(2).flatten(1).mean(1).sqrt().view(-1, *[1] * (p.dim() - 1)) + 1e-8)
                else:
                    upd = m / (m.pow(2).mean().sqrt() + 1e-8)       # масштаб — на весь тензор
                if not self.surprise:
                    pass
                elif s > 0:
                    upd = upd * min(self.max_good, 1 + self.good * s)
                elif s < 0:
                    a = grad.abs().flatten()
                    if a.numel() > 65536:
                        a = a[torch.randint(0, a.numel(), (65536,), device=a.device)]
                    culprit = (grad.abs() >= a.quantile(1 - self.top)).float()
                    upd = upd * (1 + culprit * (min(self.max_bad, 1 + self.bad * -s) - 1))
                vocab_table = p.dim() == 2 and p.shape[0] >= 10000
                if g["weight_decay"] and p.dim() >= 2 and not vocab_table:   # рідкісні слова не стираються
                    p.mul_(1 - g["lr"] * g["weight_decay"])
                p.add_(upd.to(p.dtype), alpha=-g["lr"] * self.scale)        # 0.2: щоб розмір кроку був як у AdamW


class InstinctOpt(torch.optim.Optimizer):
    """Друга версія (Стефан: досвід — не пам'ять кожного м'яза, а одна думка-інстинкт, що веде все тіло).

    * на кожну вагу одна інерція (fp32) — пам'яті вдвічі менше, ніж в AdamW;
    * картка досвіду = ключ ситуації + очікуване loss + **по одному підсилювачу на тензор ваг** (сотні чисел,
      а не копія моделі); ситуацію впізнають за входом, нова → найсхожа або вільна картка;
    * інстинкт вчиться на несподіванці й розподіляє сміливість (середнє підсилювачів картки = 1): погана →
      підсилювачі винних тензорів (найбільший відносний градієнт саме в цю мить) ростуть («тут тримай
      міцніше»); хороша → назад до рівномірного; без несподіванки — повільно до 1. Межі 0.5–3;
    * що вища ставка (сильне здивування АБО великий крок навчання), то більше крок іде за інстинктом;
    * масштаб кроку — на тензор (таблиці слів — на рядок, без згасання ваг), як у першій версії.
    """

    def __init__(self, params, lr=1e-3, weight_decay=0.1, cards=4, beta=0.9, new_card=0.5, ema=0.98,
                 good=0.05, bad=0.2, top=0.1, relax=0.01, scale=0.2, lr_peak=None, norm="tensor", **_):
        super().__init__(params, dict(lr=lr, weight_decay=weight_decay))
        self.K, self.beta, self.new_card, self.ema = cards, beta, new_card, ema
        self.good, self.bad, self.top, self.relax, self.scale = good, bad, top, relax, scale
        self.lr_peak = lr_peak or lr
        self.norm = norm                   # "row" — масштаб на кожен рядок матриці (кращий у розборі)
        self.plist = [p for g in self.param_groups for p in g["params"]]
        self.gain = torch.ones(cards, len(self.plist))      # інстинкт: підсилювач кожного тензора в кожній картці
        self.keys, self.used = None, torch.zeros(cards, dtype=torch.bool)
        self.expect = [None] * cards
        self.last = dict(card=0, s=0.0, stake=0.0)

    _pick = ExperienceOpt._pick

    @torch.no_grad()
    def step(self, loss, situation):
        k = self._pick(situation)
        loss = float(loss)
        if self.expect[k] is None:
            self.expect[k] = loss
        s = (self.expect[k] - loss) / max(self.expect[k], 1e-6)
        self.expect[k] = self.ema * self.expect[k] + (1 - self.ema) * loss
        lr_now = self.param_groups[0]["lr"]
        stake = min(1.0, 4 * abs(s) + 0.5 * lr_now / self.lr_peak)        # ставка: здивування + швидкість
        self.last = dict(card=k, s=s, stake=stake)
        rel = torch.zeros(len(self.plist))
        i = 0
        for g in self.param_groups:
            for p in g["params"]:
                if p.grad is not None:
                    st = self.state[p]
                    if "m" not in st:
                        st["m"] = torch.zeros_like(p, dtype=torch.float32)
                    grad = p.grad.float()
                    m = st["m"].mul_(self.beta).add_(grad, alpha=1 - self.beta)
                    vocab_table = p.dim() == 2 and p.shape[0] >= 10000
                    if vocab_table:
                        rms = m.pow(2).mean(1, keepdim=True).sqrt()
                        upd = torch.where(grad.abs().sum(1, keepdim=True) > 0, m / (rms + 1e-8), torch.zeros_like(m))
                    elif self.norm == "row" and p.dim() >= 2:
                        upd = m / (m.pow(2).flatten(1).mean(1).sqrt().view(-1, *[1] * (p.dim() - 1)) + 1e-8)
                    else:
                        upd = m / (m.pow(2).mean().sqrt() + 1e-8)
                    rel[i] = grad.norm() / (p.norm() + 1e-8)           # наскільки тензор «діяв» у цю мить
                    gain = 1 + (float(self.gain[k, i]) - 1) * stake
                    if g["weight_decay"] and p.dim() >= 2 and not vocab_table:
                        p.mul_(1 - g["lr"] * g["weight_decay"])
                    p.add_(upd.to(p.dtype), alpha=-g["lr"] * self.scale * gain)
                i += 1
        # інстинкт вчиться після кроку
        # інстинкт розподіляє сміливість між частинами, а не роздуває її: середнє підсилювачів картки = 1
        gk = self.gain[k]
        if s < 0:                                    # погано: «тут тримай міцніше» саме для винних у цю мить
            culprit = rel >= rel.quantile(1 - self.top)
            gk[culprit] += self.bad * min(1.0, -s * 10)
        elif s > 0:                                  # добре: усе спрацювало — назад до рівномірного
            gk += self.good * min(1.0, s * 10) * (1 - gk)
        gk += self.relax * (1 - gk)
        gk.clamp_(0.5, 3.0)
        gk /= gk.mean()


import os as _os
# BF16_ALL=1 (Стефан, 03.11): усі ваги в bf16 без fp32-копій. Щоб малі оновлення не губились округленням до найближчого
# bf16 (вага «замерзає»), нове значення ваги (і інерції) записується СТОХАСТИЧНИМ округленням: вгору чи вниз з
# імовірністю, пропорційною відкинутому залишку, — у середньому оновлення зберігається точно.
_SR = (_os.environ.get("BF16_ALL") == "1" or _os.environ.get("OPT_SR") == "1") and _os.environ.get("OPT_SR_OFF") != "1"


def _sr_bf16(x):
    """fp32 → bf16 зі стохастичним округленням: до бітового подання fp32 додається випадкове число в молодших 16 бітах,
    потім молодші 16 біт відкидаються (bf16 — це старші 16 біт fp32). Для від'ємних так само (знак окремим бітом)."""
    xi = x.contiguous().view(torch.int32)
    r = torch.randint_like(xi, 0, 1 << 16)
    return ((xi + r) & -65536).view(torch.float32).to(torch.bfloat16)


def _instinct_step_impl(p, grad, m, w, g_i, stake, lr_t, beta: float, wd: float, kind: str, want_rel: bool, cs=None,
                        seed_t=None):
    """Один крок інстинкту для однієї ваги (математика — як InstinctOptB._update_eager).
    w — точна fp32-копія ваги, якщо сама вага лежить у bf16 (тоді модель множить bf16 напряму, без щокрокового
    перетворення), інакше None. m — інерція (fp32 або bf16). Повертає |g|/|w| (want_rel) або 0."""
    if (_PALLAS and seed_t is not None and kind in ("hs", "hs3", "vocab") and cs is None and w is None and _SR and p.dtype == torch.bfloat16
            and m.dtype == torch.bfloat16 and not hasattr(p, "placements")):
        # PALLAS_OPT=1 (TPU): злите ядро Instinct — два проходи по пам'яті (memarch/pallas_instinct.py); математика та
        # сама, стохастичне округлення — лічильникове (хеш номера елемента й насіння кроку)
        import pallas_instinct
        rel = (grad.float().norm() / (p.float().norm() + 1e-8)) if want_rel else torch.zeros((), device=p.device)
        eff = 1 + (g_i - 1) * stake
        seed = (seed_t.add_(1) * 1000003).to(torch.int32).view(torch.float32)
        sc = torch.stack([torch.full((), beta, device=p.device), (lr_t * eff).float(),
                          (1 - lr_t * wd).float() * torch.ones((), device=p.device), seed])
        m_new, w_new = pallas_instinct.torch_op(kind)(grad.to(torch.bfloat16), m, p, sc)
        if _REBIND:                           # OPT_REBIND: нові буфери віддаються назовні — без копії в старі
            return rel, m_new, w_new, None
        if not (pallas_instinct.TORCH_ALIAS or pallas_instinct.TORCH_INPLACE):   # ядро вже записало m і w на місці
            m.copy_(m_new)
            p.copy_(w_new)
        return rel
    if (_PALLAS and seed_t is not None and kind in ("hs", "hs3") and cs is not None and w is None and _SR
            and p.dtype == torch.bfloat16 and m.dtype == torch.bfloat16 and not hasattr(p, "placements")):
        # PALLAS_OPT=1 + OPT_HS_LAG=1: гомеостаз за ОДИН прохід (memarch/pallas_hs1.py) — норма стовпців із
        # попереднього кроку (cs); перший крок (у cs нулі) — ядро само рахує свіжу (як torch.where нижче)
        import pallas_hs1
        rel = (grad.float().norm() / (p.float().norm() + 1e-8)) if want_rel else torch.zeros((), device=p.device)
        eff = 1 + (g_i - 1) * stake
        seed = (seed_t.add_(1) * 1000003).to(torch.int32).view(torch.float32)
        sc = torch.stack([torch.full((), beta, device=p.device), (lr_t * eff).float(),
                          (1 - lr_t * wd).float() * torch.ones((), device=p.device), seed])
        m_new, w_new, cs_new = pallas_hs1.torch_op(kind)(grad.to(torch.bfloat16), m, p, cs, sc)
        if _REBIND:
            return rel, m_new, w_new, cs_new
        m.copy_(m_new)
        p.copy_(w_new)
        cs.copy_(cs_new)
        return rel
    grad = grad.float()
    mf = m.float() * beta + grad * (1 - beta)
    m.copy_(_sr_bf16(mf) if (_SR and m.dtype == torch.bfloat16) else mf)
    if kind == "vocab":
        rms = mf.pow(2).mean(1, keepdim=True).sqrt()
        upd = torch.where(grad.abs().sum(1, keepdim=True) > 0, mf / (rms + 1e-8), torch.zeros_like(mf))
    elif kind == "row3":
        upd = mf / (mf.pow(2).mean(1, keepdim=True).sqrt() + 1e-8)
    elif kind.startswith("hsn"):
        # гомеостаз кількома хвилями: рядок → (стовпець → рядок) × n; 3D-експерти (E, вхід, вихід): нейрон — dim 1
        n = int(kind[3:-1]) if kind.endswith("e") else int(kind[3:])
        ro, co = (1, 2) if kind.endswith("e") else (1, 0)
        upd = mf / (mf.pow(2).mean(ro, keepdim=True).sqrt() + 1e-8)
        for _ in range(n):
            upd = upd / (upd.pow(2).mean(co, keepdim=True).sqrt() + 1e-8)
            upd = upd / (upd.pow(2).mean(ro, keepdim=True).sqrt() + 1e-8)
    elif kind in ("hs", "hs3") and cs is not None:
        # OPT_HS_LAG: гомеостаз за один прохід — норма «другої» осі береться з попереднього кроку (cs), нова рахується
        # попутно з того самого проміжного й зберігається на наступний крок; перший крок (cs == 0) — свіжа норма
        ro, co = (1, 0) if kind == "hs" else (1, 2)
        upd = mf / (mf.pow(2).mean(ro, keepdim=True).sqrt() + 1e-8)
        now = upd.pow(2).mean(co, keepdim=True)
        col = torch.where(cs > 0, cs, now)
        cs.copy_(now)
        upd = upd / (col.sqrt() + 1e-8)
        upd = upd / (upd.pow(2).mean(ro, keepdim=True).sqrt() + 1e-8)
    elif kind == "hs3":
        upd = mf / (mf.pow(2).mean(1, keepdim=True).sqrt() + 1e-8)
        upd = upd / (upd.pow(2).mean(2, keepdim=True).sqrt() + 1e-8)
        upd = upd / (upd.pow(2).mean(1, keepdim=True).sqrt() + 1e-8)
    elif kind == "hs":
        # гомеостаз: і кожен нейрон (рядок), і кожен вхід (стовпець) міняються однаково сильно
        upd = mf / (mf.pow(2).mean(1, keepdim=True).sqrt() + 1e-8)
        upd = upd / (upd.pow(2).mean(0, keepdim=True).sqrt() + 1e-8)
        upd = upd / (upd.pow(2).mean(1, keepdim=True).sqrt() + 1e-8)
    elif kind.startswith("hsS"):
        # те саме для ваги, порізаної по рядках між чипами (hyperop SHARD): норма стовпця — сума по всіх чипах
        # (один маленький all-reduce у графі оптимізатора); kind = "hsS<W>,<усього рядків>"
        W_, N_ = (int(v) for v in kind[3:].split(","))
        import torch.distributed._functional_collectives as funcol
        import torch.distributed as tdist
        upd = mf / (mf.pow(2).mean(1, keepdim=True).sqrt() + 1e-8)
        col = funcol.all_reduce(upd.pow(2).sum(0, keepdim=True), "sum", tdist.group.WORLD) / N_
        upd = upd / (col.sqrt() + 1e-8)
        upd = upd / (upd.pow(2).mean(1, keepdim=True).sqrt() + 1e-8)
    elif kind == "row":
        upd = mf / (mf.pow(2).flatten(1).mean(1).sqrt().view(-1, *[1] * (p.dim() - 1)) + 1e-8)
    else:
        upd = mf / (mf.pow(2).mean().sqrt() + 1e-8)
    t = p if w is None else w
    rel = grad.norm() / (t.norm().float() + 1e-8) if want_rel else torch.zeros((), device=p.device)
    eff = 1 + (g_i - 1) * stake
    if _SR and w is None and t.dtype == torch.bfloat16 and not hasattr(t, "placements"):
        new = t.float()                                  # уся арифметика кроку — у fp32, запис — зі стох. округленням
        if wd:
            new = new * (1 - lr_t * wd)
        t.copy_(_sr_bf16(new - upd * (lr_t * eff)))
        return rel
    if wd:
        t.mul_(1 - lr_t * wd)
    delta = (upd * (lr_t * eff)).to(t.dtype)
    if hasattr(t, "placements"):         # FSDP (DTensor): оновлення могло стати повним — позамісцево, назад у шматок
        t.copy_((t - delta).redistribute(t.device_mesh, t.placements))
    else:
        t.sub_(delta)
    if w is not None:
        p.copy_(w)
    return rel


def _vocab_rows_impl(p, grad, m, w, rows, g_i, stake, lr_t, beta: float):
    """Таблиця слів / словосполучень: оновлюються лише задіяні в кроці рядки (повтори дають ті самі значення)."""
    g = grad[rows].float()
    mr = m[rows].float() * beta + g * (1 - beta)
    m[rows] = _sr_bf16(mr) if (_SR and m.dtype == torch.bfloat16) else mr.to(m.dtype)
    upd = mr / (mr.pow(2).mean(1, keepdim=True).sqrt() + 1e-8)
    t = p if w is None else w
    if _SR and w is None and t.dtype == torch.bfloat16:
        t[rows] = _sr_bf16(t[rows].float() - upd * (lr_t * (1 + (g_i - 1) * stake)))
        return
    t[rows] = t[rows] - (upd * (lr_t * (1 + (g_i - 1) * stake))).to(t.dtype)
    if w is not None:
        p[rows] = w[rows].to(p.dtype)


_compiled = {}
_NO_COMPILE = bool(__import__("os").environ.get("NO_COMPILE"))
# OPT_FN=1: prepare/finish Instinct — чистими функціями, кожна однією скомпільованою програмою (0 — як було)
_OPT_FN = _os.environ.get("OPT_FN", "1") == "1"
# OPT_HS_LAG=1: гомеостаз (norm="hs") за один прохід з нормою стовпців із попереднього кроку (див. _instinct_step_impl)
_HS_LAG = _os.environ.get("OPT_HS_LAG") == "1"
# PALLAS_OPT=1 (лише TPU): крок «hs»/«hs3» — злитим Pallas-ядром (pallas_instinct.py); насіння стох. округлення —
# лічильник на пристрої (крокує в графі, без хоста)
_PALLAS = _os.environ.get("PALLAS_OPT") == "1"
# OPT_REBIND=1 (з PALLAS_OPT): нові m/w із ядер Pallas не копіюються в старі буфери, а підставляються (p.data = нове)
_REBIND = _PALLAS and _os.environ.get("OPT_REBIND") == "1"
# OPT_STACK_INGRAPH=1: вибірка g[sel] і запис rel[sel] — усередині скомпільованого _step_all (жодних eager-програм
# довкола кроку оптимізатора); за замовчуванням вимкнено — поведінка як раніше
_STACK_INGRAPH = _os.environ.get("OPT_STACK_INGRAPH") == "1"
if _PALLAS and _os.environ.get("DEVICE") == "tpu":
    # операції моста — ДО torch.compile (jax_op уперше під трасуванням dynamo падає на побудові схеми, проба 57)
    try:
        import pallas_instinct as _pi
        for _k in ("hs", "hs3", "vocab"):
            _pi.torch_op(_k)
        if _HS_LAG:
            import pallas_hs1 as _ph1
            for _k in ("hs", "hs3"):
                _ph1.torch_op(_k)
    except Exception as _e:                      # без torch_tpu (ПК) — пропускаємо
        print("exp_opt: операції Pallas не створено заздалегідь:", type(_e).__name__, _e)


def _prepare_impl(keys, used, expect, seen, gain, sit, loss, lr_ratio, new_card: float, ema: float):
    """Те саме, що InstinctOptB.prepare: вибір картки ситуації, очікування й несподіванка, ставка, підсилювачі."""
    K = keys.shape[0]
    sit = F.normalize(sit, dim=0)
    any_used = used.any()
    best_sim, best = (keys @ sit).masked_fill(~used, -2).max(0)
    free = ~used
    first_free = free.int().argmax()
    newer = (best_sim < new_card) & free.any()
    k = torch.where(~any_used, torch.zeros_like(best), torch.where(newer, first_free, best))
    oh = F.one_hot(k, K).float()
    was = (oh * used).sum() > 0
    key_k = oh @ keys
    new_key = torch.where(was, F.normalize(0.95 * key_k + 0.05 * sit, dim=0), sit)
    keys.add_(oh[:, None] * (new_key - key_k)[None])
    used.copy_(used | oh.bool())
    loss = loss.float()
    e = torch.where((oh * seen).sum() > 0, (oh * expect).sum(), loss)
    s = (e - loss) / e.clamp(min=1e-6)
    expect.add_(oh * (ema * e + (1 - ema) * loss - expect))
    seen.add_(oh * (1 - seen))
    stake = (4 * s.abs() + 0.5 * lr_ratio).clamp(max=1.0)
    g = oh @ gain
    return s, stake, k, g, oh


def _finish_impl(s, g, rel, gain, oh, top: float, bad: float, good: float, relax: float):
    """Те саме, що InstinctOptB.finish: інстинкт вчиться (підсилювачі картки за наслідком кроку)."""
    gk = g.clone()
    culprit = (rel >= rel.quantile(1 - top)).float()
    gk += torch.where(s < 0, culprit * bad * (-s * 10).clamp(max=1), torch.zeros_like(gk))
    gk += torch.where(s > 0, good * (s * 10).clamp(max=1) * (1 - gk), torch.zeros_like(gk))
    gk += relax * (1 - gk)
    gk = gk.clamp(0.5, 3.0)
    gain.add_(oh[:, None] * (gk / gk.mean() - gain))


def _instinct_step(p, grad, m, w, g_i, stake, lr_t, beta, wd, kind, want_rel):
    """Окрема скомпільована копія на кожне поєднання (тип ваги, форма, dtype, з fp32-копією, з |g|/|w|, wd):
    у компілятора межа ~8 перекомпіляцій на функцію (у потоці зворотного проходу налаштування межі не діє),
    після неї він мовчки виконує крок поелементно — ≈12 кернелів на вагу замість 2–3."""
    import types
    key = (kind, tuple(p.shape), p.dtype, m.dtype, w is None, bool(want_rel), bool(wd))
    if _NO_COMPILE:
        return _instinct_step_impl(p, grad, m, w, g_i, stake, lr_t, beta, wd, kind, want_rel)
    fn = _compiled.get(key)
    if fn is None:
        name = f"_inst_{len(_compiled)}"
        code = _instinct_step_impl.__code__.replace(co_name=name)
        fn = _compiled[key] = torch.compile(types.FunctionType(code, globals(), name), dynamic=False)
    return fn(p, grad, m, w, g_i, stake, lr_t, beta, wd, kind, want_rel)


def _step_all_impl(ps, grads, ms, ws, g, stake, lr_t, lr_v, beta: float, spec, want_rel: bool, css=None, seed_t=None,
                   gsel=None, rel_buf=None):
    """Крок інстинкту для всіх ваг одним графом (OPT_ALL): одна програма на пристрої замість сотні дрібних.
    OPT_STACK_INGRAPH=1: gsel — кортеж номерів ваг (сталі Python-числа), g — ПОВНИЙ вектор g, rel_buf — повний буфер
    |g|/|w|; вибірка g[...] і запис rel[...] робляться тут, усередині скомпільованого графа (без окремих eager-програм)."""
    rels, news = [], []
    for i, (p, gr, m, w) in enumerate(zip(ps, grads, ms, ws)):
        kind, wd, vocab = spec[i]
        r = _instinct_step_impl(p, gr, m, w, g[gsel[i]] if gsel is not None else g[i], stake, lr_v if vocab else lr_t, beta, wd, kind, want_rel,
                                css[i] if css is not None else None, seed_t)
        if isinstance(r, tuple):                         # OPT_REBIND: (|g|/|w|, нова m, нова вага, новий cs)
            r, *nw = r
            news.append(tuple(nw))
        else:
            news.append((None, None, None))
        rels.append(r.full_tensor() if hasattr(r, "full_tensor") else r)    # FSDP: норма — повна, як число
    if rel_buf is not None:                              # OPT_STACK_INGRAPH: запис |g|/|w| у буфер — у цьому ж графі
        if want_rel:
            for k, r in zip(gsel, rels):
                rel_buf[k].copy_(r)
        return (None, news) if _REBIND else None
    if _REBIND:
        return torch.stack(rels), news
    return torch.stack(rels)


class InstinctOptB:
    """InstinctOpt без жодної зупинки на хост і з оновленням ваг у зворотному проході (етап Б).

    Loss кроку відомий після прямого проходу — ще до першого градієнта, тож несподіванка цього ж кроку
    встигає до оновлення ваг (без затримки). Ситуація (картка) — за входом, ще до помилки. Усі числа
    інстинкту живуть на відеокарті. Порядок у кроці: prepare(loss, sit) → backward (update(p) у хуку
    кожного параметра) → finish() (інстинкт вчиться).
    """

    def __init__(self, params, lr=1e-3, weight_decay=0.1, cards=4, beta=0.9, new_card=0.5, ema=0.98,
                 good=0.05, bad=0.2, top=0.1, relax=0.01, scale=0.2, lr_peak=None, norm="tensor", d_sit=None,
                 m_bf16=False, rel_every=1, **_):
        # m_bf16 — інерція в bf16 (вдвічі менше байтів щокроку); rel_every — |g|/|w| для «винних» ваг не щокроку
        self.m_dtype = torch.bfloat16 if m_bf16 else torch.float32
        self.rel_every, self.n_step = rel_every, 0
        self.plist = list(params)
        self.idx = {id(p): i for i, p in enumerate(self.plist)}
        self.param_groups = [dict(lr=lr, weight_decay=weight_decay)]
        self.K, self.beta, self.new_card, self.ema = cards, beta, new_card, ema
        self.good, self.bad, self.top, self.relax, self.scale, self.norm = good, bad, top, relax, scale, norm
        self.lr_peak = lr_peak or lr
        dev = self.plist[0].device
        P = len(self.plist)
        self.gain = torch.ones(cards, P, device=dev)
        self.keys = None
        self.used = torch.zeros(cards, dtype=torch.bool, device=dev)
        self.expect = torch.zeros(cards, device=dev)
        self.seen = torch.zeros(cards, device=dev)   # чи картка вже має очікування
        self.rel = torch.zeros(P, device=dev)
        self.m = [torch.zeros_like(p, dtype=self.m_dtype) for p in self.plist]
        # bf16-ваги (експерти) — точна копія в fp32 живе тут; модель бачить лише bf16.
        # BF16_ALL=1 — копій немає: оновлення пишуться в саму bf16-вагу зі стохастичним округленням (_sr_bf16)
        no_master = _os.environ.get("BF16_ALL") == "1"
        self.w = [p.detach().float().clone() if (p.dtype == torch.bfloat16 and not no_master) else None for p in self.plist]
        self.k = torch.zeros((), dtype=torch.long, device=dev)
        self.s = torch.zeros((), device=dev)
        self.stake = torch.zeros((), device=dev)
        import os
        # злитий крок і без torch.compile (NO_COMPILE: XLA збирає граф сама; на CPU — просто повільніше)
        self.fused, self._lr_last = os.environ.get("OPT_FUSED", "1") == "1", None

    @torch.no_grad()
    def prepare(self, loss, situation):
        if _OPT_FN:                                       # одна скомпільована програма замість ~30 дрібних операцій
            return self._prepare_fn(loss, situation)
        sit = F.normalize(situation.float().flatten(), dim=0)
        if self.keys is None:
            self.keys = torch.zeros(self.K, sit.numel(), device=sit.device)
        # (усе через маску картки oh: вибірка за номером-тензором змусила б хост чекати на відеокарту)
        any_used = self.used.any()
        best_sim, best = (self.keys @ sit).masked_fill(~self.used, -2).max(0)
        free = ~self.used
        first_free = free.int().argmax()
        newer = (best_sim < self.new_card) & free.any()
        k = torch.where(~any_used, torch.zeros_like(best), torch.where(newer, first_free, best))
        oh = F.one_hot(k, self.K).float()
        was = (oh * self.used).sum() > 0
        key_k = oh @ self.keys
        new_key = torch.where(was, F.normalize(0.95 * key_k + 0.05 * sit, dim=0), sit)
        self.keys.add_(oh[:, None] * (new_key - key_k)[None])
        self.used |= oh.bool()
        loss = loss.detach().float()
        e = torch.where((oh * self.seen).sum() > 0, (oh * self.expect).sum(), loss)
        self.s = (e - loss) / e.clamp(min=1e-6)
        self.expect.add_(oh * (self.ema * e + (1 - self.ema) * loss - self.expect))
        self.seen.add_(oh * (1 - self.seen))
        self.oh = oh
        lr = self.param_groups[0]["lr"]
        self.stake = (4 * self.s.abs() + 0.5 * lr / self.lr_peak).clamp(max=1.0)
        self.k = k
        self.g = oh @ self.gain

    def _ratio_sync(self, dev):
        """lr/lr_peak як 0-вимірний тензор (дані графа, не константа): змінюється lr — програма та сама."""
        lr = self.param_groups[0]["lr"]
        if not hasattr(self, "_lr_ratio"):
            self._lr_ratio, self._ratio_last = torch.zeros((), device=dev), None
        if self._ratio_last != lr:
            self._lr_ratio.copy_(torch.tensor(lr / self.lr_peak))
            self._ratio_last = lr

    def _prepare_fn(self, loss, situation):
        sit = situation.float().flatten()
        if self.keys is None:
            self.keys = torch.zeros(self.K, sit.numel(), device=sit.device)
        self._ratio_sync(sit.device)
        fn = _prepare_impl if _NO_COMPILE else _compiled.get("prepare")
        if fn is None:
            fn = _compiled["prepare"] = torch.compile(_prepare_impl, dynamic=False)
        self.s, self.stake, self.k, self.g, self.oh = fn(self.keys, self.used, self.expect, self.seen, self.gain, sit,
                                                         loss.detach(), self._lr_ratio, float(self.new_card), float(self.ema))

    @torch.no_grad()
    def update(self, p):
        if self.fused:
            return self._update_fused(p)
        return self._update_eager(p)

    @torch.no_grad()
    def _update_fused(self, p):
        """Те саме оновлення, що _update_eager, але одним скомпільованим графом: інерція, масштаб, норми для
        інстинкту й крок — за 2–3 проходи пам'яттю замість ~12 окремих кернелів (і без запусків з процесора)."""
        i = self.idx[id(p)]
        grp = self.param_groups[0]
        kind, wd, vocab_table = self._spec(p)
        self._lr_sync(p)
        want_rel = self.n_step % self.rel_every == 0
        rows = getattr(self, "rows", {}).get(id(p)) if vocab_table else None
        if rows is not None:                                 # лише слова, потрібні на цьому кроці
            key = ("rows", tuple(p.shape), rows.numel(), p.dtype, self.m[i].dtype, self.w[i] is None)
            fn = _vocab_rows_impl if _NO_COMPILE else _compiled.get(key)
            if fn is None:
                fn = _compiled[key] = torch.compile(_vocab_rows_impl, dynamic=False)
            fn(p, p.grad, self.m[i], self.w[i], rows, self.g[i], self.stake, self._lr_v, self.beta)
            if want_rel:
                self.rel[i] = 0.0
            return
        rel = _instinct_step(p, p.grad, self.m[i], self.w[i], self.g[i], self.stake,
                             self._lr_v if vocab_table else self._lr_t,
                             self.beta, grp["weight_decay"] / self.scale if wd else 0.0, kind, want_rel)
        if want_rel:
            self.rel[i] = rel

    @torch.no_grad()
    def update_all(self):
        """Усі ваги з готовими градієнтами — одним скомпільованим кроком (OPT_ALL=1; таблиці з SPARSE_ROWS — окремо)."""
        rows = getattr(self, "rows", {})
        grp = self.param_groups[0]
        sel = [i for i, p in enumerate(self.plist) if p.grad is not None and id(p) not in rows]
        for i, p in enumerate(self.plist):
            if p.grad is not None and id(p) in rows:
                self._update_fused(p)
        if not sel:
            return
        self._lr_sync(self.plist[sel[0]])
        want_rel = self.n_step % self.rel_every == 0
        spec = []
        for i in sel:
            kind, wd, vocab = self._spec(self.plist[i])
            spec.append((kind, grp["weight_decay"] / self.scale if wd else 0.0, vocab))
        # OPT_HS_LAG=1: стан гомеостазу — середні квадрати по стовпцях з попереднього кроку (ліниво, за формою ваги)
        cs_all = self.__dict__.setdefault("cs", [None] * len(self.plist))
        if _HS_LAG:
            for j, i in enumerate(sel):
                if cs_all[i] is None and spec[j][0] in ("hs", "hs3"):
                    p = self.plist[i]
                    shp = (1, p.shape[1]) if spec[j][0] == "hs" else (p.shape[0], p.shape[1], 1)
                    cs_all[i] = torch.zeros(shp, device=p.device, dtype=torch.float32)
        # OPT_GROUP_PARAMS: крок групами по ~N параметрів (кожна група — своя програма): математика та сама, але тимчасові
        # fp32-проміжні тримаються лише для однієї групи (на ПК: тимчасова пам'ять ÷3 за +1.5% часу). 0 — одним графом.
        gsize = int(_os.environ.get("OPT_GROUP_PARAMS", 50_000_000))
        chunks, cur, acc = [], [], 0
        for j, i in enumerate(sel):
            cur.append(j)
            acc += self.plist[i].numel()
            if gsize and acc >= gsize:
                chunks.append(cur)
                cur, acc = [], 0
        if cur:
            chunks.append(cur)
        cache = self.__dict__.setdefault("_sel_t", {})
        for ch in chunks:
            ssel = [sel[j] for j in ch]
            sspec = tuple(spec[j] for j in ch)
            key = ("all", tuple(ssel), sspec, bool(want_rel), _HS_LAG)
            fn = _step_all_impl if _NO_COMPILE else _compiled.get(key)
            if fn is None:
                # кожен варіант (група ваг × чи рахувати |g|/|w|) — ОКРЕМА копія функції зі своїм іменем: у dynamo межа
                # ~8 перекомпіляцій на одну функцію, після неї він мовчки виконує крок без компіляції (на ПК: крок
                # 1000+ мс при 28 мс роботи GPU); на справжній моделі груп ~7 × 2 = 14 варіантів — понад межу
                import types
                name = f"_step_all_{len(_compiled)}"
                code = _step_all_impl.__code__.replace(co_name=name)
                fn = _compiled[key] = torch.compile(types.FunctionType(code, globals(), name, _step_all_impl.__defaults__),
                                                    dynamic=False)
            ps = [self.plist[i] for i in ssel]
            # номери ваг — тензор на пристрої, створений РАЗ (а не щокроку з Python-списку: передача з хоста посеред кроку)
            sel_t = cache.get(tuple(ssel))
            if sel_t is None:
                sel_t = cache[tuple(ssel)] = torch.tensor(ssel, device=self.g.device)
            if _PALLAS and not hasattr(self, "_seed_t"):  # насіння стох. округлення злитого ядра — лічильник на пристрої
                self._seed_t = torch.zeros((), dtype=torch.int64, device=self.g.device)
            if _STACK_INGRAPH:
                # OPT_STACK_INGRAPH=1: повний g і буфер rel — у граф; вибірка/запис за сталими номерами ssel усередині
                rel = fn(ps, [p.grad for p in ps], [self.m[i] for i in ssel], [self.w[i] for i in ssel],
                         self.g, self.stake, self._lr_t, self._lr_v, self.beta, sspec, want_rel,
                         [cs_all[i] for i in ssel], getattr(self, "_seed_t", None), tuple(ssel), self.rel)
            else:
                rel = fn(ps, [p.grad for p in ps], [self.m[i] for i in ssel], [self.w[i] for i in ssel],
                         self.g[sel_t], self.stake, self._lr_t, self._lr_v, self.beta, sspec, want_rel,
                         [cs_all[i] for i in ssel], getattr(self, "_seed_t", None))
            if _REBIND:
                # OPT_REBIND=1: замість копіювання нових m/w у старі буфери (≈6 мс захисних копій на крок, проба 64) —
                # переприв'язка: вага бере новий буфер (p.data — лише вказівник), стан оптимізатора — нові тензори
                rel, news = rel
                with torch.no_grad():
                    for i, (mn, wn, csn) in zip(ssel, news):
                        if wn is not None:
                            self.plist[i].data = wn
                            self.m[i] = mn
                            if csn is not None:
                                cs_all[i] = csn
            if want_rel and not _STACK_INGRAPH:
                if hasattr(rel, "full_tensor"):              # FSDP під компіляцією: норми виходять DTensor — у звичайний буфер
                    rel = rel.full_tensor()
                self.rel[sel_t] = rel
            for p in ps:
                p.grad = None

    def _lr_sync(self, p):
        grp = self.param_groups[0]
        if not hasattr(self, "_lr_t"):
            self._lr_t = torch.zeros((), device=p.device)
        if not hasattr(self, "_lr_v"):
            import os
            self._lr_v, self._vmul = torch.zeros((), device=p.device), float(os.environ.get("VOCAB_LR_MUL", 1))
        if self._lr_last != grp["lr"]:                      # запуск на відеокарті лише коли lr змінився (без зупинки)
            if p.device.type in ("xla", "tpu"):              # число як дані графа, а не константа — без перекомпіляції щокроку
                self._lr_t.copy_(torch.tensor(grp["lr"] * self.scale))
                self._lr_v.copy_(torch.tensor(grp["lr"] * self.scale * self._vmul))
            else:
                self._lr_t.fill_(grp["lr"] * self.scale)
                self._lr_v.fill_(grp["lr"] * self.scale * self._vmul)   # таблиці слів: «швидке закріплення» слів
            self._lr_last = grp["lr"]

    def _spec(self, p):
        """(вид кроку, чи є затухання ваг, чи це таблиця слів) — від форми ваги."""
        grp = self.param_groups[0]
        vocab_table = p.dim() == 2 and p.shape[0] >= 10000
        kind = "vocab" if vocab_table else ("row3" if self.norm == "row" and p.dim() == 3 else
                                           (self.norm.replace("hs", "hsn") + "e") if self.norm.startswith("hs") and self.norm != "hs" and p.dim() == 3 else
                                           self.norm.replace("hs", "hsn") if self.norm.startswith("hs") and self.norm != "hs" and p.dim() == 2 else
                                           "hs3" if self.norm == "hs" and p.dim() == 3 else
                                           (f"hsS{p._shard_W},{p._rows_total}" if getattr(p, "_sharded", False) else "hs")
                                           if self.norm == "hs" and p.dim() == 2 else
                                           "row" if (self.norm in ("row", "row_old") or self.norm.startswith("hs")) and p.dim() >= 2 else "tensor")
        wd = bool(grp["weight_decay"]) and p.dim() >= 2 and not vocab_table
        return kind, wd, vocab_table

    @torch.no_grad()
    def _update_eager(self, p):
        i = self.idx[id(p)]
        assert self.w[i] is None and self.m_dtype == torch.float32, "bf16-ваги / bf16-інерція — лише зі злитим кроком"
        grp = self.param_groups[0]
        grad = p.grad.float()
        m = self.m[i].mul_(self.beta).add_(grad, alpha=1 - self.beta)
        vocab_table = p.dim() == 2 and p.shape[0] >= 10000
        if vocab_table:
            rms = m.pow(2).mean(1, keepdim=True).sqrt()
            upd = torch.where(grad.abs().sum(1, keepdim=True) > 0, m / (rms + 1e-8), torch.zeros_like(m))
        elif self.norm == "row" and p.dim() == 3:
            # експерти (E, вхід, вихід), множаться як x @ w: «рядок» nn.Linear (ваги одного вихідного нейрона) тут —
            # стовпець; масштаб на кожен вихідний нейрон кожного експерта, а не один на весь експерт
            upd = m / (m.pow(2).mean(1, keepdim=True).sqrt() + 1e-8)
        elif self.norm in ("row", "row_old") and p.dim() >= 2:          # row_old — до виправлення (для порівняння)
            upd = m / (m.pow(2).flatten(1).mean(1).sqrt().view(-1, *[1] * (p.dim() - 1)) + 1e-8)
        else:
            upd = m / (m.pow(2).mean().sqrt() + 1e-8)
        self.rel[i] = grad.norm() / (p.norm() + 1e-8)
        eff = 1 + (self.g[i] - 1) * self.stake
        if grp["weight_decay"] and p.dim() >= 2 and not vocab_table:
            p.mul_(1 - grp["lr"] * grp["weight_decay"])
        p.sub_((upd * (grp["lr"] * self.scale * eff)).to(p.dtype))

    @torch.no_grad()
    def finish(self):
        if _OPT_FN:
            self.n_step += 1
            fn = _finish_impl if _NO_COMPILE else _compiled.get("finish")
            if fn is None:
                fn = _compiled["finish"] = torch.compile(_finish_impl, dynamic=False)
            fn(self.s, self.g, self.rel, self.gain, self.oh, float(self.top), float(self.bad), float(self.good),
               float(self.relax))
            return
        self.n_step += 1
        s, gk = self.s, self.g.clone()
        culprit = (self.rel >= self.rel.quantile(1 - self.top)).float()
        gk += torch.where(s < 0, culprit * self.bad * (-s * 10).clamp(max=1), torch.zeros_like(gk))
        gk += torch.where(s > 0, self.good * (s * 10).clamp(max=1) * (1 - gk), torch.zeros_like(gk))
        gk += self.relax * (1 - gk)
        gk = gk.clamp(0.5, 3.0)
        self.gain.add_(self.oh[:, None] * (gk / gk.mean() - self.gain))

    @torch.no_grad()
    def swap(self, old, new, slot_map):
        """Експерти переїхали між банками (BankMoE.rebank): новий параметр займає місце старого (підсилювачі
        інстинкту — ті самі, бо це той самий тензор за роллю), інерція тих, хто лишився вчитись, — по місцях."""
        i = self.idx.pop(id(old))
        self.plist[i] = new
        self.idx[id(new)] = i
        m = torch.zeros_like(new, dtype=self.m_dtype)
        w = new.detach().float().clone() if new.dtype == torch.bfloat16 else None
        for j, s in enumerate(slot_map):
            if s is not None:
                m[j] = self.m[i][s]
                if w is not None and self.w[i] is not None:
                    w[j] = self.w[i][s]                     # точна копія того, хто лишився вчитись
        self.m[i], self.w[i] = m, w

    @property
    def last(self):
        return dict(card=self.k, s=self.s, stake=self.stake)
