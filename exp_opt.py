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


def _instinct_step_impl(p, grad, m, w, g_i, stake, lr_t, beta: float, wd: float, kind: str, want_rel: bool):
    """Один крок інстинкту для однієї ваги (математика — як InstinctOptB._update_eager).
    w — точна fp32-копія ваги, якщо сама вага лежить у bf16 (тоді модель множить bf16 напряму, без щокрокового
    перетворення), інакше None. m — інерція (fp32 або bf16). Повертає |g|/|w| (want_rel) або 0."""
    grad = grad.float()
    mf = m.float() * beta + grad * (1 - beta)
    m.copy_(mf)
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
    elif kind == "hs3":
        upd = mf / (mf.pow(2).mean(1, keepdim=True).sqrt() + 1e-8)
        upd = upd / (upd.pow(2).mean(2, keepdim=True).sqrt() + 1e-8)
        upd = upd / (upd.pow(2).mean(1, keepdim=True).sqrt() + 1e-8)
    elif kind == "hs":
        # гомеостаз: і кожен нейрон (рядок), і кожен вхід (стовпець) міняються однаково сильно
        upd = mf / (mf.pow(2).mean(1, keepdim=True).sqrt() + 1e-8)
        upd = upd / (upd.pow(2).mean(0, keepdim=True).sqrt() + 1e-8)
        upd = upd / (upd.pow(2).mean(1, keepdim=True).sqrt() + 1e-8)
    elif kind == "row":
        upd = mf / (mf.pow(2).flatten(1).mean(1).sqrt().view(-1, *[1] * (p.dim() - 1)) + 1e-8)
    else:
        upd = mf / (mf.pow(2).mean().sqrt() + 1e-8)
    t = p if w is None else w
    rel = grad.norm() / (t.norm() + 1e-8) if want_rel else torch.zeros((), device=p.device)
    eff = 1 + (g_i - 1) * stake
    if wd:
        t.mul_(1 - lr_t * wd)
    t.sub_((upd * (lr_t * eff)).to(t.dtype))
    if w is not None:
        p.copy_(w)
    return rel


def _vocab_rows_impl(p, grad, m, w, rows, g_i, stake, lr_t, beta: float):
    """Таблиця слів / словосполучень: оновлюються лише задіяні в кроці рядки (повтори дають ті самі значення)."""
    g = grad[rows].float()
    mr = m[rows].float() * beta + g * (1 - beta)
    m[rows] = mr.to(m.dtype)
    upd = mr / (mr.pow(2).mean(1, keepdim=True).sqrt() + 1e-8)
    t = p if w is None else w
    t[rows] = t[rows] - (upd * (lr_t * (1 + (g_i - 1) * stake))).to(t.dtype)
    if w is not None:
        p[rows] = w[rows].to(p.dtype)


_compiled = {}


def _instinct_step(p, grad, m, w, g_i, stake, lr_t, beta, wd, kind, want_rel):
    """Окрема скомпільована копія на кожне поєднання (тип ваги, форма, dtype, з fp32-копією, з |g|/|w|, wd):
    у компілятора межа ~8 перекомпіляцій на функцію (у потоці зворотного проходу налаштування межі не діє),
    після неї він мовчки виконує крок поелементно — ≈12 кернелів на вагу замість 2–3."""
    import types
    key = (kind, tuple(p.shape), p.dtype, m.dtype, w is None, bool(want_rel), bool(wd))
    fn = _compiled.get(key)
    if fn is None:
        name = f"_inst_{len(_compiled)}"
        code = _instinct_step_impl.__code__.replace(co_name=name)
        fn = _compiled[key] = torch.compile(types.FunctionType(code, globals(), name), dynamic=False)
    return fn(p, grad, m, w, g_i, stake, lr_t, beta, wd, kind, want_rel)


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
        # bf16-ваги (експерти) — точна копія в fp32 живе тут; модель бачить лише bf16
        self.w = [p.detach().float().clone() if p.dtype == torch.bfloat16 else None for p in self.plist]
        self.k = torch.zeros((), dtype=torch.long, device=dev)
        self.s = torch.zeros((), device=dev)
        self.stake = torch.zeros((), device=dev)
        import os
        self.fused, self._lr_last = not os.environ.get("NO_COMPILE") and os.environ.get("OPT_FUSED", "1") == "1", None

    @torch.no_grad()
    def prepare(self, loss, situation):
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
        vocab_table = p.dim() == 2 and p.shape[0] >= 10000
        kind = "vocab" if vocab_table else ("row3" if self.norm == "row" and p.dim() == 3 else
                                           (self.norm.replace("hs", "hsn") + "e") if self.norm.startswith("hs") and self.norm != "hs" and p.dim() == 3 else
                                           self.norm.replace("hs", "hsn") if self.norm.startswith("hs") and self.norm != "hs" and p.dim() == 2 else
                                           "hs3" if self.norm == "hs" and p.dim() == 3 else
                                           "hs" if self.norm == "hs" and p.dim() == 2 else
                                           "row" if (self.norm in ("row", "row_old") or self.norm.startswith("hs")) and p.dim() >= 2 else "tensor")
        wd = bool(grp["weight_decay"]) and p.dim() >= 2 and not vocab_table
        if not hasattr(self, "_lr_t"):
            self._lr_t = torch.zeros((), device=p.device)
        if not hasattr(self, "_lr_v"):
            import os
            self._lr_v, self._vmul = torch.zeros((), device=p.device), float(os.environ.get("VOCAB_LR_MUL", 1))
        if self._lr_last != grp["lr"]:                      # запуск на відеокарті лише коли lr змінився (без зупинки)
            self._lr_t.fill_(grp["lr"] * self.scale)
            self._lr_v.fill_(grp["lr"] * self.scale * self._vmul)   # таблиці слів: «швидке закріплення» слів
            self._lr_last = grp["lr"]
        want_rel = self.n_step % self.rel_every == 0
        rows = getattr(self, "rows", {}).get(id(p)) if vocab_table else None
        if rows is not None:                                 # лише слова, потрібні на цьому кроці
            key = ("rows", tuple(p.shape), rows.numel(), p.dtype, self.m[i].dtype, self.w[i] is None)
            fn = _compiled.get(key)
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
