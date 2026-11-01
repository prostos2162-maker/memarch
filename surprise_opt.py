"""Оптимізатор за несподіванкою (ARCHITECTURE.md, «Ідея 5»).

Основа — AdamW. Поверх нього крок масштабується тим, наскільки результат розійшовся з очікуванням:
  * очікування = ковзне середнє loss;
  * несподіванка s = (очікуване − поточне) / очікуване  (s > 0 — краще, ніж чекали);
  * хороша несподіванка («боявся, а вийшло круто»): крок ширший для всіх, × (1 + good·s), до ×max_good;
  * погана несподіванка («думав легко — впав»): крок точковий — посилюються лише `top` частка
    параметрів з найбільшим градієнтом (ті, що діяли в мить помилки), × (1 + bad·|s|), до ×max_bad;
  * без несподіванки — звичайний крок AdamW.
"""
import torch


class SurpriseAdamW(torch.optim.AdamW):
    def __init__(self, params, good=4.0, bad=6.0, max_good=2.0, max_bad=3.0, top=0.1, ema=0.98, **kw):
        super().__init__(params, **kw)
        self.good, self.bad, self.max_good, self.max_bad, self.top, self.ema = good, bad, max_good, max_bad, top, ema
        self.expect = None
        self.last_s = 0.0

    @torch.no_grad()
    def step(self, loss=None):
        if loss is None:
            return super().step()
        loss = float(loss)
        if self.expect is None:
            self.expect = loss
        s = (self.expect - loss) / max(self.expect, 1e-6)
        self.expect = self.ema * self.expect + (1 - self.ema) * loss
        self.last_s = s
        # запам'ятати ваги, зробити звичайний крок AdamW, а потім масштабувати саме зміну
        before = [[p.detach().clone() for p in g["params"] if p.grad is not None] for g in self.param_groups]
        super().step()
        for g, b in zip(self.param_groups, before):
            ps = [p for p in g["params"] if p.grad is not None]
            for p, p0 in zip(ps, b):
                delta = p - p0
                if s > 0:                                   # хороша: ширше для всіх
                    p.copy_(p0 + delta * min(self.max_good, 1 + self.good * s))
                elif s < 0:                                 # погана: точково — найвинніші параметри
                    a = p.grad.abs().flatten()
                    if a.numel() > 65536:                   # поріг «найвинніших» — за вибіркою (повне сортування дороге)
                        a = a[torch.randint(0, a.numel(), (65536,), device=a.device)]
                    thr = a.quantile(1 - self.top)
                    culprit = (p.grad.abs() >= thr).to(delta.dtype)
                    mult = 1 + culprit * (min(self.max_bad, 1 + self.bad * -s) - 1)
                    p.copy_(p0 + delta * mult)
