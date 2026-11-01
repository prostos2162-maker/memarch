"""Незалежна пам'ять (ARCHITECTURE.md, компоненти 2–8, 10–11).

Своя система, окремо від якорів-здивувань: сама вирішує, що для неї яскраве.

Пам'ять — набір карток фіксованого розміру:
  свіжа:  `fresh` карток, повний зміст;
  тьмяна: стиснений зміст і менша вага при читанні; розтягується до `dim_max`,
          а коли дотягнулась до межі — прибирання до `dim` за оцінкою.
Картка: зміст, ланцюжок лиць (`faces` останніх образів), закладка (позиція в
тексті), яскравість, скільки разів діставали, коли востаннє оновлювалась.

Текст іде шматками по `seg` слів. Слова шматка k читають пам'ять станом після
шматків < k (усе ближче видно через вікно легких шарів), тож ніщо не підглядає
в майбутнє. Після шматка — запис:
  * яскравість для пам'яті = новизна (лице не схоже на жодну наявну картку)
    × важливість (навчена: зміст картки множиться на неї, тож градієнт іде лише
    якщо картку потім реально дістають);
  * найяскравіші `write` слів шматка: якщо лице схоже на наявну картку — оновлюють
    її (новий образ стає першою ланкою ланцюжка, старий лишається), інакше —
    нова картка у свіжій пам'яті; найслабша свіжа опускається в тьмяну;
  * картку з тьмяної, яку активно дістають, піднімаємо у свіжу.
Факт до власника й відкрите питання виходять із того, що лице рахується зі
стану слова в контексті: у «Петро живе в будинку 3» стан «3» уже містить Петра,
тож його лице веде в картку Петра; у «У будинку 3 … живе Петро» стан «Петро»
уже містить «будинок 3». Тобто це вчиться, а не прописано вручну.
Фокус: `focus` карток, яких торкались останніми; слово читає фокус і пам'ять,
а навчений перемикач вирішує, скільки брати з фокуса.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def scale_grad(x, s):
    """Те саме значення, але градієнт × s (картку читають багато слів — градієнти не мають складатися в лавину)."""
    return x * s + x.detach() * (1 - s)


class Memory(nn.Module):
    def __init__(self, d, fresh=64, dim=192, dim_max=256, faces=2, d_face=64, d_dim=None,
                 write=32, focus=8, match=0.8, w_use=2.0, w_age=1e-3):
        super().__init__()
        self.d, self.Fn, self.Dn, self.Dmax = d, fresh, dim, dim_max
        self.nf, self.df, self.W, self.K = faces, d_face, write, focus
        self.match, self.w_use, self.w_age = match, w_use, w_age
        dc = d_dim or d // 4
        self.norm_w = nn.RMSNorm(d)
        self.face = nn.Linear(d, d_face, bias=False)
        self.content = nn.Linear(d, d, bias=False)
        self.importance = nn.Linear(d, 1)
        self.gate = nn.Linear(2 * d, 1)                       # скільки нового образу влити в картку
        self.down = nn.Linear(d, dc, bias=False)              # стиснення для тьмяної
        self.up = nn.Linear(dc, d, bias=False)
        self.norm_r = nn.RMSNorm(d)
        self.norm_c = nn.RMSNorm(d)                           # зміст картки перед читанням
        self.query = nn.Linear(d, d_face, bias=False)
        self.value = nn.Linear(d, d, bias=False)
        self.out = nn.Linear(d, d, bias=False)
        self.dim_weight = nn.Parameter(torch.tensor(-0.5))    # тьмяніння: менший вплив при читанні
        self.focus_gate = nn.Linear(2 * d, 1)

    # ---- стан ---------------------------------------------------------------
    def init_state(self, B, device, dtype):
        S = self.Fn + self.Dmax
        z = lambda *s, dt=torch.float32: torch.zeros(*s, device=device, dtype=dt)
        return dict(
            fc=z(B, self.Fn, self.d, dt=dtype), dc=z(B, self.Dmax, self.down.out_features, dt=dtype),
            faces=z(B, S, self.nf, self.df, dt=dtype),
            pos=z(B, S, dt=torch.long), valid=z(B, S, dt=torch.bool),
            bright=z(B, S), uses=z(B, S), touched=z(B, S),
        )

    def _contents(self, st):
        """Зміст усіх карток у повному розмірі (тьмяні — розгорнуті зі стиснення)."""
        return torch.cat([st["fc"], self.up(st["dc"])], 1)

    def _score(self, st, now):
        sc = st["bright"] + self.w_use * st["uses"] - self.w_age * (now - st["touched"])
        return sc.masked_fill(~st["valid"], -1e9)

    # ---- читання ------------------------------------------------------------
    def read(self, h, st, grad_scale=1.0):
        """h: (B,S,d) слова шматка. Повертає (додаток до h, вага кожної картки).
        grad_scale: кожен шматок передає назад у пам'ять лише свою частку градієнта, бо ту саму
        картку читають усі наступні шматки — інакше градієнти складаються в лавину."""
        B = h.shape[0]
        n = self.norm_r(h)
        q = F.normalize(self.query(n), dim=-1)                                      # (B,S,f)
        fa = F.normalize(scale_grad(st["faces"], grad_scale), dim=-1)
        sc = torch.einsum("bsf,bclf->bscl", q, fa).amax(-1) * math.sqrt(self.df)   # будь-яка ланка лиця
        dimw = F.logsigmoid(self.dim_weight)
        sc = torch.cat([sc[..., :self.Fn], sc[..., self.Fn:] + dimw], -1)
        sc = sc.masked_fill(~st["valid"][:, None], -1e4)
        w = F.softmax(sc.float(), -1).type_as(h) * st["valid"].any(-1)[:, None, None]
        vals = self.value(self.norm_c(scale_grad(self._contents(st), grad_scale)))  # (B,C,d)
        mem = torch.einsum("bsc,bcd->bsd", w, vals)

        # фокус: картки, яких торкались останніми
        k = min(self.K, st["touched"].shape[1])
        fidx = st["touched"].masked_fill(~st["valid"], -1e9).topk(k, -1).indices   # (B,K)
        fsc = sc.gather(-1, fidx[:, None].expand(-1, sc.shape[1], -1))
        fw = F.softmax(fsc.float(), -1).type_as(h) * st["valid"].gather(1, fidx).any(-1)[:, None, None]
        foc = torch.einsum("bsk,bkd->bsd", fw, vals.gather(1, fidx[..., None].expand(-1, -1, vals.shape[-1])))
        g = torch.sigmoid(self.focus_gate(torch.cat([n, foc], -1)))
        return self.out(g * foc + (1 - g) * mem), w.detach()

    # ---- запис --------------------------------------------------------------
    def write(self, h, positions, st, now, w_read=None):
        """h: (B,S,d) стан слів шматка (у контексті); positions: (B,S) позиції в тексті;
        w_read: (B,S,C) як слова шматка читали картки (для обліку й підйому з тьмяної)."""
        B, S, d = h.shape
        Fn = self.Fn
        ar = torch.arange(B, device=h.device)[:, None]
        n = self.norm_w(h)
        f = self.face(n)                                                            # (B,S,f)
        imp = torch.sigmoid(self.importance(n))                                     # (B,S,1)
        c = self.content(n) * imp                                                   # важливість іде через зміст
        st = dict(st)
        if w_read is not None:                                                      # облік: скільки разів діставали
            used = w_read.float().sum(1).detach()
            st["uses"] = st["uses"] + used
            st["touched"] = torch.where(used > 0.5, torch.zeros_like(st["touched"]) + now, st["touched"])

        with torch.no_grad():
            fa = F.normalize(st["faces"], dim=-1)
            sim = torch.einsum("bsf,bclf->bscl", F.normalize(f, dim=-1).type_as(fa), fa).amax(-1)
            sim = sim.masked_fill(~st["valid"][:, None], -1).amax(-1)              # (B,S)
            bright = (1 - sim.clamp(min=0)) * imp.squeeze(-1).float()               # новизна × важливість
            W = min(self.W, S)
            cand = bright.topk(W, -1).indices                                       # найяскравіші слова
            cos = torch.einsum("bwf,bclf->bwcl", F.normalize(f[ar, cand], dim=-1).type_as(fa), fa).amax(-1)
            best_cos, best = cos.masked_fill(~st["valid"][:, None], -2).max(-1)
            upd = best_cos > self.match                                             # оновлює наявну картку
            is_new = ~upd
            score = self._score(st, now)
            # нові → у свіжу: спершу порожні місця, потім найслабші свіжі (ті опускаються в тьмяну)
            key_f = torch.where(st["valid"][:, :Fn], score[:, :Fn], torch.full_like(score[:, :Fn], -2e9))
            slot = key_f.argsort(-1).gather(1, (is_new.long().cumsum(-1) - 1).clamp(min=0))
            demote = is_new & st["valid"][ar, slot]
            key_d = torch.where(st["valid"][:, Fn:], score[:, Fn:], torch.full_like(score[:, Fn:], -2e9))
            dslot = key_d.argsort(-1).gather(1, (demote.long().cumsum(-1) - 1).clamp(min=0))  # індекс у тьмяній
            # підйом: тьмяну картку, яку цього шматка активно діставали, міняємо з найслабшою свіжою
            if w_read is not None:
                heat = w_read.float().sum(1)[:, Fn:].masked_fill(~st["valid"][:, Fn:], 0)
                hot_v, hot = heat.max(-1)                                           # (B,)
                promote = hot_v > 1.0
                weak = key_f.argmin(-1)
            else:
                promote = torch.zeros(B, dtype=torch.bool, device=h.device)
                hot = weak = torch.zeros(B, dtype=torch.long, device=h.device)

        fw, cw, pw, bw = f[ar, cand], c[ar, cand], positions[ar, cand], bright.gather(1, cand)
        contents = self._contents(st)
        old = contents[ar, best]
        g = torch.sigmoid(self.gate(torch.cat([cw, old], -1)))
        new_val = old + g * (cw - old)                                              # уточнений образ

        def put(t, idx, val, mask):
            """Функційний запис t[b, idx] = val там, де mask. Через маску «хто куди пише»:
            вимкнені записи нікуди не цілять, тож градієнт не копіюється до них (без лавини)."""
            oh = F.one_hot(idx, t.shape[1]).to(torch.float32) * mask[..., None].float()   # (B,W,C)
            hit = oh.sum(1).clamp(max=1)                                                   # (B,C)
            oh = oh / oh.sum(1, keepdim=True).clamp(min=1)                                 # кілька в один слот — середнє
            shp = (*hit.shape, *([1] * (t.dim() - 2)))
            if t.dtype == torch.bool or not t.is_floating_point():
                new = torch.einsum("bwc,bw...->bc...", oh, val.to(torch.float32))
                return torch.where(hit.view(shp) > 0, new.round().to(t.dtype), t)
            new = torch.einsum("bwc,bw...->bc...", oh.to(t.dtype), val.to(t.dtype))
            return t * (1 - hit.view(shp).to(t.dtype)) + new

        fc, dc = st["fc"], st["dc"]
        faces, pos, valid = st["faces"], st["pos"], st["valid"]
        brightS, uses, touched = st["bright"], st["uses"], st["touched"]
        meta = lambda: (faces, pos, valid, brightS, uses, touched)

        # 1) витіснена свіжа → тьмяна (стиснення)
        dfull = dslot + Fn
        dc = put(dc, dslot, self.down(fc[ar, slot]), demote)
        faces, pos, valid, brightS, uses, touched = [put(t, dfull, t[ar, slot], demote) for t in meta()]
        # 2) нові картки у свіжих місцях
        fc = put(fc, slot, cw, is_new)
        chain0 = torch.cat([fw[:, :, None], torch.zeros_like(fw)[:, :, None].expand(-1, -1, self.nf - 1, -1)], 2)
        faces = put(faces, slot, chain0, is_new)
        pos = put(pos, slot, pw, is_new)
        valid = put(valid, slot, torch.ones_like(is_new), is_new)
        brightS = put(brightS, slot, bw, is_new)
        uses = put(uses, slot, torch.zeros_like(bw), is_new)
        touched = put(touched, slot, torch.zeros_like(bw) + now, is_new)
        # 3) оновлення наявних: зміст вливається за воротами, новий образ — перша ланка ланцюжка
        is_f, is_d = upd & (best < Fn), upd & (best >= Fn)
        fc = put(fc, best.clamp(max=Fn - 1), new_val, is_f)
        dc = put(dc, (best - Fn).clamp(min=0), self.down(new_val), is_d)
        chain = torch.cat([fw[:, :, None], faces[ar, best][:, :, :-1]], 2)
        faces = put(faces, best, chain, upd)
        pos = put(pos, best, pw, upd)
        brightS = put(brightS, best, torch.maximum(bw, brightS[ar, best]), upd)
        touched = put(touched, best, torch.zeros_like(bw) + now, upd)
        # 4) обмін: гаряча тьмяна ↔ найслабша свіжа (без умов на Python: крок має бути одним графом)
        if True:
            p = promote[:, None]
            hf, wf = hot[:, None], weak[:, None]
            hot_c, weak_c = self.up(dc[ar, hf]), self.down(fc[ar, wf])
            fc, dc = put(fc, wf, hot_c, p), put(dc, hf, weak_c, p)
            hfull = hf + Fn
            swapped = []
            for t in meta():
                a, b = t[ar, wf], t[ar, hfull]
                t = put(t, wf, b, p)
                swapped.append(put(t, hfull, a, p))
            faces, pos, valid, brightS, uses, touched = swapped
        # 5) прибирання тьмяної: дотягнулась до межі — лишаємо `dim` найсильніших
        with torch.no_grad():
            over = valid[:, Fn:].sum(1) > self.Dmax - self.W
            if True:                                                                 # без умов на Python
                sc = (brightS + self.w_use * uses - self.w_age * (now - touched))[:, Fn:]
                sc = sc.masked_fill(~valid[:, Fn:], -1e9)
                keep = sc.argsort(-1, descending=True).argsort(-1) < self.Dn
                valid = torch.cat([valid[:, :Fn], torch.where(over[:, None], valid[:, Fn:] & keep, valid[:, Fn:])], 1)
        return dict(fc=fc, dc=dc, faces=faces, pos=pos, valid=valid, bright=brightS, uses=uses, touched=touched)
