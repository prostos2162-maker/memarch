"""Пам'ять v3: обсяг комірок, а не кількість карток (ARCHITECTURE.md, «Розбіжність з ідеєю»).

Комірка — це фіксована кількість **шматочків** (свіжа: повний розмір, тьмяна: стиснений).
Картка — будь-яка кількість шматочків з одним лицем (спільний номер картки). Тож у комірку
влазить і 48 однакових карток, і 100 різного розміру: розмір картки — скільки в ній інформації.
  * новий факт про відому сутність (лице схоже на наявну картку) — **новий шматочок тієї ж картки**:
    картка росте; новий образ стає першою ланкою ланцюжка лиць на всіх її шматочках;
  * нова сутність — нова картка з одного шматочка;
  * найслабші свіжі шматочки опускаються в тьмяну (стиснення);
  * прибирання тьмяної: найслабший шматочок **зливається** з іншим шматочком своєї картки
    («два факти → суть»), а викидається лише тоді, коли зливати нема з чим.
Читання як у mem2: за лицем по всіх шматочках, тож велика картка дає більше деталей.
"""
import torch
import torch.nn.functional as F

from mem2 import Memory


class VolumeMemory(Memory):
    def init_state(self, B, device, dtype):
        st = super().init_state(B, device, dtype)
        C = self.Fn + self.Dmax
        st["card"] = torch.full((B, C), -1, dtype=torch.long, device=device)   # номер картки шматочка
        st["next_id"] = torch.zeros(B, dtype=torch.long, device=device)
        return st

    def write(self, h, positions, st, now, w_read=None):
        B, S, d = h.shape
        Fn, C = self.Fn, self.Fn + self.Dmax
        dev = h.device
        ar = torch.arange(B, device=dev)[:, None]
        n = self.norm_w(h)
        f = self.face(n)
        imp = torch.sigmoid(self.importance(n))
        c = self.content(n) * imp
        st = dict(st)
        if w_read is not None:
            used = w_read.float().sum(1).detach()
            st["uses"] = st["uses"] + used
            st["touched"] = torch.where(used > 0.5, torch.zeros_like(st["touched"]) + now, st["touched"])

        with torch.no_grad():
            fa = F.normalize(st["faces"], dim=-1)
            sim = torch.einsum("bsf,bclf->bscl", F.normalize(f, dim=-1).type_as(fa), fa).amax(-1)
            sim = sim.masked_fill(~st["valid"][:, None], -1).amax(-1)
            bright = (1 - sim.clamp(min=0)) * imp.squeeze(-1).float()
            W = min(self.W, S, Fn)                                       # не більше, ніж місць у свіжій
            cand = bright.topk(W, -1).indices
            cos = torch.einsum("bwf,bclf->bwcl", F.normalize(f[ar, cand], dim=-1).type_as(fa), fa).amax(-1)
            best_cos, best = cos.masked_fill(~st["valid"][:, None], -2).max(-1)
            upd = best_cos > self.match                                   # факт про відому сутність
            new_id = st["next_id"][:, None] + torch.arange(W, device=dev)
            cid = torch.where(upd, st["card"].gather(1, best), new_id)     # номер картки кожного кандидата
            score = self._score(st, now)
            # усі кандидати — нові шматочки у свіжій: спершу порожні місця, потім найслабші (опускаються)
            key_f = torch.where(st["valid"][:, :Fn], score[:, :Fn], torch.full_like(score[:, :Fn], -2e9))
            slot = key_f.argsort(-1)[:, :W]
            demote = st["valid"][ar, slot]
            key_d = torch.where(st["valid"][:, Fn:], score[:, Fn:], torch.full_like(score[:, Fn:], -2e9))
            dslot = key_d.argsort(-1).gather(1, (demote.long().cumsum(-1) - 1).clamp(min=0))
            if w_read is not None:
                heat = w_read.float().sum(1)[:, Fn:].masked_fill(~st["valid"][:, Fn:], 0)
                hot_v, hot = heat.max(-1)
                promote = hot_v > 1.0
                weak = key_f.argmin(-1)
            else:
                promote = torch.zeros(B, dtype=torch.bool, device=dev)
                hot = weak = torch.zeros(B, dtype=torch.long, device=dev)

        fw, cw, pw, bw = f[ar, cand], c[ar, cand], positions[ar, cand], bright.gather(1, cand)

        def put(t, idx, val, mask):
            oh = F.one_hot(idx, t.shape[1]).to(torch.float32) * mask[..., None].float()
            hit = oh.sum(1).clamp(max=1)
            oh = oh / oh.sum(1, keepdim=True).clamp(min=1)
            shp = (*hit.shape, *([1] * (t.dim() - 2)))
            if t.dtype == torch.bool or not t.is_floating_point():
                new = torch.einsum("bwc,bw...->bc...", oh, val.to(torch.float32))
                return torch.where(hit.view(shp) > 0, new.round().to(t.dtype), t)
            new = torch.einsum("bwc,bw...->bc...", oh.to(t.dtype), val.to(t.dtype))
            return t * (1 - hit.view(shp).to(t.dtype)) + new

        fc, dc = st["fc"], st["dc"]
        faces, pos, valid, brightS, uses, touched, card = (st[k] for k in
                                                          ("faces", "pos", "valid", "bright", "uses", "touched", "card"))
        meta = lambda: (faces, pos, valid, brightS, uses, touched, card)
        yes = torch.ones_like(upd)

        # 1) витіснені свіжі шматочки → тьмяна (стиснення)
        dfull = dslot + Fn
        dc = put(dc, dslot, self.down(fc[ar, slot]), demote)
        faces, pos, valid, brightS, uses, touched, card = [put(t, dfull, t[ar, slot], demote) for t in meta()]
        # 2) ланцюжок лиць: новий образ — перша ланка на ВСІХ шматочках картки
        old_face = faces[ar, best]                                        # (B,W,L,f) лице знайденої картки
        chain = torch.where(upd[..., None, None],
                            torch.cat([fw[:, :, None], old_face[:, :, :-1]], 2),
                            torch.cat([fw[:, :, None], torch.zeros_like(old_face[:, :, :-1])], 2))
        same = (card[:, None, :] == cid[:, :, None]) & upd[:, :, None] & valid[:, None, :]   # (B,W,C)
        last = (same.long() * (torch.arange(W, device=dev) + 1)[None, :, None]).amax(1) - 1  # останній кандидат
        has = last >= 0
        pick = chain[ar, last.clamp(min=0)]                               # (B,C,L,f)
        faces = torch.where(has[..., None, None], pick.type_as(faces), faces)
        touched = torch.where(has, torch.zeros_like(touched) + now, touched)
        # 3) усі кандидати — нові шматочки у свіжих місцях (з номером своєї картки)
        fc = put(fc, slot, cw, yes)
        faces = put(faces, slot, chain, yes)
        pos = put(pos, slot, pw, yes)
        valid = put(valid, slot, yes, yes)
        brightS = put(brightS, slot, bw, yes)
        uses = put(uses, slot, torch.zeros_like(bw), yes)
        touched = put(touched, slot, torch.zeros_like(bw) + now, yes)
        card = put(card, slot, cid, yes)
        next_id = st["next_id"] + W
        # 4) обмін: гарячий тьмяний шматочок ↔ найслабший свіжий
        p = promote[:, None]
        hf, wf = hot[:, None], weak[:, None]
        hot_c, weak_c = self.up(dc[ar, hf]), self.down(fc[ar, wf])
        fc, dc = put(fc, wf, hot_c, p), put(dc, hf, weak_c, p)
        swapped = []
        for t in meta():
            a, b = t[ar, wf], t[ar, hf + Fn]
            t = put(t, wf, b, p)
            swapped.append(put(t, hf + Fn, a, p))
        faces, pos, valid, brightS, uses, touched, card = swapped
        # 5) прибирання тьмяної: найслабші шматочки зливаються з іншим шматочком своєї картки
        with torch.no_grad():
            vd = valid[:, Fn:]
            over = vd.sum(1) > self.Dmax - self.W
            sc = (brightS + self.w_use * uses - self.w_age * (now - touched))[:, Fn:].masked_fill(~vd, 1e9)
            rank = sc.argsort(-1).argsort(-1)
            drop = over[:, None] & vd & (rank < (vd.sum(1, keepdim=True) - self.Dn))
            cd = card[:, Fn:]
            mate = (cd[:, :, None] == cd[:, None, :]) & vd[:, None, :] & ~drop[:, None, :]   # (B,from,to)
            mate = mate & ~torch.eye(self.Dmax, dtype=torch.bool, device=dev)[None]
            to = mate.float().argmax(-1)                                  # перший живий шматочок тієї ж картки
            merge = drop & mate.any(-1)
            self.last_merge = merge.float().sum(1).mean()                  # для діагностики
            self.last_drop = drop.float().sum(1).mean()
        oh = F.one_hot(to, self.Dmax).to(dc.dtype) * merge[..., None].to(dc.dtype)        # (B,from,to)
        add = torch.einsum("bft,bfd->btd", oh, dc)
        cnt = oh.sum(1)[..., None]
        dc = (dc + add) / (1 + cnt)                                       # «два факти → суть»
        valid = torch.cat([valid[:, :Fn], valid[:, Fn:] & ~drop], 1)
        return dict(fc=fc, dc=dc, faces=faces, pos=pos, valid=valid, bright=brightS, uses=uses,
                    touched=touched, card=card, next_id=next_id)
