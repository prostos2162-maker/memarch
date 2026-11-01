"""Наскільки розмовник певен у наступному слові після кожного слова: ентропія прогнозу пар (V чисел)."""
import torch
ph = torch.load("_tmp/phr/bigram_d512.pt")
E, H, lp = ph["emb"].cuda(), ph["head"].cuda(), ph["logp_next"].cuda()
En = E / E.pow(2).mean(1, keepdim=True).sqrt().clamp(min=1e-6)
ent = []
for i in range(0, E.shape[0], 2048):
    lo = 0.05 * En[i:i + 2048] @ H.T + lp
    p = lo.softmax(-1)
    ent.append(-(p * lo.log_softmax(-1)).sum(-1))
ph["ent"] = torch.cat(ent).cpu()
torch.save(ph, "_tmp/phr/bigram_d512.pt")
print("ентропія: мін", ph["ent"].min().item(), "медіана", ph["ent"].median().item(), "макс", ph["ent"].max().item())
