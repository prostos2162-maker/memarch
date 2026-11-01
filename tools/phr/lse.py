"""Для здивування від словосполучень: logsumexp прогнозу пар для кожного попереднього слова (V чисел)."""
import torch
ph = torch.load(f"_tmp/phr/bigram{__import__('os').environ.get('TAG','')}_d512.pt")
E, H, lp = ph["emb"].cuda(), ph["head"].cuda(), ph["logp_next"].cuda()
En = E / E.pow(2).mean(1, keepdim=True).sqrt().clamp(min=1e-6)
c = 0.05
lse = torch.cat([torch.logsumexp(c * En[i:i + 2048] @ H.T + lp, -1) for i in range(0, E.shape[0], 2048)])
ph["lse"] = lse.cpu()
torch.save(ph, f"_tmp/phr/bigram{__import__('os').environ.get('TAG','')}_d512.pt")
print("ok", lse[:5].tolist())
