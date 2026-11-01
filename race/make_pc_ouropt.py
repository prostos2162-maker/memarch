# OUR_OPT=1: матриці шарів вчить НАШ оптимізатор (інерція + гомеостаз: рядок → стовпець → рядок) замість NorMuon;
# таблиці слів і скаляри — як у суперника (Adam). Усе інше в рецепті без змін.
_a = "class Block(nn.Module):\n"
_b = '''class HomeoOpt(torch.optim.Optimizer):
    """Інерція + гомеостаз (exp_opt.InstinctOptB, norm="hs"): кожен нейрон (рядок) і кожен вхід (стовпець)
    міняються однаково сильно; без ортогоналізації (жодних множень матриць)."""

    def __init__(self, params, lr=1e-3, weight_decay=0.1, momentum=0.9):
        super().__init__(params, dict(lr=lr, weight_decay=weight_decay, momentum=momentum))

    def reset(self):
        self.state.clear()

    @torch.no_grad()
    def step(self):
        for g in self.param_groups:
            beta = g["momentum"]
            for p in g["params"]:
                if p.grad is None:
                    continue
                st = self.state[p]
                if "m" not in st:
                    st["m"] = torch.zeros_like(p, dtype=torch.float32)
                lr = g["lr"] * getattr(p, "lr_mul", 1.0)
                _homeo_step(p, p.grad, st["m"], torch.tensor(lr, device=p.device), beta, g["weight_decay"])


@torch.compile(dynamic=False)
def _homeo_step(p, grad, m, lr, beta: float, wd: float):
    mf = m * beta + grad.float() * (1 - beta)
    m.copy_(mf)
    u = mf / (mf.pow(2).mean(1, keepdim=True).sqrt() + 1e-8)
    u = u / (u.pow(2).mean(0, keepdim=True).sqrt() + 1e-8)
    u = u / (u.pow(2).mean(1, keepdim=True).sqrt() + 1e-8)
    pf = p.float() * (1 - lr * wd) - lr * u
    p.copy_(pf.to(p.dtype))


class Block(nn.Module):
'''
rep(_a, _b)
_a = "        self.muon_opt = NorMuon(muon_params, lr=0.015, momentum=0.95, beta2=0.95, weight_decay=1.2)\n"
_b = _a + '''        if os.environ.get("OUR_OPT"):
            self.muon_opt = HomeoOpt(muon_params, lr=float(os.environ.get("OUR_LR", 1e-3)))
'''
rep(_a, _b)
