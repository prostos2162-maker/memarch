# «Думати на здивуваннях» у рецепті суперника: FFN шарів SURP_LO..SURP_HI-1 рахується лише для слів, яких
# маршрутизатор вважає важкими (поріг — ковзний квантиль, частка SURP_FRAC; стеля SURP_CAP слів; причинно).
_a = '''    def forward(self, x: Tensor, attn_args: AttnArgs):
        x = x + self.attn(norm(x), attn_args)
        x = x + self.mlp(norm(x))
        return x
'''
_b = '''    def forward(self, x: Tensor, attn_args: AttnArgs):
        x = x + self.attn(norm(x), attn_args)
        if not hasattr(self, "router"):
            return x + self.mlp(norm(x))
        xs = norm(x)[0]                                           # (T, d), батч 1 (упаковані документи)
        T = xs.size(0)
        r = self.router(xs).squeeze(-1).float()
        if self.training:
            with torch.no_grad():
                q = torch.quantile(r[: min(T, 16384)], 1 - SURP_FRAC)
                self.tau.mul_(0.9).add_(0.1 * q)
        K = max(16, int(T * SURP_CAP) // 16 * 16)
        sel = r > self.tau
        keep = sel & (sel.cumsum(0) <= K)
        pos = torch.arange(T, device=xs.device)
        idx = ((~keep).long() * T + pos).argsort()[:K]
        valid = keep[idx]
        y = self.mlp(xs[idx]) * (torch.sigmoid(r[idx]) * 2 * valid).to(xs.dtype)[:, None]
        return x + torch.zeros_like(x[0]).index_add(0, idx, y)[None]
'''
rep(_a, _b)
_a = '''        self.blocks = nn.ModuleList([Block(model_dim, head_dim, num_heads, i) for i in range(num_layers)])
'''
_b = _a + '''        if os.environ.get("SURP"):
            for i in range(SURP_LO, SURP_HI):
                blk = self.blocks[i]
                blk.router = CastedLinear(model_dim, 1)
                blk.router.weight.label = 'attn_gate'           # оптимізується як інші ворота (Muon-група)
                with torch.no_grad():
                    blk.router.weight.normal_(0, model_dim ** -0.5)
                blk.register_buffer("tau", torch.zeros(()))
'''
rep(_a, _b)
rep("class Block(nn.Module):\n", "SURP_FRAC, SURP_CAP = float(os.environ.get('SURP_FRAC', 0.25)), float(os.environ.get('SURP_CAP', 0.3))\n"
    "SURP_LO, SURP_HI = int(os.environ.get('SURP_LO', 2)), int(os.environ.get('SURP_HI', 14))\nclass Block(nn.Module):\n")
