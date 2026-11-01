# Робить nanogpt_pc.py: той самий рецепт modded-nanogpt Medium, але для одного ПК на Windows
# (без NCCL/FlashAttention): dist -> заглушка для 1 процесу, varlen_attn -> flex_attention з тією ж маскою
# (документи + причинність + ковзне вікно), розміри моделі й батча — з env.
src = open("nanogpt_medium_g4.py", encoding="utf-8").read()
def rep(a, b, n=1):
    global src
    assert src.count(a) == n, (a, src.count(a)); src = src.replace(a, b)
rep("import torch.distributed as dist\n", "import fake_dist as dist\n")
rep("from torch.nn.attention.varlen import varlen_attn  # гонка на G4: FA3 (лише H100) -> рівноцінний varlen PyTorch\n",
    "from torch.nn.attention.flex_attention import flex_attention, create_block_mask\n"
    "def varlen_attn(q, k, v, cu, cu2, mq, mk, scale, window_size):\n"
    "    T = q.size(0); W = window_size[0]\n"
    "    doc = torch.searchsorted(cu, torch.arange(T, device=q.device, dtype=cu.dtype), right=True)\n"
    "    def mm(b, h, qi, ki):\n"
    "        return (qi >= ki) & (qi - ki <= W) & (doc[qi] == doc[ki])\n"
    "    bm = create_block_mask(mm, None, None, T, T, device=q.device)\n"
    "    y = flex_attention(q.transpose(0, 1)[None], k.transpose(0, 1)[None], v.transpose(0, 1)[None], block_mask=bm, scale=scale)\n"
    "    return y[0].transpose(0, 1)\n")
rep('dist.init_process_group(backend="nccl", device_id=device)', "dist.init_process_group()")
rep("    num_layers=16,\n    num_heads=8,\n    head_dim=128,\n    model_dim=1024,",
    "    num_layers=16,\n    num_heads=int(os.environ.get('RACE_DIM', 1024)) // 128,\n    head_dim=128,\n    model_dim=int(os.environ.get('RACE_DIM', 1024)),")
# батч і довжина — поділені на RACE_BS_DIV (той самий розклад, менші числа)
rep("    train_bs_extension: int = 32 * 2048 * 8\n",
    "    train_bs_extension: int = 32 * 2048 * 8\n"
    "    def __post_init__(self):\n"
    "        d = int(os.environ.get('RACE_BS_DIV', 1))\n"
    "        self.train_bs_schedule = tuple(b // d for b in self.train_bs_schedule)\n"
    "        self.train_bs_extension //= d\n"
    "        self.val_batch_size //= int(os.environ.get('RACE_VAL_DIV', 1))\n")
exec(open("make_pc_ourval.py", encoding="utf-8").read())
exec(open("make_pc_ouropt.py", encoding="utf-8").read())
exec(open("make_pc_dump.py", encoding="utf-8").read())
open("nanogpt_pc.py", "w", encoding="utf-8").write(src)
print("ok")
