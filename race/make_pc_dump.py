# DUMP_TOK=шлях: на останній оцінці нашим протоколом — похибка кожного слова (для розбору «де суперник кращий»).
# Запис вмикає прапорець _DUMP_ON лише на прогоні без компіляції (інакше компілятор перекомпілює щовиклику).
_a = '                loss += F.cross_entropy(logits.view(-1, logits.size(-1)), target_seq.chunk(4)[i], reduction="mean")/4\n'
_b = ('                _ce = F.cross_entropy(logits.view(-1, logits.size(-1)), target_seq.chunk(4)[i], reduction="none")\n'
      '                if _DUMP_ON[0]:\n'
      '                    _DUMP.append(_ce.detach())\n'
      '                loss += _ce.mean()/4\n')
rep(_a, _b)
rep("class Block(nn.Module):\n", "_DUMP = []\n_DUMP_ON = [False]\nclass Block(nn.Module):\n")
_a = '            print0(f"step:{step}/{train_steps} our_val:{_tot / 20:.4f}", console=True)\n'
_b = _a + '''            if last_step and os.environ.get("DUMP_TOK"):
                _DUMP.clear(); _DUMP_ON[0] = True
                _g = torch.Generator(device="cuda").manual_seed(0)
                with torch.no_grad():
                    for _ in range(20):
                        _st = torch.randint(0, len(_t) - 1024 - 1, (8,), device="cuda", generator=_g)
                        _xy = _t[_st[:, None] + torch.arange(1025, device="cuda")]
                        model._orig_mod(_xy[:, :-1].reshape(-1).int(), _xy[:, 1:].reshape(-1).long(), _cu, training_manager.get_forward_args())
                _DUMP_ON[0] = False
                torch.save(torch.cat(_DUMP).cpu(), os.environ["DUMP_TOK"])
'''
rep(_a, _b)
