# оцінка нашим протоколом (fastloop.val_loss): ті самі 20×8 вікон по 1024 з перших 2M токенів fineweb.npy,
# кожне вікно — окремий «документ» (як у нас: модель бачить лише вікно)
_a = '        print0(f"step:{step}/{train_steps} val_loss:{val_loss:.4f} train_time:{training_time_ms:.0f}ms step_avg:{training_time_ms/max(step, 1):.2f}ms", console=True)\n'
_b = _a + '''        if os.environ.get("OUR_VAL"):
            import numpy as _np
            _t = torch.from_numpy(_np.asarray(_np.load(os.environ["OUR_VAL"], mmap_mode="r")[:2_000_000], dtype=_np.int32)).cuda()
            _g = torch.Generator(device="cuda").manual_seed(0)
            _tot = 0.0
            with torch.no_grad():
                for _ in range(20):
                    _st = torch.randint(0, len(_t) - 1024 - 1, (8,), device="cuda", generator=_g)
                    _xy = _t[_st[:, None] + torch.arange(1025, device="cuda")]
                    _cu = torch.arange(0, 8 * 1024 + 1, 1024, device="cuda", dtype=torch.int32)
                    _tot += float(model(_xy[:, :-1].reshape(-1).int(), _xy[:, 1:].reshape(-1).long(), _cu, training_manager.get_forward_args()))
            print0(f"step:{step}/{train_steps} our_val:{_tot / 20:.4f}", console=True)
'''
rep(_a, _b)
