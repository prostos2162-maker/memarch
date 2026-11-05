- prims.rmsnorm — за замовчуванням (x.f32·rsqrt(mean+eps)·w.f32).astype(bf16), один cast; прапор RMS_CAST_FIRST/cast_first=True дає варіант «cast, потім ·w у bf16» — злитий nn.RMSNorm torch 2.14 на CPU побітово рахує саме так (перевірено); на TPU (питання 3) підтвердити дампом і за потреби перемкнути прапор.
- prims.linear — у bf16: dot_general з f32-накопиченням і bf16-виходом (емуляція autocast); у f32 — Precision.HIGHEST для точного паритету з torch CPU.
- attn.splash — масштаб 1/sqrt(hd) множиться на q до splash (у JAX-splash масштаб не вбудований); маска LocalMask((T,T),(window-1,0)) = вікно window ключів із собою. На CPU не перевірено (лише dense_ref); паритет splash↔dense — тест на TPU.
- attn.dense_ref — softmax у f32, P·V у f32, вихід cast до dtype v; та сама маска j≤i, i-j<window і при T≤window (= каузальна SDPA torch).
- instinct — SR лічильниковий (хеш lowbias32 як у Pallas, насіння = seed_t·1000003; generic-види hsS/tensor/row беруть насіння hash(seed_t_start·1000003 ^ 0x5BD1E995·(i+1)) без кроку лічильника); cfg.sr=False — до найближчого — у torch generic SR через torch.randint, невідтворюваний; потрібна детермінованість і тести
- instinct — bf16-вага без SR: обчислення в fp32 і ОДНЕ округлення (w·(1−lr·wd) − upd·lr·eff → bf16) — у torch eager-шлях з bf16 без копії округлює двічі (mul_ і sub_), це випадковість
- instinct — kind за іменами (emb/head/l_head1/book → vocab; 3D → hs3; 2D → hs, light-лінійні через predicate/kinds → hsS; ≥4D → row; інакше tensor) — design §3, не shape[0]≥10000
- instinct — hsS лишається як у torch: свіжа норма стовпця щокроку без lag (sum/rows_total, psum за axis_name при W>1) — паритет; перехід на hs1-lag — окремим рішенням (§9 п.5)
- instinct — (1−β) у jnp-шляху — float64-скаляр → f32 (як torch scalar), а в Pallas — 1−β у f32 всередині ядра (як torch_tpu з PALLAS_OPT) — побітовий паритет з відповідним torch-шляхом
- instinct — has_grad: статична маска в spec (+ grad None) — вага пропускається цілком (без згасання m, wd, rel), як update_all sel
- ce/surprise/induct — sigmoid на bf16-параметрах (mix_w, ind_in_g) рахується у fp32 і округлюється раз до bf16 — так робить torch (CPU-перевірено, побітово); jax.nn.sigmoid прямо на bf16 давав інші біти
- induct — induct_raw лише на хості (numpy int64, CPU_PREP), не в графі — ключі трійок потребують int64; induct_in — gather-форма замість (B,T,3,T) bmm (fwd побітово, grad — інший порядок сумування)
- induct — стат mix_g рахується у fp32 (torch з bf16 g сумує в bf16 — випадковість, лише статистика)
- induct — lock_w лише через stop_gradient у induct_val (заморожена, класти в consts)
- ce — chunk_ce завжди повертає VJP і для b (db = ΣG, як torch); у моделі b — сталі (sv_mask, prior0), тож градієнт відкидається
- anchors — select_anchors(..., quantile_hook=None): одночиповий torch-шлях; на W>1 передати lambda q: lax.pmean(q,'x') (рішення §10 «спільний tau»); tau тоді однаковий на всіх чіпах
- moe (M4) — RMSNorm: нормування і множення на bf16-вагу у fp32, один cast у кінці (а не cast→mul у bf16, як у design §3) — так torch nn.RMSNorm на CPU (opmath) збігається побітово (bf16 forward bank_moe = 0 різниці); на TPU підтвердити дампом (питання 3).
- moe (M4) — ErrProbe/err, rebank, wave, hint, budget-гілка викинуті — мертві під цільовим env (WAVE_WARM=1e6, hint_w=0, budget=False), на loss/grad не впливають.
- moe (M4) — на CPU-бекенді JAX bf16×bf16 dot з f32-накопиченням підміняється f32×f32 (CPU-XLA не підтримує BF16xBF16=F32); добутки bf16 точні в f32, тож математика та сама; на TPU — рідний bf16 dot.
- moe (M4) — EP-обмін — змінний Exchange(send, recv): IDENTITY для W=1, a2a_exchange(axis,W) через lax.all_to_all (розкладка як all_to_all_single: експерт e на чипі e//El); a2a поки не тестований (M9).
- memory/reread — einsum-и пам'яті й перечитування (sc/sim/cos/mem/foc/put) рахуються у fp32 (HIGHEST), лінійні шари bf16 (MEM_BF16) — torch_tpu-поведінка autocast для einsum невідома; fp32 = очевидний задум (закладки >256 не квантуються). Емуляція torch CPU autocast (einsum -> bf16, крім згорток розміру 1) — прапор MemCfg.eb / RereadCfg.eb (MEM_EINSUM_BF16), лише для тестів паритету.
- memory/reread — розриви рівних у top_k/argsort/argmax: стабільні «менший індекс першим» (lax.top_k, argsort stable=True); у тестах torch topk/argsort підмінено стабільними.
- memory/reread — sigmoid на bf16 рахується у fp32 з одним округленням (як torch), бо jax.nn.sigmoid у bf16 дає інші значення на ~30% елементів і перемикає вибір карток.
- memory — 0-вимірний logsigmoid(dim_weight) додається в типі sc (torch не підвищує тип 0-вимірним тензором).
- memory — F.normalize через sqrt(max(sum x², eps²)): те саме значення, але без NaN-градієнта на нульових ланках лиць.
- memory — стан pos у int32 (torch int64); memory_pass додатково повертає st і діагностику шматків (upd/new/demote/promote/prune/dup).
- reread — scatter o_k назад: zeros.at[sel].set (замість one-hot bmm), hs_k — take_along_axis fp32 (точне збирання).
- тести memory/reread — XLA_FLAGS=--xla_allow_excess_precision=false: без нього XLA пропускає проміжні bf16-округлення, і вільний bf16-прогін розходиться з eager torch уже з 2-го шматка. Для TPU-продукції прапор не обов'язковий.
- model (M7) — consts (short_consts, lock_bucket, lock_w, ind_a) — статичні python/numpy, передаються замиканням, не аргументом jit (K/bucket/prior0 мають бути python-числами); lock_w/ind_a заморожені — не в params.
- model (M7) — induct_raw (prep) лише на хості (model.make_prep), як CPU_PREP; forward приймає готовий prep.
- model (M7) — стати wave/hint/err_r/move_r/learned/frozen не рахуються (мертві під цільовим env); rare_drop повертається завжди (у torch — лише eager).
- model (M7) — memory.gate.* отримує нульовий grad і в torch, і в JAX (MEM_LEAN) — лишаємо параметр заради ckpt; has_grad=True (torch grad — нулі, не None).
- config (M7) — додано поля mem_fresh/mem_dim/mem_dim_max/mem_write/mem_focus/mem_eb (типові = mem2.Memory за замовчуванням для Fv2moeK40).
- convert (M7) — torch_to_jax відкидає невикористані буфери (err, learned, up_snap, s_mean, ...); heavy.* складаються стеком по шару.
- instinct — make_spec(stacked=("heavy",)): складені по шару листки heavy (L,...) розгортаються на L окремих ваг heavy.{i}.{k}; вид — за формою без осі шару (norm→tensor, router→hs, up/down→hs3), свій rel/gain/cs на шар — інакше вид визначався з осі L (up/down→row, router→hs3, norm→hs з wd), а норми hs/hs3 змішували шари; torch має 14 окремих параметрів (знайдено на рев'ю).
- dist (M9) — Mesh(8, ('x',)); shard_map з check_vma=False; градієнти беремо ЗОВНІ shard_map від loss = mean_r(loss_r) (loss_r — нормований на свої B·T): jax.grad сам транспонує all_gather/psum/all_to_all, тож репліковані ваги отримують середнє (pmean), EP/vocab-шарди — суму внесків усіх чипів /W (= torch reduce-scatter SUM/W). Явного sync_grads немає.
- dist (M9) — розкладка: emb/head/l_head1/book — P('x',None) (V/W рядків); heavy.up/down — P(None,'x',...) (E/W експертів, moe.a2a_exchange); решта (light, memory, reread, router, норми, скаляри) — репліковано (design §9.6 (б)). book теж шардований (той самий VocabShard.take), хоч design допускав репліку.
- dist (M9) — VocabShard.take: номери слів на чипах різні → all_gather номерів, свої рядки для слів усіх чипів, psum_scatter (сума однієї ненульової частини — точна). take_same (psum без збирання) — лише для однакових на всіх чипах номерів (короткий словник sv_ids+скринька). Перша версія з голим psum змішувала рядки різних чипів — знайдено тестом (b).
- dist (M9) — VocabShard.ce (surprise, light_head_aux, рідкісна гілка short_logp): як shard.short_logp_vocab — all_gather h і y, chunk_ce по своїх рядках (зсув — свій зріз b), lse = logsumexp(all_gather(lse_loc)), ly = psum у власника. Наслідок: у bwd chunk_ce G = g·exp(lo − lse_loc) рахується з ЛОКАЛЬНИМ lse і округлюється до bf16 (як у torch shard) → градієнти vocab-таблиць відрізняються від W=1 до ~4e-3 відн. (виміряно 2.4e-3 для l_head1). Решта градієнтів — шум fp32 (≤2.7e-4; для порівняння W=1 з ce_chunk 512↔1024 дає 1.5e-3).
- dist (M9) — синхронний стан (§10): tau — quantile_hook=pmean (спільний), loss для Instinct = mean по чипах, ситуація = psum(emb рядків слів УСІХ чипів)/(W·B·T) (справжнє середнє), instinct.step виконується поза shard_map над глобальними масивами під jit → rel/норми по всьому тензору (XLA вставляє колективи). Pallas-ядра Instinct під таким jit на TPU ще не перевірені (на CPU тест — jnp-шлях, fp32).
- dist (M9) — не синхронне (свідомо): роутинг/місткість/черги/aux MoE і вибір якорів — на чип (місткість від локального N; маска якорів — від спільного СТАРОГО tau). Тому при різних даних на чипах loss_r = loss окремого W=1 прогону цього шарду (тест (b): збіг побітовий).
- dist (M9) — у ce/reread/model додано необов'язкові аргументи vocab/take/dist (None → поведінка W=1 без змін).
- step (M10) — стан = dict {params, opt, tau, step}, batch = {xy, prep, lr}; train_step = ОДНА jax.jit з donate_argnums=(0,) (весь state: params+opt+tau+step); lr рахується на хості (instinct.lr_at) і йде в batch як f32 — без перекомпіляції.
- step (M10) — порядок як torch step.py: value_and_grad → loss_mean(loss) і situation(params-до-кроку, xy) → grad_sync → instinct.step (prepare → update → finish). Ситуація W=1 = full_situation_xy (emb[x].f32.mean).
- step (M10) — інтерфейс для dist.py: Collectives(axis_name, rows_total, loss_mean, situation, grad_sync) + loss_fn(params, xy, prep, tau) (спільний tau — через свій loss_fn з anchors.quantile_hook) + wrap(f) (shard_map) до jit. Внутрішностей dist step.py не знає.
- remat (M10) — model.forward(remat=...) приймає bool (стара поведінка: True → heavy+memory), назву з model.REMAT_POLICIES або кортеж (light, heavy, mem). Назви: none / heavy (MoE-scan + сегменти пам'яті, nothing_saveable) / all (+ кожен light-блок) / dots_saveable (усе з checkpoint_policies.dots_saveable) / dots_nobatch (design §4: light — dots_with_no_batch_dims_saveable, heavy/mem — нічого). memory_pass/heavy_stack: remat=True або policy-функція (jax.checkpoint(body, policy=...)).
- test_step — порівняння grads за глобальною шкалою (max|Δ| / max|g| по всіх листках): memory.importance має grad ~1e-9 (шум порядку сумування jit↔eager, відносна різниця ~0.2); у порівнянні params після кроку jit↔eager ці листки виключені, бо Instinct нормує крок і знак шуму дає ±lr.
- data (M11) — data.py — порт data/loader.py (а не імпорт: на TPU-VM шлях D:\ai\memarch відсутній): та сама логіка, побітово ті самі батчі (тест проти torch-лоадера, 400 батчів, 2 епохи); EOS — поле Corpus (не глобальна), вихід int32, кеш у tempdir. HostLoader — префетч у треді + prep_fn (data.induct_prep = induct_raw на хості); у черзі кожен батч несе стан корпусу ПІСЛЯ себе → state_dict точний при будь-якій глибині префетчу.
- ckpt (M11) — тека з meta.json (шляхи листків, dtype, shape, порожні контейнери, data_state, extra) + arrays/ (orbax StandardCheckpointer або npz по листку; bf16 у npz як uint16-біти). backend="auto" → orbax, якщо імпортується (тут orbax-checkpoint 0.12.7). Запис у <dir>.tmp → os.replace (атомарно для spot). Порожні dict (напр. opt.cs без hs_lag) відновлюються явно (jax їх не бачить як листки). torch↔jax — ckpt.import_torch/export_torch поверх convert.py (лише params/frozen/tau; стан Instinct torch↔jax не конвертується).
- dist+step (M11) — dist.make_step: step.make_train_step з loss_fn = shard_map(train_loss) з dist.py → fwd+bwd (колективи всередині shard_map) + Instinct (поза shard_map, глобальні масиви) = ОДНА jit-програма з donate стану й remat. Ситуація й per-chip loss приходять з aux loss_fn (4-й елемент {"situation", "per_chip"}); step.py бере ситуацію звідти, якщо є (інакше situation_full). Старий dist.make_train_step лишається для test_dist (c).
- dist+step (M11) — jit з in_shardings/out_shardings = шардування стану після розкладки: вихідний state має ту саму розкладку, що й вхідний → крок 2+ без перекомпіляції (тест: компіляцій по кроках [1,0,…,0], _cache_size 1). opt будується jit(instinct.init)(шардовані params); листки, що не залежать від params (скаляри, keys, gain…), jit кладе на ОДИН пристрій — їх явно реплікуємо P() (інакше «incompatible devices»).
- dist (M11) — loss при B=1 на рядок і B=8 на одному чипі різний (на випадкових словах 11.42 проти 8.53): MoE-місткість/якорі/пам'ять залежать від N рядків чипа. W=8 по рядку на чип == середнє 8 окремих W=1 B=1 (|d| 7e-7) — це і є очікувана семантика §10; порівнювати W=8 з W=1 треба за тим самим B на чип.
- instinct (M12) — make_spec(frozen=...): torch кладе в plist InstinctOptB усі requires_grad параметри, включно з lock_w/ind_a (grad None) → P=63, а не 61. P впливає на finish (rel.quantile(1−top), gk.mean()) і на номери насінь SR. Додано заглушки frozen (leaf=None, has_grad=False) на місцях torch-порядку; без них gain розходився б з torch. Для справжнього тренування з torch-ckpt order/frozen треба брати з torch named_parameters (як у test_combined).
- test_combined (M12) — 20 кроків JAX vs torch, W=1, усі стовпи, fp32, без SR, V=10240 (≥10000 → torch бачить таблиці як vocab, види кроку torch==jax перевіряються). Teacher-forced (ваги torch на кожному кроці → jax grads): loss ≤1e-7 відн., усі grads ≤1e-3 відн., окрім memory.importance.* (|g| ~1e-8, відн. до 0.26 — шум порядку сумування). Вільний прогін: Instinct нормує крок, тож шум importance дає ±lr → інші рішення пам'яті → розбіжність росте з кроку 3 (loss 4e-4, ваги memory.focus_gate до 36%). Тому перевірка — з обнуленими grads memory.importance/gate в ОБОХ (jax: Collectives.grad_sync, torch: grad.zero_()): loss ≤7.1e-5, tau ≤1.5e-4, s ≤5e-5, картка k збігається всі 20 кроків, ваги ≤3.3e-2 (focus_gate; кроки 0–3 ≤5e-4). Висновок для Стефана: memory.importance фактично вчиться шумом (і в torch).
- bench_tpu (M12) — ваги ініціалізуються одразу шардованими (jit(init_params, out_shardings=param_specs від eval_shape)) — 24B параметрів не проходять через хост; кроки виконуються через lowered.compile() (час компіляції окремо, без другої компіляції в кеші jit). Pallas Instinct — лише на TPU і bf16 (на CPU smoke — jnp). MFU рахується від STEP_FLOPS=4.4e9/tok і піку v6e 918 TF/чип.

## review M9–M12 (адверсарний)
- **Pallas-ядра Instinct під 8-чиповою jit — через shard_map.** instinct.step працює поза shard_map над глобальними
  масивами; Pallas/Mosaic custom call GSPMD не розбиває, тож на TPU шардовані листки (emb/head/l_head1/book за
  рядками, heavy.up/down за експертами) XLA зібрав би повністю на кожен чип (24B → OOM, 8× роботи). На CPU це не
  видно (interpret = звичайні jnp-операції). Тепер `instinct._pallas_call` при `shard=(mesh, {leaf: spec})`
  (`step.Collectives.shard`, ставить `dist.make_step`) виконує ядро в shard_map на своєму шматку. Точно, бо ядра
  локальні по шардованій осі (vocab — рядок, hs3 — кожен експерт). Насіння SR змішане з номером чипа (індекс у
  хеші локальний), тому біти SR при W>1 інші, ніж W=1 — паритет лише при W=1 (§10). Тест: tests/test_dist_pallas.py.
- **Ядра pallas_instinct.py / pallas_hs1.py скопійовано в пакет** (memarch_jax/), бо на TPU-VM немає D:\ai\memarch:
  без цього bench_tpu.py падав би на ImportError при трасуванні кроку з use_pallas. Оригінал — запасний шлях.
  Копію треба оновлювати, якщо ядра змінюються в memarch.

## Пам'ять HBM на v6e-8 (Fv2moeK40h14d2560, T=8192) — 2026-11-05
Замір: AOT-компіляція повного кроку (`jit_local` = step.make_local_step.local: fwd+bwd+Instinct) на 8 CPU-пристроях
(`--xla_force_host_platform_device_count=8`), абстрактний стан із шардуваннями dist.param_specs, splash замінено
двійником із тими самими залишками, CPU-підняття bf16→f32 вимкнено. Аргументи (params + m, bf16) = 6.73 GiB/чип.
temp_size_in_bytes, GiB/чип (jnp-Instinct / Pallas-Instinct):

| конфіг               | було          | +CE блоками | +Instinct fori | remat all (усе разом) |
|----------------------|---------------|-------------|----------------|-----------------------|
| B=1 remat heavy      | 19.08 / 17.29 | 18.21       | 12.60 / 13.71  | —                     |
| B=1 remat all        | 15.47         |             |                | 7.38 / 8.38           |
| B=2 remat heavy      | 29.55 / 27.91 |             | 19.07 / 20.12  |                       |
| B=4 remat heavy      | 50.52         |             | 32.46 / 33.43  |                       |
| B=4 remat all        | 37.46         |             |                | 14.50 / 15.47         |
| B=4 dots_nobatch     |               |             | 24.19 / 25.35  |                       |
| B=8 remat all        |               |             |                | 23.83 (jnp)           |

Разом з аргументами: B=1 all ≈ 14.1–15.1 GiB, B=4 all ≈ 21.2–22.2 GiB, B=4 heavy ≈ 39 GiB (не влазить), B=8 all ≈ 30.6 (впритул).
Рекомендовано: `--remat all` (B=1 і B=4). TPU-цифра 49.03G не відтворена на CPU — перевірити дампом на VM.

- **ce.chunk_ce — блоки слів (TOK=8192).** Було: логіти (W·N, rows) fp32 для ВСІХ зібраних слів одразу
  (здивування: f32[65528·B, 16416] = 4 GiB·B; light_head_aux і рідкісні — ще 1–1.25 GiB·B). Тепер при N > TOK
  fwd — lax.map по блоках, bwd — lax.scan (dW/db — fp32-накопичення в carry, bf16-округлення dW один раз, як було
  в одному Gbᵀ@hb). Хвіст доповнено нулями з g=0. Шматок словника НЕ зменшували (змінило б двоетапний lse).
  ly/dh побітово, lse ≤2 ulp fp32 (тіло lax.map компілюється окремо — інше злиття exp/суми). Тест:
  test_heads.test_chunk_ce_token_blocks. При N ≤ TOK — старий код без змін.
- **instinct.update — складені шари (heavy) через fori_loop під jit.** Було: розгорнутий цикл p[li] … .at[li].set
  на (L,E,·,·) листках → XLA тримав копії всього heavy m/params (copy-of-copy, 1.28 GiB кожна). Тепер блок
  L×K ваг (li-major, однаковий крок номерів у spec) — fori_loop з carry (p, m, cs, rel, seed_t) і
  dynamic_update_index_in_dim (in-place). Номери rel/gain, послідовність насінь SR і хеш (0x5BD1E995·(i+1)) mod 2³²
  для traced i — ті самі. Лише під jit (traced): eager-шлях старий, бо eager-по-одній і скомпільоване тіло
  різняться на 1 ulp; під jit — побітово = окремим вагам (test_instinct.test_stacked_heavy_jit, jnp і Pallas).
  −5.6 GiB/чип на B=1. Також головний підозрюваний TPU-only 49 GiB (копії складених листків довкола shard_map Pallas).
- **remat light:** «heavy» лишає 6 легких блоків без перерахунку (~3.6 GiB на рядок батча). `--remat all`
  точний (тест test_step.test_remat_policies: |dloss|=0) — рекомендований для B≥1 на v6e-8.
- Не робили (запас): offload Instinct m у pinned_host (−3.4 GiB аргументів), bf16-котангенс у light bwd (~0.6 GiB/рядок).

## 05.11 MoE: швидкий шлях обміну (moe.py, MoECfg.fast / a2a_chunks)
- **fast=True (за замовч.):** one-hot dispatch/combine einsum одразу в bf16 (на TPU `preferred_element_type=bf16`: одне
  округлення fp32-накопичення = ті самі біти; на CPU — fp32+astype, буквально старе); a2a над (W,El,c,d) без
  transpose(1,0,2,3), експерти — `einsum('wecd,edh->wech')`. CPU W=8: fwd+grad **побітово** = старий шлях (f32 і bf16).
  Старий шлях — `fast=False`.
- **a2a_chunks K (за замовч. 1):** K незалежних ланцюжків по групах G (dispatch→a2a→експерти→a2a→combine) для перекриття.
  fwd і dX побітово; dW експертів = сума K часткових згорток (кожна округлена до bf16) → ≤1 bf16-ulp на елемент. Тому
  дефолт 1; вмикати K=2..4 лише після TPU-заміру. `a2a_exchange(..., impl='ppermute')` — кільце collective-permute
  (асинхронний на TPU), побітово = all_to_all.
- checkpoint_name: `moe_disp`, `moe_a2a_in`, `moe_a2a_out` (обидва шляхи). `heavy_stack(..., stats=dict)` → `moe_fill`.
- Тест: tests/test_moe_ep.py (CPU потребує `--xla_allow_excess_precision=false`, інакше XLA сам викидає bf16-округлення).

## 2026-11-05 — VocabShard.take через all_gather таблиці; ce через all_to_all (dist.py, ce.py)
- `take_mode="ag"` (типово, cfg.vocab_take; "masked" — старий шлях): fwd = all_gather (V,D) + локальний gather N рядків
  замість gather W·N рядків + RS 1.34 GB (7/8 нулі). bwd — custom_vjp = linear_transpose СТАРОГО шляху → градієнт побітово
  той самий; fwd теж побітово (копія рядків). Тимчасово +V·D bf16 на чип (emb ≈ +588 MB). Очікувано −25…30 мс (RS 20 + gather 9.6 − AG ~3).
- `ce_a2a=True` (cfg.vocab_ce_a2a): lse — all_to_all свого шматка (W,N) замість all_gather (W,W·N); логіт слова — psum_scatter
  замість psum+slice. Побітово (ті самі числа, той самий порядок; сума «1 ненульова + нулі» точна). Дрібний виграш (~1–3 мс).
- hg у ce позначено checkpoint_name("ce_h_gather") — політика remat може зберегти, щоб bwd не повторював all_gather.
- ce.py: dh у scan bwd chunk_ce кастується в dtype h усередині тіла — без fp32-стеку (nb,TOK,d); побітово.
- A/B CPU W=8 (masked vs ag+a2a): виходи і всі градієнти побітово рівні. Тести dist/dist_step/heads/model/step/combined/dist_pallas — OK.

## 05.11 — remat-політики, rope-розкладка, Instinct rel (агент light/model/instinct)
- `dots_nobatch`=`all` (824 ms): policy ДОХОДИЛА до легких блоків (print_saved_residuals: зберігались f32-виходи 5 dot-ів), але splash (custom call, не dot) і решта блоку все одно перераховувались, а heavy/mem були жорстко True. Плюс bench_tpu мовчки перетворював невідомі назви в кортежі `--remat a,b,c` на False — виправлено (будь-яка назва policy, перевірка до компіляції).
- Нові назви remat: `moe_save` (light: pre_attn/post_attn окремими checkpoint, splash поза remat → його залишки q,k,v,out,lse зберігаються; мітки `light_o`,`light_up`,`attn_out`; heavy: save_only_these_names(moe_a2a_in, moe_a2a_out, moe_disp); mem: нічого), `moe_save_dots` (+dots без batch-вимірів), `moe_save_heavy` (light без remat), `dots_nobatch_all`. Старі назви не змінено (A/B).
- Мітку `light_o` ставимо на вихід o-проєкції, а не на суму x+o: на суму зберігання змінювало біти градієнта (XLA прибирає bf16-округлення проміжку у злитому перерахунку). Зараз усі нові політики побітово = без remat (легкий блок, CPU).
- `light.ROPE_BTHD=True`: rope у (B,T,H,hd) до транспозиції. Fwd побітово; bwd на CPU — до шуму порядку XLA-злиття (rel 2.7e-4 у bf16, як і між remat-варіантами). `False` — старий шлях.
- `InstinctCfg.rel_cond=True`: норми rel під lax.cond лише на кроках want_rel. A/B побітово (7 кроків, rel_every=3).

## 05.11 Інтеграція трьох агентів (moe / vocab / remat)
- Проводка: config.moe_fast / moe_a2a_chunks / moe_a2a_impl / vocab_take / vocab_ce_a2a; model.moe_cfg передає fast+a2a_chunks, heavy_stack(stats=stats) → moe_fill у метриках; make_dist передає impl. bench_tpu: --moe_chunks --moe_impl --moe_old --vocab_take --no_ce_a2a --rope_old (A/B).
- Імена checkpoint moe_a2a_in/moe_a2a_out/moe_disp є в moe.py (обидва шляхи) — політика moe_save реально зберігає (CPU smoke: FLOP/крок 9.64e9 heavy → 9.18e9 moe_save).
- Усі 15 тестів у tests/ пройшли (8 CPU). Smoke bench small: all/heavy/moe_save/moe_save_heavy/moe_save+ppermute K=2 — ок, втрати однакові.
- AOT memprobe (CPU, песимістичний: heavy B=4 показує 32.2 GiB temp, хоча на TPU працює) temp+arg GiB/чип: B4 heavy 38.9, B4 dots_nobatch 30.0, B4 moe_save 27.4; B8 heavy 65.9, B8 dots_nobatch 48.8 (на TPU працював), B8 moe_save 43.7. Висновок: moe_save дешевший за обидва робочі базові варіанти → B=4 і B=8 мають вміститися; перевірити на TPU.

## 2026-11-05 — пам'ять для B=5/6 з moe_save: offload Instinct m + економніші remat-назви

TPU v6e-8 B=4×8192 moe_save: 690 мс, 380k tok/s, args 6.74 + temps 20.38 ≈ 27.1 GiB/чип; B=8 — OOM.
CPU AOT (memprobe, 8 пристроїв; для moe_save відкалібровано: B4 27.4 проти TPU 27.1):

| варіант | args | temps | разом |
|---|---|---|---|
| B4 moe_save | 6.73 | 20.65 | 27.4 |
| B5 moe_save (інтерп.) | 6.73 | ≈25.1 | ≈31.9 |
| B6 moe_save | 6.73 | 29.60 | 36.3 |
| B5 moe_save_out | 6.73 | 22.35 | 29.1 |
| B6 moe_save_out | 6.73 | 26.14 | 32.9 |
| B6 moe_save_a2a / _disp / _lite | 6.73 | 29.60 | 36.3 (нуль виграшу) |

- `--offload_opt` (dist.make_step(offload_opt=True), step.make_local_step(offload=...)): Instinct m у
  `pinned_host`; у jit `device_put(m, sh.with_memory_kind("device"))` після grad_sync, назад — після оновлення.
  m = 3.36 GiB/чип. CPU memory_analysis не розділяє хост/чип, але скомпільований HLO має всі листки m у S(5)
  (хост) на вході й виході → на TPU очікуємо args −3.36 GiB. Тест test_dist_step.test_offload_opt: loss і params
  біт-у-біт як без offload. Ризик: PCIe-трансфер 2×3.36 GiB/крок (~0.2-0.3 с, якщо не перекривається) —
  виміряти на TPU.
- remat `moe_save_out` (heavy зберігає лише moe_a2a_out): −3.46 GiB temps на B6 (≈−0.58/рядок). moe_a2a_in і
  moe_disp — той самий буфер (зберегти одну з них = зберегти обидві), тому `moe_save_a2a/_disp` = moe_save.
  Ціна: bwd перераховує dispatch+a2a_in (одна a2a на шар).
- donate/alias вже повний (alias 6.726 = output) — резерву немає. CE/словник (п.4) не чіпав.

## 05.11.2026 — MoE (moe.py): прапори під prof_b4_msh (env, бо config/bench — чужі файли)
* `MOE_UNROLL=1` (`MoECfg.unroll`): 14 важких шарів python-циклом замість `lax.scan`; ваги шарів — `lax.split`
  (транспозиція — concatenate). Залишки a2a_in/a2a_out — окремі буфери: прибирає scan-стек moe.py:286
  (DUS fwd 6.5 + DS bwd 10.5 + copy 3.5 ≈ 20 ms). Побітово те саме (fp32 і bf16, remat=True і moe_save, K=1/2).
* `MOE_UP_MID=e` (`MoECfg.up_mid`): проміжок експерта в розкладці (El,W,c,h) — як старий bmm (up на 84%),
  down повертає (W,El,c,d) для a2a. Побітово те саме. Спроба повернути fusion.2571 з 58%.
* Відкинуто: «fold_w» (вага місця в one-hot combine замість ye·w) — прибрав би broadcast_multiply+reduce_precision
  (moe.py:263-264), але НЕ побітово (k=4 доданки combine округлювались окремо) — не вмикати.
* Копії розкладки навколо a2a (≈19 ms) на CPU не перевірити — потрібен TPU-профіль з цими прапорами.

## 05.11.2026 — обруч (hoop) перенесено в основний репо
- `memarch_jax/hoop.py` (з memarch_hoop), `cfg.hoop*` у config.py (hoop=False за замовчуванням — модель без змін).
- model.py: при cfg.hoop голова обруча (на lh, stop-grad) передбачає власну помилку; top `n_slots(T, hoop_budget)`
  слів на послідовність → heavy_stack (замість якорів, tau не змінюється); `moe_cfg.fill=1` — місткість від B·A.
  loss: + hoop_w·MSE(ŝ, stop-grad nll суміші). Збережено fast MoE / remat moe_save* / stats=stats / rope.
- bench_tpu: `--hoop`, `--hoop_budget F` (0.125). Тест tests/test_hoop.py (W=8 EP == mean W=1, навчання, стиснення бюджету) — OK;
  smoke --cpu8 --remat moe_save_heavy --hoop — OK, 0 перекомпіляцій.
- Очікування (prof_b4_msh п.2: heavy ∝N ≈255 ms при частці 0.2): 0.125 → −96 ms ідеал, −58…−72 реально → ≈615–630 ms (≈415–425k tok/s).

## 2026-11-05 — VocabShard.take bwd: sort + segment_sum замість scatter-add
- dist.py: новий `VocabShard.take_bwd` ("sort" типовий, "scatter" — старий шлях через linear_transpose), прапорець cfg `vocab_take_bwd`.
- bwd: all_gather котангенса й номерів → stable argsort локальних номерів (чужі → сегмент rows, mode="drop") → segment_sum(indices_are_sorted=True). Порядок доданків у рядку той самий, накопичення в dtype градієнта.
- CPU W=8: побітово рівний старому (fp32 і bf16, зокрема V=64 з багатьма повторами). На TPU перевірити побітово/≤2 ulp; якщо ні — `vocab_take_bwd="scatter"`.
- Профіль: scatter emb 8.8 ms (10 GB/s) + book 6.1 ms (1 GB/s); очікування −5…8 ms/крок (686 → ~680). Перекриття AG котангенса / all-reduce не робив (XLA-планувальник; контролю без зміни інших файлів немає).

## 2026-11-05 — інтеграція hoop + moe(unroll/up_mid) + take(sort)
- Додано `cfg.vocab_take_bwd="sort"` у config.py (dist.py читав через getattr) і прапорець `bench_tpu --vocab_take_bwd sort|scatter`.
- Усі 16 тестів tests/ OK на 8 CPU (потрібен PYTHONIOENCODING=utf-8 при перенаправленні у файл, інакше cp1251 падає на «→/−/↔»).
- Smoke small B1 T256 moe_save_heavy: без hoop / hoop 0.125 / hoop+MOE_UNROLL=1+MOE_UP_MID=e+scatter — 0 рекомпіляцій; останні два дають бітово однакові loss.
- AOT пам'ять (CPU-оцінка, Fv2moeK40h14d2560 B=4 moe_save_heavy, temp): 36.24 GiB без hoop, 34.84 (hoop 0.125), 33.41 (0.0625); args 6.73 GiB. CPU завищує (TPU факт ~27 GiB) — дивитись на дельту: −1.4 / −2.8 GiB.
