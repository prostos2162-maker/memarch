# Дизайн порту memarch (Fv2moeK40h14d2560) з torch_tpu на JAX

## 0. Головні рішення

1. **Увесь крок виконується як одна програма `jax.jit(shard_map(step_local, mesh=('x',8)))`.** Модель пишемо в локальних формах одного чіпа, як у torch. Причина: torch-версія навмисно або випадково робить багато речей окремо на кожному чіпі:
   - tau у кожного чіпа свій;
   - loss у prepare береться локальний;
   - rel рахується за нормами локальних шардів;
   - routing і capacity MoE рахуються на кожному чіпі окремо;
   - вектор ситуації (situation) має свою особливість (див. питання 2).
   
   GSPMD з авто-шардингом усе це «полагодить» і тим самим змінить математику. З shard_map кожна така поведінка задається явно, а psum / all_to_all / all_gather стоять там, де вони були в torch.
2. **Pure-функції над pytree параметрів, без flax.** flax є, але не потрібен. Так checkpoint-конвертація зводиться до мапи імен 1:1 з torch `state_dict`.
3. **Pallas-ядра оптимізатора беремо напряму:** `pallas_hs1.instinct_hs1_2d/3d`, `pallas_instinct.instinct_vocab`. Вони вже є pure-JAX. Аліаси вмикаємо назад, `PALLAS_NO_ALIAS` не ставимо, плюс `donate_argnums`.
4. **Режим «точна копія torch» (bit-faithful) іде за замовчуванням.** Режим із виправленими особливостями (`FIX_*` прапори) додаємо як варіанти після того, як паритет доведено.

## 1. Структура пакета `D:\ai\memarch_jax\`

```
memarch_jax/
  config.py        # Cfg(dataclass, frozen): d=2560, heads=10, hd=256, E=40, heavy=14, light=2, final=2,
                   #   window=2048, seg=512, cap=.25, frac=.2, V=131328, K_short=8192, ce_chunk=16768,
                   #   rare_cap=.3, read_cost=.01, moe_k=4, cap_align=256, moe_group=512, W=8, ...
                   #   from_env(tpu_m23 ENV + G22) -> Cfg ; static consts (sv_ids, sv_pos, sv_mask,
                   #   sv_prior0, lock_bucket) з counts_fineweb.npy / lock_trust.pt (numpy)
  prims.py         # rmsnorm(x,w,eps=1.1920929e-7) ; linear(x,W) (bf16 in, f32 acc, bf16 out) ;
                   #   rope_half(x,T) ; relu2(x) ; normalize(x,eps=1e-12) ; sr_bf16(x,key)
  attn.py          # splash_local(q,k,v,window=2048) -> pallas splash (TPU) | dense_ref (CPU)
                   # flex_block(p, x, cfg) -> x
  ce.py            # chunk_ce(h,W,b,y,ch) custom_vjp -> (lse,ly) ; surprise(lh,lhead0,lhead1,x,cfg)
                   # short_logp_vocab(hn, W_shard, y, consts, cfg)   [двоетапний lse по 'x']
  induct.py        # induct_raw(x, lock_bucket) -> cand,on,bk,src (jnp sort, у графі або на хості)
                   # induct_in(h, emb_rows, on, src, lock_w, bk, ind_in_g) ; induct_mix(...)
  anchors.py       # select_anchors(s,tau,cfg,train) -> idx,valid,tau_new,stat
                   # reread_select(s,cfg) -> sel (nB,128)
  moe.py           # route(p,x,sa,cfg) ; dispatch_onehot(...) ; ep_ffn(xe,up,down) [lax.all_to_all]
                   # bank_moe(p, x, sa, cfg) -> x, aux, dropped
  memory.py        # init_state(B) ; mem_read(p,h,st) ; mem_write(p,h,qpos,st,now,w_read) ; put()
                   # memory_pass(p,h,cfg) -> reads, snaps   [lax.scan по 16 сегментах]
  reread.py        # reread_core(p, hs_k, x_ids, book, snaps, qpos_k, rows) -> o_k, cost
                   # reread_sep(p, h, s, x, snaps, cfg) -> o, cost
  model.py         # forward(params, consts, x, tau, train) -> hn, lh, s, aux_terms, stats, tau_new
                   # train_loss(params, consts, xy, tau) -> loss, (tau_new, stats)
  instinct.py      # InstinctState ; prepare(st, loss, sit, lr_ratio) ; finish(st, rel, s)
                   # update_param(kind, g, m, p, cs, sc, seed) [pallas | jnp generic]
                   # kinds.py: таблиця name -> kind (hs/hs3/hsS/vocab/tensor) + has_grad маска
  sched.py         # lr_at(step, cfg) float64 на хості -> (lr_t, lr_v, lr_ratio) f32
  shard.py         # mesh(), PartitionSpecs, pad/unpad рядків, sync_grads (семантика shard.py)
  step.py          # train_step = jit(shard_map(_step_local)), donate (params, opt, tau)
  data.py          # обгортка над fastloop CorpusData/GPUData (numpy) -> per-device xy (8,B,T+1)
  ckpt.py          # torch_to_jax(state_dict, cfg) / jax_to_torch(...) ; orbax save/restore
  tests/           # (див. §8)
  bench.py         # tok/s, профіль фаз, memory breakdown (згідно з правилом «діагностика з доказами»)
```

## 2. Параметри: pytree і шардинг

Mesh: `Mesh(devices.reshape(8), ('x',))`. Фізично v6e-8 має топологію 2x4, але одна вісь простіша: EP (40/8) і vocab-шард (131328/8) обидва ділять на 8.

### 2.1 Оцінка розмірів (bf16)

| група | параметрів | bf16 | шардинг |
|---|---|---|---|
| експерти 14×40×(2560·3840·2) | 11.0B | 22.0 GB, **2.75 GB/чіп** | EP: `P('x')` по осі експертів (5/чіп) |
| emb, head (131328×2560 кожен) | 672M | 1.34 GB, **168 MB/чіп** | `P('x',None)` рядки, 16416/чіп |
| l_head1 (V×128), book (V×128) | 34M | 67 MB | l_head1 у torch шардований (SHARD_MIN). Шардуємо `P('x')` |
| light×6 (qkv 19.7M, o 6.6M, up+down 26.2M) | 315M | 630 MB | див. 2.2 |
| memory + reread | ~21M | 42 MB | репліковані |
| router 14×(40×2560), norms, скаляри | ~1.5M | — | репліковані |

### 2.2 Обґрунтування через байти на крок

Рахуємо при B=1, T=8192, тобто 65k токенів на крок, а при 500k tok/s це ≈0.13 s на крок.

- **EP all_to_all.** Розмір xe: 40·Ct·d·2 B, де Ct=256 при N=2048 якорях. Це 52 MB на шар на напрям. Множимо на 4 (dispatch і combine, fwd і bwd) і на 14 шарів: ≈2.9 GB/крок/чіп, з них 7/8 іде мережею. Це 2.5 GB за 0.13 s ≈ 20 GB/s на чіп. ICI на v6e-8 це тримає. Зберігати експерти без шардування неможливо: 22 GB плюс m 22 GB.
- **emb/head.** Шардування обов'язкове з двох причин:
  1. Цього вимагає SHARD_VOCAB (двоетапний lse). Це питання чисел, а не лише пам'яті.
  2. all-gather 1.34 GB на кожен крок коштував би дорожче, ніж уся EP-комунікація.
- **light-лінійні (630 MB).** У torch вони row-sharded (FSDP, kind `hsS`). Є два варіанти:
  - (а) FSDP як у torch: all-gather ваг у fwd, ще раз у bwd при remat, reduce-scatter градієнтів. Це ≈3×0.63 GB ≈ 1.9 GB/крок.
  - (б) Репліковані ваги з all-reduce градієнтів: ≈2×0.63 GB = 1.26 GB/крок. Пам'ять: 630 MB ваг + 630 MB m на кожному чіпі, що прийнятно для 32 GB HBM.

  **Рекомендую (б):** менше байтів, і колоночна норма hsS стає локальною (повна матриця на кожному чіпі). Математику зберігаємо так: hsS рахує `col = Σ_rows upd²/rows_total` по повній матриці. Над реплікою це той самий результат без psum.

  Застереження: у torch градієнт обчислюється як reduce-scatter SUM/W по шардах. Для реплік це psum/W, математично те саме, а порядок fp32-сумування в межах ULP. SR-шум у hsS у torch невідтворюваний за визначенням.

  Якщо пам'яті при великому B не вистачить, лишаємо перемикач `LIGHT_FSDP=1`.
- **Малі параметри:** репліковані. `psum(g.f32)/W`, потім `.astype(bf16)`, як у torch.

**Підсумок пам'яті на чіп:**
- ваги ≈ 2.75 + 0.17 + 0.63 + 0.1 ≈ 3.7 GB;
- m ≈ 3.7 GB;
- градієнти ≈ 3.7 GB, їх частково можна позбутися донацією;
- разом ≈ 11 GB.

Під активації лишається ≈ 18 GB. Цього достатньо для B=4..8 з remat (див. §4).

### 2.3 Pytree

Ключі pytree збігаються з іменами torch.

Верхній рівень:
```
params = {
  'emb': (V,d)['x'], 'head': (V,d)['x'], 'l_norm', 'l_head0':(128,d), 'l_head1':(V,128)['x'],
  'norm', 'lock_w'(frozen), 'mix_w', 'ind_in_g', 'ind_a'(frozen),
  'light_a':[blk]*2, 'light_b':[blk]*2, 'final':[blk]*2,
  'heavy': stacked {'norm':(14,d), 'router':(14,40,d), 'up':(14,40,d,3840)['x' на осі 1],
                    'down':(14,40,3840,d)['x' на осі 1]},
  'memory': {...}, 'reread': {...}, 'book': (V,128)['x' або репл.],
}
```

Як влаштовані блоки:
- `blk` = {n1, qkv, o, n2, up, down}.
- `heavy` складаємо стеком (stacked), щоб пройти шари через `lax.scan`. Це дає швидку компіляцію і remat на шар.

Frozen-параметри (`lock_w`, `ind_a`) кладемо в окремий `consts`, а не в `params`. Тоді вони точно не отримують оновлень, а `has_grad`-маска для них не потрібна.

## 3. Модулі й сигнатури (ключові місця чисел)

### `prims.py`
- `rmsnorm(x, w)`: `(x.f32*rsqrt(mean(x.f32²)+1.1920929e-7)).astype(x.dtype) * w`. Точний порядок cast/mul треба підтвердити дампом з TPU (питання 3).
- Для memory, де вхід f32, ця формула дає вихід f32 через промоцію.
- `linear(x, W) = lax.dot_general(x.bf16, W.bf16, preferred_element_type=f32).astype(bf16)`. Це емуляція autocast.

### `attn.py`
- `flex_block`. Порядок:
  1. qkv3: три окремі зрізи `W[i*d:(i+1)*d]`.
  2. rope_half у f32, потім cast.
  3. splash.
  4. `x + o(a)`.
  5. `x + down(relu(up(n2 x))²)`, де relu² рахується в bf16.
- Splash: `splash_attention.make_splash_mha(MultiHeadMask([LocalMask((T,T),(2047,0),0)]*10), block_sizes=BlockSizes(1024,1024,512,1024,1024,512), head_shards=1, q_seq_shards=1)`. Вхід q треба заздалегідь помножити на `1/sqrt(256)`, бо в JAX splash scale не вбудований. Те, що torch_tpu ставить такий scale за замовчуванням, ще треба підтвердити (питання 3).
- На CPU: `dense_ref` з маскою `i-2047≤j≤i`.

### `ce.py`
- `chunk_ce` будуємо через `jax.custom_vjp`:
  - fwd: logits округлюються до bf16 і далі йдуть у fp32 lse по чанках, потім `logsumexp(stack)`. ly рахується як fp32 dot без округлення.
  - bwd: G переводиться в bf16; `dh += (Gb@Wc).f32`; `dW_c = Gbᵀ@hb` у bf16. Цикл по чанках робимо через `lax.fori_loop`/`scan`, щоб не матеріалізувати (N,V).
- `short_logp_vocab`, виконується всередині shard_map:
  - короткий Wk збираємо з one-hot частин: `psum` рядків `W_shard[sv_ids ∩ shard]`, всього 8193 рядки. Це точна сума.
  - Далі `chunk_ce` одним чанком.
  - rare-гілка:
    1. `all_gather(h[ridx])` дає (8·2560, d);
    2. локальний chunk_ce по своїх 16416 рядках з `sv_mask[lo:lo+rows]`;
    3. `lse = logsumexp(all_gather(lse_loc))`, `ly = psum(ly_loc·own)`;
    4. повертаємо свої C рядків.
  - Bwd через all_gather/psum JAX виводить автоматично. Сам chunk_ce має custom_vjp.

### `induct.py`
- `induct_raw` реалізуємо через `jnp.argsort(key, stable=True)`. Ключі int64 потребують `jax_enable_x64` лише для цієї функції, тому краще робити її на хості в numpy, як CPU_PREP у torch. Це просто і точно.
- `induct_in` робимо gather-формою: `h + (Σ_k wsel_k.bf16.f32 · h[src_k].f32).bf16`. Forward побітово збігається з bmm-формою.

### `anchors.py`
- tau беремо на чіп. Квантиль: `jnp.quantile(s[:,1:].ravel()[:100000], .8, method='linear')`.
- `idx = argsort((~keep)*T + arange(T))[:, :A]`. Ключі унікальні, тож результат детермінований.
- reread-вибір: `sort(lax.top_k(sp,128)[1])`. Збіг tie-порядку з torch на позиції 0 (1e4) треба перевірити тестом.

### `moe.py` (виконується всередині shard_map, форми локальні)
```python
def bank_moe(p, x, sa, cfg):            # x (B,A,d) bf16
    xf = rmsnorm(x, p['norm']).reshape(N,d)
    probs = softmax(linear(xf, p['router']).f32)
    top_p, top_e = lax.top_k(probs, 4); real = sa.reshape(N) >= 0
    w = top_p*real[:,None] / max(sum(...),1e-9)
    C = capacity(N)   # python: int(1.25*.8*N*4/40)+1 -> max(256, round(C/256)*256) (banker's)
    G,g,Cg,Ct = ...   # static python
    oh = one_hot(where(real[:,None], top_e, 40), 41)[..., :40]
    rank = cumsum(oh.reshape(G,g*4,40), 1) - 1 ; place = oh & (rank < Cg)
    D = ((re[...,None]==arange(Cg)) & pe[...,None]).astype(bf16)         # stop_gradient
    wtab = einsum('gsec,gse->egc', D, we.astype(bf16))
    xe = einsum('gsec,gsd->egcd', D, xg)                                  # bf16, точний
    ye = ep_ffn(xe.reshape(40,Ct,d), p['up'], p['down'])                  # all_to_all
    out = einsum('gsec,egcd->gsd', D, ye*wtab, preferred_element_type=f32).astype(bf16)
    aux = 40*sum(frac/sum(frac) * pm)
    return x + out[:N].reshape(B,A,d), aux, dropped

def ep_ffn(xe, up, down):    # up (5,d,3840) локальні
    r = lax.all_to_all(xe.reshape(8,5,Ct,d), 'x', 0, 0, tiled=False)    # (8 src,5,Ct,d)
    r = r.transpose(1,0,2,3).reshape(5, 8*Ct, d)
    a = bmm(r, up).astype(bf16); y = bmm(relu(a)**2, down).astype(bf16)
    y = y.reshape(5,8,Ct,d).transpose(1,0,2,3)
    return lax.all_to_all(y, 'x', 0, 0).reshape(40,Ct,d)
```
ErrProbe прибираємо, бо ніхто його не читає. Буфер `err` лишаємо в ckpt нулями або старими значеннями. Rebank, wave і hint викидаємо, а `assert step < WAVE_WARM` додаємо в trainer.

### `memory.py`
`memory_pass`: `lax.scan(body, init_state(B), h.reshape(B,16,512,d).swapaxes(0,1))`.
- `body` обгорнутий у `jax.checkpoint` і повертає `(st_new, (r_j, snap_j))`.
- `snap = (faces, contents=concat(fc, up(dc)).f32, pos, valid)`.
- `put()` робимо через one-hot einsum з усередненням дублікатів, точно як у torch. Скатер не годиться.
- Кроки 1–5 write виконуються в тій самій послідовності.
- Ширина einsum у `put` для int/meta залежить від питання 1. Реалізуємо обидва варіанти під прапором `MEM_EINSUM_BF16`.

### `reread.py`
- hs_k беремо через `take_along_axis` з f32 hs. Це точний gather.
- Scatter назад: `zeros.at[sel].set(o_k)` у bf16. Еквівалентність one-hot bmm тут під питанням: bmm у torch йде під autocast, а одна ненульова позиція дає ту саму величину.
- Всередині `reread_core`:
  - потрібні рядки `book` збираємо з шардів через `psum` маскованого gather;
  - halt-ланцюг розгортаємо на 4 кроки;
  - v дає 64 MB/чіп, тому на reread ставимо remat.

### `model.py`
Порядок forward:
1. emb + induct_in;
2. light_a×2;
3. memory_pass і додавання reads;
4. light_b×2;
5. l_norm, surprise (no-grad), anchors;
6. `lax.scan` по 14 heavy;
7. scatter_add;
8. final×2;
9. reread_sep;
10. norm.

Loss складається з чотирьох доданків:
- `L_main` (з mix);
- `0.01·mean(aux)`;
- `0.3·light-head CE`;
- `0.01·cost`.

### `instinct.py`
Стан (на чіп, `P('x')` по провідній осі — див. питання 4):
- `gain (4,P)`, `keys (4,d)`, `used`, `expect`, `seen`, `rel (P,)`;
- `seed_t` int32, бо int64 потрібен лише до truncation `*1000003` (див. нижче);
- `n_step`;
- `m`, `cs` у формі як pytree параметрів.

Функції:
- `prepare(st, loss_local, sit, lr_ratio) -> st, k, oh, s, stake, g`
- `update(params, grads, st, lr_t, lr_v, s_stake...)`: Python-цикл по листках у фіксованому sel-порядку (порядок torch `plist`). Для кожного листка:
  - `seed = (seed_t+=1) * 1000003` із wrap у int32, bitcast у f32;
  - kind `hs`/`hs3` → `pallas_hs1.instinct_hs1_2d/3d`;
  - kind `vocab` → `pallas_instinct.instinct_vocab`;
  - kind `hsS`, `tensor` → jnp generic. SR беремо з `jax.random` (fold_in step), бо в torch він теж невідтворюваний.
- Для `has_grad==False` оновлення пропускаємо повністю (`lax.cond` не потрібен, бо маска статична).
- `rel` записуємо лише при `n_step%8==0`. Норми рахуємо по локальному шарду.
- `finish(st, rel, s)` використовує `jnp.quantile(rel, .9)`.

Класифікація kind задається явно за іменами, а не за `shape[0]≥10000` (ризик 8):
- `vocab`: emb, head, l_head1, book;
- `hs3`: up і down експертів;
- `hs`: router і 2D-параметри memory/reread;
- `hsS`: light-лінійні;
- `tensor`: 1D-параметри і скаляри.

### `step.py`
```python
@partial(jax.jit, donate_argnums=(0,1,2))
def train_step(params, opt, tau, xy, prep, lr3):     # xy (8,B,T+1) P('x')
    return shard_map(_local, mesh, in_specs=..., out_specs=..., check_rep=False)(...)
def _local(params, opt, tau, xy, prep, lr3):
    (loss,(tau_n,stats)), grads = value_and_grad(train_loss, has_aux=True)(params, consts, xy, prep, tau)
    sit = situation(params['emb'], xy)                # own-tokens ∩ own-rows, psum, /(B*T)
    opt = prepare(opt, loss, sit, lr3.ratio)          # (порядок як у torch; від grads не залежить)
    grads = sync_grads(grads)                          # small: psum f32/W->bf16; EP: /W; vocab-shard: /W
                                                       # (psum всередині bwd all_gather вже дає SUM)
    params, opt = instinct_update(params, grads, opt, lr3)
    opt = finish(opt)
    return params, opt, tau_n, loss, stats
```
Що треба пояснити:
- В EP і vocab-шардах transpose від `all_gather`/`all_to_all` вже сумує градієнти по чіпах, тому `/W` дає таку саму семантику, як у torch.
- Подвійного ділення бути не може: власний `loss` кожного чіпа не ділиться на W до backward.

## 4. Remat і батч (для 400–500k tok/s)

- `jax.checkpoint` з `policy=checkpoint_dots_with_no_batch_dims` на кожен light-блок. Вихід splash зберігаємо через `save_only_these_names('attn_out')`. MaxText-підхід.
- Heavy: `scan` з `jax.checkpoint` і policy `nothing_saveable` на шар. Зберігається лише вхід `a` (B,A,d), а all_to_all перераховується в bwd. Якщо профіль покаже, що впираємося в комунікацію, переходимо на `save_only_these_names('ep_out')`.
- Memory scan: checkpoint на сегмент. Carry займає ~(64·2560 + 256·640)·4·B ≈ 1.3 MB·B, дрібниця.
- Reread і CE-чанки рахуємо у fwd і перераховуємо в bwd (custom_vjp CE вже так працює).
- **Батч:** починаємо з B=1×8192 для паритету, потім проходимо B=2,4,8. Мета: 4–8 послідовностей по 8192 токенів на чіп. Математика при B>1 змінюється, і це очікувано та коректно:
  - anchors і routing рахуються на рядок або на чіп;
  - C росте разом з N;
  - reread іде загальним (не LEAN) шляхом.
- Грубий бюджет FLOPs: ≈0.75B «активних» на токен → ~4.5 GFLOP/token × 500k ≈ 2.3 PF/s. Це ≈31% MFU від 8×918 TF. Досяжно, якщо EP-комунікація перекривається обчисленнями: XLA async collectives, `--xla_tpu_enable_async_all_to_all`, latency-hiding scheduler.

## 5. Дані

`data.py` імпортує `fastloop.CorpusData` / `GPUData` (numpy) з `sys.path=D:\ai\memarch`. Для кожного rank r використовуємо seed 1+r або `CorpusData(r, 8, CORPUS_SEED)`, як у torch. Отримані 8 масивів складаємо в `(8,B,T+1)` і передаємо через `jax.make_array_from_process_local_data` з `P('x')`.

`induct_raw` рахуємо тут у numpy (CPU_PREP), поки XLA виконує попередній крок. Префетч робимо в треді.

## 6. Checkpoint torch ↔ JAX

`ckpt.torch_to_jax(sd, cfg, opt_sd=None)`:
- **Повний (не-SHARD) torch ckpt:** ключі 1:1. Heavy-шари склеюємо в стек. Експерти ріжемо по EP, а при EP_ALLOC_LOCAL збираємо з 8 rank-файлів.
- **SHARD-ckpt із TPU-рекорду неповний:** там лише rank 0. Для імпорту потрібен збір: додати в torch `save_full=1`, де кожен rank пише свій шард, а `ckpt.py` склеює `weight_shard` у правильному порядку рядків з обрізанням паддингу.
- Optimizer: `m` і `cs` за тією самою мапою. `gain`, `keys`, `used`, `expect`, `seen`, `rel`, `seed_t`, `n_step` беремо з `_opt_state`.
- `jax_to_torch` будується дзеркально (для перевірки торчем).
- Власне збереження: Orbax з глобальними шардованими масивами, асинхронно, на gs://.

## 7. Тестовий стенд еквівалентності (`tests/`)

Спільна частина: `tests/torch_ref.py` запускає torch з `NO_COMPILE=1 DEVICE=cpu` і тим самим env. Він дампить:
- ваги (`sd`);
- вхід;
- проміжні результати: hook-и на light out, surprise s, idx/valid, heavy out, reads, sel, o_k;
- окремі loss-доданки;
- grads;
- стан opt після 1 кроку.

Формат дампу: `.npz`. JAX завантажує ті самі ваги й порівнює. Допуски:
- int/bool: точний збіг;
- bf16: max-rel ≤ 1e-2, mean ≤ 1e-3;
- f32-режим (параметр `cfg.dtype=f32` для обох): ≤ 1e-5.

Мала конфігурація: `Fv2moeKs8`-подібна (d=256, heads=1 при hd=256 або HEAD_DIM=64, E=8, heavy=2, seg=64, T=512..4096, SHORT_V пропорційно), із тими самими прапорами. Друга мала конфігурація — «tiny K40» з E=40, d=128 для перевірки EP 40/8.

Для перевірки шардування на CPU: `XLA_FLAGS=--xla_force_host_platform_device_count=8`, а torch-референс запускається через `hyperop/check_shard_cpu.py` (gloo, 8 процесів).

## 8. Етапи реалізації (кожен перевіряється на CPU)

| # | етап | тест |
|---|---|---|
| M0 | config + consts (sv_*, lock_bucket) з тих самих файлів, що на VM; lr_at | константи і lr збігаються з torch побітово |
| M1 | prims + flex_block + dense window attn | паритет блоку з torch `ATTN_IMPL=block` (fp32 точно, bf16 в межах ULP) |
| M2 | chunk_ce custom_vjp, surprise, short_logp (без шардів) | fwd/bwd проти `_ChunkCE` при ch=16768 і ch=8193 |
| M3 | induct_raw/in/mix, anchors, reread_select | точний збіг індексів; induct_in fwd побітово |
| M4 | bank_moe при W=1 (all_to_all замінено на тотожність) | probs, top_e, C/G/Cg, place, wtab, out, aux, grads; крайові випадки capacity |
| M5 | memory read/write/put + scan | посегментний паритет станів (int точно), unit-тести put/prune/swap/seg0 |
| M6 | reread_core + reread_sep | sel, qpos_k, o_k, cost, grads book/offset |
| M7 | повний forward і loss на малій конфігурації, W=1 | 4 loss-доданки + stats + усі grads; lock_w/ind_a без grad |
| M8 | Instinct: prepare/finish (jnp), pallas `interpret=True`, generic hsS/tensor, has_grad маска | 20 кроків prepare/finish точно; один крок оптимізатора з вимкненим SR |
| M9 | shard_map W=8 на CPU: EP all_to_all, vocab двоетапний lse, sync_grads, situation, tau по чіпах | проти check_shard_cpu: втрата на rank і ваги після 3 кроків |
| M10 | повний train_step (jit+shard_map+remat+donate) на CPU, 10 кроків | криві loss накладаються; стан instinct (k, s, stake, gain) збігається |
| M11 | ckpt torch↔jax + Orbax | round-trip побітово |
| M12 | **комбінований тест** (за правилом Стефана): усі стовпи разом, мала конфігурація, 20 кроків JAX і torch | loss-криві та stats |

**Перший TPU-бенчмарк (M13), v6e-8:**
1. Fv2moeK40h14d2560, B=1×8192, з одного повного ckpt. Перевіряємо 3 кроки проти torch_tpu: loss, tau, stats. Ціль паритету: розбіжність у межах bf16-шуму.
2. Пам'ять увімкнена, всі стовпи присутні (вимога «пам'ять першою»).
3. Далі B=1,2,4,8 з remat. Для кожного B друкуємо tok/s, профіль top-N, розбивку пам'яті та частку all_to_all у часі кроку.
4. **Ціль: ≥300k tok/s на першій ітерації, 400–500k після перекриття комунікацій** (база: ≈14k у torch_tpu).
5. На 400–500k: видалити VM і спитати тебе (правило «зупинитися на цілі»).
6. Потім 200 кроків паралельно з torch-рекордом для порівняння кривої.

## 9. Що потребує твого рішення

1. **Ширина einsum під autocast у torch_tpu** (memory `put`, sim/cos). Якщо там bf16, позиції закладок >256 квантуються. Що робимо: копіюємо цю особливість, чи робимо fp32 як очевидний задум? Спершу треба зробити probe на TPU.
2. **Situation vector.** Кожен чіп сумує лише ті свої токени, що потрапили у свій блок рядків. Це не середнє по батчу. Копіюємо помилку як є, чи замінюємо на справжнє середнє (`psum` повного gather)?
3. **RMSNorm** (порядок cast і mul з bf16 вагою) і **scale у splash_sdpa** на справжньому torch_tpu. Потрібен дамп одного шару. Пропоную вважати еталоном те, що покаже дамп.
4. **Стан Instinct і tau окремо на кожному чіпі:**
   - локальний loss у prepare;
   - rel за локальними шардами;
   - tau без синхронізації.

   Наслідок: репліки малих параметрів поступово розходяться. Варіант (а) — точна копія: стан на чіп і малі параметри теж на чіп. Варіант (б) — виправити: `pmean` loss, глобальні норми, синхронний tau. Математика зміниться, але стане чистішою. Моя рекомендація: (а) для паритету, потім (б) як окремий експеримент.
5. **hsS для light-лінійних:** лишаємо свіжу колоночну норму без lag, як у torch, чи переводимо на `pallas_hs1` з HS_LAG (швидше, але трохи інша математика)?
6. **Light-ваги:** репліковані (рекомендую, менше байтів) чи FSDP, як у torch? Математично це одне й те саме.
7. **Tie-break для topk/argsort** (focus, слоти memory, reread top_k): оголошуємо канонічним стабільний порядок «менший індекс першим»?
8. **Буфери, що не використовуються** (err, learned, up_snap, rebank/wave): викидаємо з JAX-стану (рекомендую), чи тримаємо заради round-trip ckpt?
9. **Файли counts_fineweb.npy / lock_trust.pt:** локальний counts має 50304 записи при VOCAB 131328. Треба взяти саме ті файли, що лежать на VM.
10. **Імпорт ваг:** рекордні SHARD-ckpt мають лише rank 0. Стартуємо JAX з нуля (новий seed, бо torch-init однаково не відтворити), чи спершу дописуємо в torch `save_full` і переносимо?
11. **Конфігурація рекордного запуску:** lr, steps, WARMUP і eval seq (при T≤2048 attention стає повністю causal). Також HEAVY_ATTN має лишатися вимкненим, тобто heavy-шари без attention. Підтверди.
12. **Великий батч на чіп** змінює per-chip ємність MoE і кількість якорів на крок. Чи це прийнятне «та сама математика»? Формули ті самі, тільки N більше.
13. **Параметри без grad** (окрім lock_w/ind_a, наприклад деякі параметри reread при REREAD_SEP): списка `p.grad is None` ще немає. Отримаємо його з одного torch-кроку і зафіксуємо як статичну маску. Згоден?

Нічого не змінено, файлів не створено. Основні джерела: D:\ai\memarch\full2.py, D:\ai\memarch\mem2.py, D:\ai\memarch\exp_opt.py, D:\ai\memarch\pallas_hs1.py, D:\ai\memarch\pallas_instinct.py, D:\ai\memarch\fastloop.py, D:\ai\memarch_fable\hyperop\step.py, D:\ai\memarch_fable\hyperop\shard.py, D:\ai\memarch_fable\hyperop\mempass.py. Новий пакет: D:\ai\memarch_jax\ (ще не створений).
## 10. Рішення Стефана (05.11)
- Старт ваг: з нуля в JAX (новий seed, той самий розподіл init). Для тестів еквівалентності — конвертер ваг torch→JAX на малих моделях.
- Великий батч на чіп (4–8×8192): ємність MoE і якорі рахуються від фактичного N тими ж формулами — прийнятно.
- Паритет з torch перевіряється на ОДНОМУ чіпі (W=1, ті самі ваги/дані → той самий loss/градієнти) — це лише пошук помилок порту.
- На 8 чіпах — одна синхронна модель: спільний tau (pmean квантиля), спільний loss для Instinct prepare (pmean), rel/норми за глобальними тензорами, situation = справжнє середнє по батчу всіх чіпів. Torch-особливості багаточиповості НЕ копіюємо.
- Дрібниці (einsum-точність, RMSNorm cast, tie-break, невикористані буфери): вирішую сам — як у torch на TPU, де відомо; інакше очевидний задум; невикористані буфери викидаємо. Кожен вибір записую в docs/decisions.md.
