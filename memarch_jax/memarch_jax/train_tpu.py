"""Тренування повної моделі на 8 чипах (v6e-8) з живим логом швидкості: той самий крок, що в bench_tpu
(dist.make_step: fwd+bwd у shard_map + Instinct, remat, donate), але на справжніх даних (data.Corpus з бакета,
HostLoader з префетчем і induct-підготовкою на хості), з розкладом lr (instinct.lr_at, WSD, warmup 200),
held-out eval, чекпойнтами (локально + GCS, атомарно, з data_state) і перевіркою resume.

  python -m memarch_jax.train_tpu --steps 1500 --remat moe_save_heavy --uroboros --uroboros_budget 0.0625 \\
      --counts .../counts_fineweb.npy --lock_trust .../lock_trust.npy --ckpt_gcs gs://.../ckpt/t1_main
  (env MOE_UNROLL=1 MOE_UP_MID=e — як у bench Q5)

Кожні --log_every кроків один рядок "[train] step ..." (flush одразу): loss і доданки, tok/s миттєвий (за вікно)
і середній (без кроків компіляції), мс/крок, lr, tau, обруч (corr/eaten/nll_sel/nll_skip), MoE (fill/drop/aux),
норма ваг, час очікування даних, минуло/ETA. Кожні --eval_every — "[train] eval ..." (mix/code/uk з held-out).

Resume-тест (запускач робить два процеси): A: --stop_at 60 --ckpt_every 50 --loss_log A.jsonl;
B: --resume --stop_at 60 --compare A.jsonl → "[train] RESUME_TEST PASS|FAIL" (кроки 51–60, |Δ| ≤ --cmp_tol).

Held-out: від ОСТАННЬОГО шматка кожного джерела відрізається хвіст документів (≥ --heldout_tok токенів, ≤10%
документів) — у тренуванні ці документи додаються до .drop-маски, eval бачить лише їх (без повторів за якістю).
Так не втрачаються цілі шматки джерел з 1–2 шматками (finewiki_uk, hplt_uk)."""
import argparse
import dataclasses
import json
import math
import os
import shutil
import subprocess
import sys
import threading
import time

# Суміш для повного прогону (див. docs/decisions.md, 06.11.2026): код 40%, українська 18%, edu 16%, math 15%, wiki 11%.
MIX = {
    "stack_v3": 0.25, "opc_fineweb_code": 0.15,                       # код
    "hplt_uk": 0.12, "finewiki_uk": 0.06,                             # українська
    "fineweb_edu": 0.10, "finepdfs_edu_en": 0.06,                     # edu
    "nemotron_math4": 0.08, "finemath4": 0.03, "opc_fineweb_math": 0.02, "openwebmath": 0.02,   # math
    "finewiki_en": 0.04, "finewiki_pl": 0.03, "finewiki_de": 0.02, "finewiki_es": 0.02,         # wiki
}
# фішки (Стефан 11.11): види тексту — увесь крок один вид (docs/fishki_plan.md у memarch)
FISH_KINDS = {"code": ("stack_v3", "opc_fineweb_code"), "en": ("fineweb_edu", "finepdfs_edu_en", "finewiki_en"),
              "uk": ("hplt_uk", "finewiki_uk"), "math": ("nemotron_math4", "finemath4", "opc_fineweb_math", "openwebmath"),
              "other": ("finewiki_pl", "finewiki_de", "finewiki_es")}
EVAL_SETS = {"mix": None, "code": {"stack_v3": 0.6, "opc_fineweb_code": 0.4}, "uk": {"hplt_uk": 0.7, "finewiki_uk": 0.3}}
EOS_DEFAULT = 131072      # <|endoftext|> у tok/qwen36_pruned131k (у бакеті немає .json-маніфестів)


def P(*a):
    print("[train]", *a, flush=True)
    try:
        from . import diag as _dg
        _dg.touch()
    except Exception:                                                   # noqa: BLE001
        pass


def _args(argv=None):
    a = argparse.ArgumentParser()
    # --- модель/крок (ті самі назви, що в bench_tpu — бо _cfg/_consts звідти)
    a.add_argument("--model", default="Fv2moeK40h14d2560")
    a.add_argument("--batch", type=int, default=4, help="рядків НА ЧИП")
    a.add_argument("--seq", type=int, default=8192)
    a.add_argument("--remat", default="moe_save_heavy")
    a.add_argument("--dtype", default=None)
    a.add_argument("--attn", default=None)
    a.add_argument("--no_pallas", action="store_true")
    a.add_argument("--counts", default=None)
    a.add_argument("--lock_trust", default=None)
    a.add_argument("--moe_chunks", type=int, default=None)
    a.add_argument("--moe_impl", default=None)
    a.add_argument("--moe_old", action="store_true")
    a.add_argument("--vocab_take", default=None)
    a.add_argument("--vocab_take_bwd", default=None)
    a.add_argument("--no_ce_a2a", action="store_true")
    a.add_argument("--rope_old", action="store_true")
    a.add_argument("--rope_mem", action="store_true", help="cfg.rope=True: вагонетка замість mem2/reread (rope.py)")
    a.add_argument("--rope_gather_cap", type=int, default=0, help="слів на запит у зібраній частині (0 = k·span_cap, точно)")
    # ті самі прапори вагонетки, що в bench_tpu (без них виміряну збірку «трос v3, 1 шар, Pallas» не можна було задати)
    a.add_argument("--rope_route", default=None, help="cfg.rope_route: hier|exact")
    a.add_argument("--rope_route_lazy", action="store_true")
    a.add_argument("--rope_route_bf16", action="store_true")
    a.add_argument("--rope_layers", type=int, default=0, help="вагонетка лише в перших N шарах light_b (0 — усі)")
    a.add_argument("--rope_share_route", action="store_true")
    a.add_argument("--rope_read", default=None, help="v1 | v3 (типове)")
    a.add_argument("--rope_pallas", action="store_true", help="Pallas-ядра вагонетки (потрібен --rope_route exact)")
    a.add_argument("--rope_dense_read", action="store_true")
    a.add_argument("--anchor_frac", type=float, default=None, help="частка слів-якорів (cfg.frac); без прапора — як у config (0.2)")
    a.add_argument("--anchor_cap", type=float, default=None, help="стеля якорів (cfg.cap); без прапора при --anchor_frac: = anchor_frac")
    a.add_argument("--uroboros", "--hoop", dest="uroboros", action="store_true")
    a.add_argument("--uroboros_budget", "--hoop_budget", dest="uroboros_budget", type=float, default=0.0625)
    a.add_argument("--chips", type=int, default=None)
    a.add_argument("--cpu8", action="store_true", help="8 CPU-пристроїв (smoke, напр. --model small --seq 256)")
    a.add_argument("--seed", type=int, default=0)
    a.add_argument("--vocab", type=int, default=None, help="V для small/tiny_k40 на справжніх даних (131328; їхні 4096 замалі)")
    a.add_argument("--short_v", type=int, default=None)
    # --- розклад
    a.add_argument("--steps", type=int, default=1500, help="довжина розкладу lr (WSD) і прогону")
    a.add_argument("--stop_at", type=int, default=None, help="зупинитися на цьому кроці (resume-тест); розклад — за --steps")
    a.add_argument("--lr", type=float, default=6e-4, help="lr_peak Instinct (типовий у наших тестах; ефективний ×scale 0.45)")
    a.add_argument("--warmup", type=int, default=200)
    a.add_argument("--opt_scale", type=float, default=None,
                   help="множник кроку Instinct (InstinctCfg.scale, типово 0.45); загасання ваги від нього не залежить")
    a.add_argument("--sched", default="wsd")
    # --- дані
    a.add_argument("--data_root", default="gs://YOUR_BUCKET/tokens")
    a.add_argument("--mix", default=None, help="JSON {джерело: вага} замість MIX")
    a.add_argument("--cache", default="/content/corpus_cache")
    a.add_argument("--eos", type=int, default=None)
    a.add_argument("--heldout_tok", type=int, default=2_000_000, help="токенів held-out з хвоста останнього шматка джерела")
    a.add_argument("--eval_batches", type=int, default=1, help="глобальних батчів на кожен eval-набір")
    # --- лог / eval / ckpt
    a.add_argument("--log_every", type=int, default=10)
    a.add_argument("--eval_every", type=int, default=250)
    a.add_argument("--eval_at_resume", action="store_true",
                   help="12.11: після resume — eval чекпойнта ДО першого кроку (у межах самого навчання; TPU лише для навчання)")
    a.add_argument("--eval_only", action="store_true",
                   help="12.11: лише eval відновленого ckpt (без жодного кроку тренування) і вихід")
    a.add_argument("--ckpt_every", type=int, default=500)
    a.add_argument("--ckpt_local", default="/dev/shm/memarch_ckpt", help="локальна тека ckpt (RAM: диск VM замалий; ціль Fv2moeK40 ≈45 GiB на ckpt)")
    a.add_argument("--ckpt_gcs", default=None, help="gs://... — копія кожного ckpt (у фоні) + маркер COMPLETE")
    a.add_argument("--no_final_ckpt", action="store_true")
    a.add_argument("--resume", action="store_true", help="продовжити з найновішого ckpt (локального, інакше з GCS)")
    a.add_argument("--resume_force", action="store_true", help="брати ckpt навіть з іншим run_id (свідомо: інші аргументи запуску)")
    a.add_argument("--loss_log", default=None, help="jsonl: {step, loss} кожного кроку")
    a.add_argument("--compare", default=None, help="jsonl іншого прогону: порівняти кроки після resume")
    a.add_argument("--cmp_tol", type=float, default=1e-6)
    a.add_argument("--remind", type=float, default=0.0,
                   help="частка рядків з нагадуванням-голкою (torch RECALL_REMIND=p; 0 — вимкнено, старий шлях)")
    a.add_argument("--remind_q", type=int, default=16, help="пар ключ→значення в голці (torch --pairs)")
    a.add_argument("--fish_n", type=int, default=0, help="фішки: >0 — важкі MoE замінено на fish_n FFN-фішок (docs/fishki_plan.md)")
    a.add_argument("--fish_hf", type=int, default=19456, help="ширина фішки (2·d·hf параметрів)")
    a.add_argument("--fish_pool", type=int, default=17, help="фішок у наборі виду (готова модель)")
    a.add_argument("--fish_c", type=int, default=2, help="фішок за крок на тренуванні (випадкові з набору виду)")
    a.add_argument("--fish_center", type=int, default=0, help="1 — вхід фішки без спільного вектора (фікс мертвих relu², fw_s2 14.11)")
    a.add_argument("--inst_word", action="store_true",
                   help="15.11 Instinct на словах: норма ≥0.5 — нічого; розмазано — легкий поштовх до 0.3; впевнено неправильне — обережно вниз")
    a.add_argument("--opt_rollback", action="store_true",
                   help="15.11 запобіжник-відкатник: loss > 1.5× звичного → ваги/інерція на стан до попереднього кроку, крок ×0.5")
    a.add_argument("--opt_err_step", action="store_true",
                   help="15.11 задум Стефана: крок тензора = lr·(loss/очікуване)·(звична ||g||/||g|| зараз)")
    a.add_argument("--opt_err_cap", type=float, default=3.0, help="err_step: верхня межа множника кроку")
    a.add_argument("--opt_gearbox", action="store_true",
                   help="15.11 Стефан: «механіка» — у кожного тензора передача 0.25/0.5/1/1.25 за готовою інструкцією")
    a.add_argument("--fish_reset", action="store_true",
                   help="14.11: при resume фішки — з початкових ваг (fishki.init), їхня інерція Instinct = 0")
    a.add_argument("--eval_nofish", action="store_true",
                   help="14.11: eval mix ще й з обнуленими фішками (down=0) → внесок фішок = mix_nofish − mix")
    a.add_argument("--fish_full_from", type=int, default=-1, help="з цього кроку — фінальна фаза: увесь набір виду")
    a.add_argument("--heavy_full_every", type=int, default=0,
                   help="тасування (ідея Стефана 11.11): кожен K-й крок один важкий шар (по колу 0..heavy-1) працює на ВСІХ словах; 0 — вимкнено")
    a.add_argument("--heavy_all", action="store_true",
                   help="усі важкі шари на всіх словах (фінальна фаза перед SFT / режим готової моделі)")
    a.add_argument("--remind_near", type=float, default=0.0,
                   help="частка голок справжнього тексту У ВІКНІ (відстань 0..вікно, трос відкритий): лікує заважання троса близьким фактам")
    a.add_argument("--remind_dist", type=int, default=0, help="мін. відстань факти→питання (0 = cfg.window)")
    a.add_argument("--remind_tok0", type=int, default=1000, help="перший токен голок: ключі +0..511, значення +512..575, мітка +576")
    a.add_argument("--remind_cand", action="store_true",
                   help="fix19: втрата відповіді голки — softmax лише по 64 значеннях (не по всьому словнику)")
    a.add_argument("--remind_mask_main", action="store_true",
                   help="fix19: у рядках з голками main не вчиться на токенах голок (k0..мітка)")
    a.add_argument("--remind_w", type=float, default=1.0, help="вага доданку remind")
    a.add_argument("--remind_use_w", type=float, default=0.0,
                   help="штраф на питаннях нагадувань: w·(1 − частка уваги троса), диференційовна mass (0 — немає)")
    a.add_argument("--remind_synth", type=int, default=0,
                   help="етап 1: перші N кроків — чиста синтетика пригадування (сміття + голка, втрата лише на відповідях, "
                        "як torch rope_recall); далі звичайний текст з нагадуваннями. Друга скомпільована програма кроку")
    a.add_argument("--rope_doc_mask", action="store_true", help="трос не відкриває якорі з іншого документа (межа EOS; лише маска, швидкість та сама)")
    a.add_argument("--remind_pack", action="store_true",
                   help="етап 1 (--remind_synth) точно як torch rope_recall: КОРОТКІ рядки (факти на початку рядка, сміття, "
                        "питання; довжина 4·пар+сміття, рядків — щоб слів у батчі було B·T), рівні (пар 4→8→q, сміття "
                        "16→…) за точністю, втрата — середнє по відповідях. Крок компілюється раз на рівень")
    a.add_argument("--remind_strict", action="store_true",
                   help="етап 1 «чесний тест» (Стефан 12.11): після першої помилки в епізоді правильні відповіді не "
                        "винагороджуються, а ймовірність на інших значеннях рядка штрафується завжди (проти перебору)")
    a.add_argument("--remind_strict_w", type=float, default=1.0, help="вага штрафу --remind_strict")
    a.add_argument("--remind_mix", default="",
                   help="плавний перехід після етапу 1 (Стефан 12.11): частки СИНТЕТИЧНИХ кроків по фазах у %%, напр. "
                        "90,75,50,25,5; остання лишається назавжди. Порожньо — різкий перехід, як було")
    a.add_argument("--remind_mix_len", type=int, default=250, help="кроків на фазу --remind_mix")
    a.add_argument("--remind_blank", action="store_true",
                   help="етап 1 і eval pack_fN: на місці відповіді в тексті заглушка, справжня відповідь — лише ціль "
                        "(Стефан 12.11: «відповіді бачив лише після тесту» — без цього модель викреслює вже названі)")
    a.add_argument("--opt_gnorm", action="store_true",
                   help="ПРОПОЗИЦІЯ (вимкнено): Instinct ділить градієнт кожної ваги на його rms перед інерцією (InstinctCfg.gnorm)")
    a.add_argument("--opt_culprit", default="abs", choices=("abs", "ema"),
                   help="12.11: як Instinct вибирає «винних» на поганій несподіванці: abs — найбільший |g|/|w| (як було), ema — найбільше перевищення власного середнього")
    a.add_argument("--opt_snr", action="store_true",
                   help="13.11 ПРОПОЗИЦІЯ: Instinct — крок тензора × впевненість c=||m||/sqrt(EMA||g||²) (шум → ~0.23, "
                        "сталий сигнал → 1); напрям той самий. Захищає трос від випадкового блукання (InstinctCfg.snr)")
    a.add_argument("--opt_snr_ref", type=float, default=1.0, help="c ≥ ref → повний крок (InstinctCfg.snr_ref)")
    a.add_argument("--opt_snr_beta", type=float, default=0.95, help="EMA ||g||² (InstinctCfg.snr_beta)")
    a.add_argument("--opt_snr_floor", type=float, default=0.1, help="найменший множник (InstinctCfg.snr_floor)")
    a.add_argument("--opt_snr_only", default="", help="через кому префікси тензорів (напр. light_a.,light_b.); порожньо — усі, крім фішок")
    a.add_argument("--opt_lr_mul", default="",
                   help="14.11 fix18: множник кроку Instinct для префіксів тензорів, напр. final.qkv:0.25,final.o:0.25; порожньо — вимкнено")
    a.add_argument("--opt_card_kind", action="store_true",
                   help="13.11 (Стефан): картка ситуації Instinct = вид кроку фішок (+ остання — синтетика), а не схожість "
                        "входу; при resume зі старого ckpt (4 картки) стан карток береться з ініціалізації")
    a.add_argument("--opt_stream_proj", action="store_true",
                   help="13.11 фікс зламу 15000 (K1): Instinct прибирає з градієнта й інерції light_a/light_b/final "
                        "(.o/.down/.qkv) складову вздовж спільного вектора потоку перед фінальною norm (InstinctCfg.stream_proj)")
    a.add_argument("--opt_norm_anchor", action="store_true",
                   help="13.11 фікс зламу 15000 (K2): 1D norm-ваги light_a/light_b/final тягнуться до 1 (wd для норм)")
    a.add_argument("--norm_dump_proj", action="store_true",
                   help="fix14 F1 (вимкнено): фінальна norm прибирає з виходу складову вздовж EMA-напряму спільного вектора "
                        "потоку opt.U (будь-який канал), знаменник — по повному h. Потребує --opt_stream_proj (стан U)")
    a.add_argument("--opt_gnorm_rawrel", action="store_true",
                   help="fix14 (вимкнено): з --opt_gnorm rel=|g|/|w| для винних/карток — від СИРОГО градієнта; при resume "
                        "rel_avg (зіпсований ~150 після fix13) скидається")
    a.add_argument("--opt_m_rate", action="store_true",
                   help="fix15 (вимкнено): обмеження швидкості виходу m — U-складова кроку матриць потоку не швидше "
                        "κ·звичайної когерентної швидкості, надлишок відкладається і віддається наступними кроками")
    a.add_argument("--opt_m_rate_kappa", type=float, default=2.0)
    a.add_argument("--opt_m_rate_warm", type=int, default=100)
    a.add_argument("--opt_m_rate_gate", action="store_true",
                   help="13.11 fix15: відкладений рух m віддається лише там, де свіжий градієнт ще згоден (інакше гаситься)")
    a.add_argument("--opt_snr_dz", action="store_true",
                   help="fix14 (вимкнено): з --opt_snr — dead-zone впевненості (гальмує лише блукання, сигнал — повний крок)")
    a.add_argument("--opt_snr_horizon", type=float, default=250.0, help="горизонт H для --opt_snr_dz (кроків)")
    a.add_argument("--data_reseed", type=int, default=-1,
                   help="14.11 контроль: при resume документи з нового зерна (розклад видів і фішки ті самі); -1 — вимкнено")
    a.add_argument("--opt_safe16", action="store_true",
                   help="13.11 fix16: Instinct для light_a/light_b/final і норм — проєкція інерції проти градієнта, cautious-маска, відносний крок 1D-норм (jnp-шлях)")
    a.add_argument("--opt_wd_norm", type=float, default=0.0,
                   help="14.11: weight decay до 0 для 1D норм-ваг (norm, l_norm, .n1, .n2) усередині Instinct; 0 — вимкнено")
    a.add_argument("--opt_surge_thr", type=float, default=0.0, help="fix20: −s>thr → крок ×(thr/−s)²; 0 — вимк.")
    a.add_argument("--opt_damp", type=float, default=0.0,
                   help="fix21: гасіння сплеску градієнта по тензору: ||g|| > k·типової → градієнт в інерцію обрізано до "
                        "k·типової, крок ×(k·типова/||g||)^p; 0 — вимк. (легкі/фінальні/норми; стан opt.dv, opt.df)")
    a.add_argument("--opt_damp_p", type=float, default=1.0, help="fix21: степінь скорочення кроку")
    a.add_argument("--opt_g_max", type=float, default=3.0, help="верхня межа підсилювачів карток Instinct (було 3.0)")
    a.add_argument("--safe_norm_ratio", type=float, default=0.0,
                   help="13.11 fix16: після фінальної norm спільний вектор (причинний) ≤ ratio·сигнал слів; 0 — вимкнено")
    a.add_argument("--safe_norm_t0", type=int, default=256)
    a.add_argument("--hn_guard_w", type=float, default=0.0,
                   help="14.11 fix17: шарнірний штраф на стовбур (|середнє hn рядка|, середня |hn| понад пороги); 0 — вимкнено")
    a.add_argument("--hn_guard_mean", type=float, default=15.0)
    a.add_argument("--hn_guard_len", type=float, default=42.0)
    a.add_argument("--opt_reset_gain", action="store_true",
                   help="12.11: після resume — підсилювачі карток Instinct (gain) = 1 (старі ckpt мали 3× на light_a/0 через abs-вибір винних)")
    a.add_argument("--d_model", type=int, default=0,
                   help="ПРОПОЗИЦІЯ (0 = вимкнено): ширина d для малих моделей на ПК (перевірка механізму, не повна модель)")
    a.add_argument("--text_lr_ramp", type=int, default=0,
                   help="ПРОПОЗИЦІЯ (0 = вимкнено): після етапу 1 крок lr на ТЕКСТОВИХ кроках росте лінійно від 0 до повного "
                        "за стільки текстових кроків (синтетичні кроки — повний lr). Лічильник не зберігається в ckpt")
    a.add_argument("--attn_gain", type=float, default=1.0,
                   help="ПРОПОЗИЦІЯ (вимкнено = 1.0): множник оцінок уваги легких шарів (cfg.attn_gain)")
    a.add_argument("--rope_pcut", type=float, default=0.0,
                   help="ПРОПОЗИЦІЯ (0 = вимкнено): p-RoPE троса — обертаються лише частоти з хвилею ≤ N слів (напр. 256); "
                        "решта вимірів без позиції → пригадування на небачених відстанях (d8125). Лише шар троса")
    a.add_argument("--rope_anchor_top", action="store_true",
                   help="ПРОПОЗИЦІЯ (вимкнено): коли слів понад поріг більше за стелю якорів, трос бере A найбільш здивованих "
                        "по всьому рядку, а не перші A (як було: кінець рядка без якорів — свіжі факти недосяжні)")
    a.add_argument("--rope_frac", type=float, default=0.0,
                   help="ПРОПОЗИЦІЯ (0 = як --anchor_frac): частка якорів лише троса (густіші якорі → ланки покривають більше слів)")
    a.add_argument("--rope_cap", type=float, default=0.0, help="стеля якорів троса (0 = як cap)")
    a.add_argument("--rope_key_diag", action="store_true",
                   help="13.11 (вимкнено): у журналі klink/kanch — частка слів-ключів голок у перших rope_links словах шматка "
                        "свого якоря (їх бачить пошук) і частка ключів-якорів; на навчання не впливає")
    a.add_argument("--remind_w_synth", type=float, default=-1.0,
                   help="13.11 (−1 = як --remind_w): вага доданку remind лише на СИНТЕТИЧНИХ кроках (alt_cfg); "
                        "дозволяє міняти вагу голок у тексті (--remind_w), не чіпаючи синтетичні кроки")
    a.add_argument("--prev_in", action="store_true",
                   help="ПРОПОЗИЦІЯ (вимкнено): вхід «попереднє слово» h[t] += σ(prev_g)·emb(x[t−1]) (cfg.prev_in; новий параметр)")
    a.add_argument("--remind_tail", action="store_true",
                   help="ПРОПОЗИЦІЯ (вимкнено): етап 1 — сміття між фактами й питаннями номінальне (16, 32, …), а доповнення "
                        "рядка до --remind_pack_round іде після питань. Форма батчу, швидкість і число відповідей ті самі")
    a.add_argument("--remind_pack_frac", default="",
                   help="етап 1: сміття сходинок у частках вікна моделі через кому (напр. 0.125,0.25,0.5,1,1.5,2,4); "
                        "сходинки з тією самою справжньою відстанню після --remind_pack_round не повторюються. "
                        "Порожньо — старий фіксований список 16,32,…")
    a.add_argument("--remind_pack_round", type=int, default=1,
                   help="--remind_pack: довжина короткого рядка кратна цьому числу (ядра Pallas троса/уваги на TPU "
                        "потребують кратності блоку, напр. 256); 1 — точно як torch (4·пар + сміття − 1)")
    a.add_argument("--remind_same_fmt", action="store_true",
                   help="12.11: голки в справжньому тексті (і eval remind_dN/synth_dN) у ТОМУ САМОМУ форматі, що синтетика етапу 1: [k v]·n без міток, відповіді приховані (з --remind_blank)")
    a.add_argument("--remind_far_frac", type=float, default=-1.0,
                   help="12.11: останній рівень синтетики — частка батчів зі сміттям ЗА вікном (лише трос); −1 — рівномірно з усіх довжин (старе: 2 з 9)")
    a.add_argument("--remind_per_row", type=int, default=1,
                   help="13.11 (solve_curriculum): голок у КОЖНОМУ рядку тексту з голкою (різні ключі, без накладань). "
                        "Разом із --remind 1.0 дає ~R·16 відповідей на рядок за ~R·64 слів (R=4: 3%% рядка 8192)")
    a.add_argument("--remind_far_log", type=float, default=-1.0,
                   help="13.11 (solve_curriculum): відстані голок у тексті — частка F за вікном, лог-рівномірно від "
                        "--remind_dist до КІНЦЯ рядка (T−2·блок, напр. 8125); решта (1−F) — у вікні 0..вікно. −1 — як було")
    a.add_argument("--remind_pack_level", type=int, default=-1,
                   help="resume з ckpt без стану сходинок синтетики (старі ckpt, напр. ckpt_safe/fix_step_500): "
                        "з якого рівня продовжити; −1 — з 0")
    a.add_argument("--remind_p0", type=float, default=None,
                   help="частка нагадувань на старті (перші --remind_warm кроків), далі лінійно за --remind_decay до --remind")
    a.add_argument("--remind_warm", type=int, default=0)
    a.add_argument("--remind_decay", type=int, default=0)
    a.add_argument("--remind_ladder", default="",
                   help="ускладнення відстані за точністю: 'auto' (від window/4 подвоєнням до найдальшої) або список через кому")
    a.add_argument("--remind_thr", type=float, default=0.9, help="поріг точності для наступного рівня (torch 0.9)")
    a.add_argument("--remind_win", type=int, default=800, help="відповідей у вікні точності рівня")
    a.add_argument("--remind_eval", default="",
                   help="відстані факти→питання для eval пригадування, через кому (порожньо: window/2, window, 2·window, "
                        "найдальша, що влазить у T)")
    a.add_argument("--spike", type=float, default=1.5, help="loss > spike·EMA → рядок SPIKE")
    return a.parse_args(argv)


# ------------------------------------------------------------------ held-out: хвіст останнього шматка джерела

class Heldout:
    """Підміна data._fetch: для останнього шматка кожного джерела train-маска .drop += хвіст, eval-маска = все, крім хвоста."""

    def __init__(self, data_mod, heldout_tok, eval_cache):
        self.d, self.tok, self.eval_cache = data_mod, heldout_tok, eval_cache
        self.last = {}            # src → ім'я останнього шматка (глобально, до розбиття по rank)
        self.tails = {}           # (src, name) → (n_docs, індекси хвоста)
        self._orig = data_mod._fetch
        data_mod._fetch = self._fetch

    def _tail(self, tok_path, src, name):
        key = (src, name)
        if key not in self.tails:
            import numpy as np
            import zstandard as zstd
            with open(tok_path, "rb") as f:
                arr = np.frombuffer(zstd.ZstdDecompressor().decompress(f.read()), dtype=np.uint32)
            ends = np.flatnonzero(arr == self.eos) + 1
            lens = np.diff(np.concatenate([[0], ends]))
            n = len(ends)
            k, acc = 0, 0
            while k < max(1, n // 10) and acc < self.tok:
                acc += int(lens[n - 1 - k])
                k += 1
            self.tails[key] = (n, np.arange(n - k, n, dtype=np.int64), acc)
        return self.tails[key]

    def _fetch(self, root, src, name, cache):
        paths = self._orig(root, src, name, cache)
        if self.last.get(src) != name or paths[0] is None:
            return paths
        import numpy as np
        n, tail, _ = self._tail(paths[0], src, name)
        drop = np.load(paths[1]).astype(np.int64) if paths[1] else np.zeros(0, np.int64)
        is_eval = os.path.abspath(cache) == os.path.abspath(self.eval_cache)
        if is_eval:
            m = np.union1d(drop, np.setdiff1d(np.arange(n, dtype=np.int64), tail))
            out = os.path.join(cache, src, name + ".heldout_eval.drop.npy")
        else:
            m = np.union1d(drop, tail)
            out = os.path.join(cache, src, name + ".heldout_train.drop.npy")
        os.makedirs(os.path.dirname(out), exist_ok=True)
        np.save(out, m)
        # eval — без повторів за якістю коду (кожен документ один раз)
        return [paths[0], out, paths[2], None if is_eval else paths[3]]


# ------------------------------------------------------------------ ckpt: шардоване відновлення + GCS

def _ckpt_dirs(root):
    if not os.path.isdir(root):
        return []
    return sorted(int(n[5:]) for n in os.listdir(root)
                  if n.startswith("step_") and n[5:].isdigit() and os.path.exists(os.path.join(root, n, "meta.json")))


# Прапорці, що не впливають на вчення: прогін з іншими їх значеннями — той самий прогін (той самий run_id).
RUN_ID_SKIP = frozenset(("stop_at", "ckpt_every", "ckpt_local", "ckpt_gcs", "no_final_ckpt", "resume", "resume_force",
                         "loss_log", "compare", "cmp_tol", "cache", "log_every", "eval_every", "eval_batches", "spike", "eval_only", "eval_at_resume",
                         "remind_pack_level"))


REMIND_DEFAULTS = dict(remind=0.0, remind_q=16, remind_dist=0, remind_tok0=1000, remind_w=1.0, remind_eval="",
                       remind_p0=None, remind_warm=0, remind_decay=0, remind_ladder="", remind_thr=0.9,
                       remind_win=800, remind_near=0.0, remind_use_w=0.0, remind_synth=0, remind_pack=False, remind_pack_round=1)


def run_key(args_dict, mix):
    # прапорці нагадувань у типових значеннях не входять у ключ: run_id старих прогонів не змінюється
    return dict(args={k: v for k, v in args_dict.items() if k not in RUN_ID_SKIP
                      and not (k in REMIND_DEFAULTS and v == REMIND_DEFAULTS[k])}, mix=mix)


def run_id(args_dict, mix):
    """Ідентифікатор прогону: хеш усього, що визначає вчення (модель, дані, розклад, seed). ckpt з іншим run_id —
    чужий: resume його не бере, копія у сховищі з тим самим номером кроку не вважається нашою."""
    import hashlib
    return hashlib.sha1(json.dumps(run_key(args_dict, mix), sort_keys=True, default=str).encode()).hexdigest()[:16]


def extra_run_id(extra):
    """run_id із extra ckpt; старі ckpt без поля — рахуємо з їхніх збережених аргументів. None — визначити не можна."""
    if not isinstance(extra, dict):
        return None
    if extra.get("run_id"):
        return extra["run_id"]
    if "args" in extra:
        return run_id(extra["args"], extra.get("mix"))
    return None


def run_diff(extra, args_dict, mix):
    """Чим чужий ckpt відрізняється від цього прогону (для повідомлення)."""
    if not isinstance(extra, dict) or "args" not in extra:
        return "немає аргументів у ckpt"
    a, b = run_key(extra["args"], extra.get("mix")), run_key(args_dict, mix)
    ks = sorted(k for k in set(a["args"]) | set(b["args"]) if a["args"].get(k, "—") != b["args"].get(k, "—"))
    out = [f"{k}: ckpt {a['args'].get(k, '—')!r} ≠ тут {b['args'].get(k, '—')!r}" for k in ks]
    if a["mix"] != b["mix"]:
        out.append("mix")
    return "; ".join(out) or "той самий набір аргументів"


def local_extra(path):
    try:
        with open(os.path.join(path, "meta.json"), encoding="utf-8") as f:
            return json.load(f).get("extra")
    except Exception:                                                   # noqa: BLE001
        return None


CARD_FIELDS = ("gain", "keys", "used", "expect", "seen", "oh")   # стан карток Instinct (розмір = кількість карток)


def restore_sharded(path, template, reset_cards=False):
    """ckpt.save(npz) → state з ТИМИ САМИМИ шардуваннями, що й template (ckpt.restore кладе все на чип 0 —
    24B-модель туди не влазить). Листок за листком: np.load(mmap) → device_put(sharding шаблону)."""
    import numpy as np
    import jax
    import jax.numpy as jnp
    from . import ckpt
    with open(os.path.join(path, "meta.json"), encoding="utf-8") as f:
        meta = json.load(f)
    assert meta["backend"] == "npz", meta["backend"]
    tree = ckpt.state_tree(template)
    paths, vals = ckpt._flatten(tree)
    lv = meta["leaves"]
    # 12.11: листки шаблону, яких немає в ckpt (нові поля стану, напр. Instinct rel_avg), лишаються з шаблону
    idx = {json.dumps(x["path"]): i for i, x in enumerate(lv)}
    miss = [p for p in paths if json.dumps(p) not in idx]
    extra_ck = len(lv) - (len(paths) - len(miss))
    assert extra_ck == 0, f"у ckpt є листки, яких немає в стані моделі ({extra_ck})"
    if miss:
        print(f"[train] resume: нових полів стану немає в ckpt — беру з ініціалізації: {[str(p) for p in miss]}", flush=True)
    leaves, treedef = jax.tree_util.tree_flatten(tree)
    new = []
    for p_, x in zip(paths, leaves):
        if json.dumps(p_) not in idx:
            new.append(x)
            continue
        i = idx[json.dumps(p_)]
        d = lv[i]
        a = np.load(os.path.join(path, "arrays", f"{i}.npy"), mmap_mode="r")
        if d["dtype"] == "bfloat16":
            a = a.view(jnp.bfloat16)
        _key = [k[1] if isinstance(k, list) else k for k in p_]
        if (reset_cards and len(_key) == 2 and _key[0] == "opt" and _key[1] in CARD_FIELDS
                and list(a.shape) != list(x.shape)):
            print(f"[train] resume: opt/{_key[1]} {list(a.shape)} → {list(x.shape)} (--opt_card_kind) — з ініціалізації",
                  flush=True)
            new.append(x)
            continue
        assert list(a.shape) == list(x.shape) and str(a.dtype) == str(x.dtype), (d, x.shape, x.dtype)
        y = jax.device_put(np.array(a), x.sharding)   # np.array: 0-d лишається 0-d (ascontiguousarray робить 1-d)
        x.delete()                                         # шаблон більше не потрібен — звільняємо HBM одразу
        new.append(y)
    st = jax.tree_util.tree_unflatten(treedef, new)
    return dict(template, **st), meta["data_state"], meta["extra"]


class GcsStore:
    """Сховище копій ckpt у GCS (gcloud storage). Копія без маркера COMPLETE вважається битою."""

    def __init__(self, dst):
        self.dst = dst

    def is_complete(self, step):
        return subprocess.run(f"gcloud storage ls {self.dst}/step_{step}/COMPLETE", shell=True,
                              capture_output=True).returncode == 0

    def upload(self, local, step):
        """→ '' або текст помилки. Спершу дані, маркер — останнім."""
        tgt = f"{self.dst}/step_{step}"
        r = subprocess.run(f"gcloud storage rm -r -q {tgt} 2>/dev/null; gcloud storage rsync -r -q {local} {tgt}",
                           shell=True, capture_output=True, text=True)
        if r.returncode:
            return r.stderr[-300:]
        r = subprocess.run(f"echo {step} | gcloud storage cp -q - {tgt}/COMPLETE", shell=True, capture_output=True,
                           text=True)
        return r.stderr[-300:] if r.returncode else ""

    def steps(self):
        out = subprocess.run(f"gcloud storage ls {self.dst}/*/COMPLETE", shell=True, capture_output=True,
                             text=True).stdout
        names = [ln.rstrip("/").split("/")[-2] for ln in out.split()]
        return sorted(int(n[5:]) for n in names if n.startswith("step_") and n[5:].isdigit())

    def delete_older(self, step):
        # 15.11: лишати ще CKPT_KEEP−1 попередніх (добрий 35500 перезаписався зламаним 35750 при keep=1)
        keep = max(int(os.environ.get("CKPT_KEEP", "3")) - 1, 0)
        ls = subprocess.run(f"gcloud storage ls {self.dst}/", shell=True, capture_output=True, text=True).stdout
        old = sorted((int(d.rstrip("/").split("/")[-1][5:]), d) for d in ls.split()
                     if d.rstrip("/").split("/")[-1].startswith("step_") and d.rstrip("/").split("/")[-1][5:].isdigit()
                     and int(d.rstrip("/").split("/")[-1][5:]) < step)
        for _, d in (old[:len(old) - keep] if keep else old):
            subprocess.run(f"gcloud storage rm -r -q {d}", shell=True)

    def fetch(self, step, dst):
        subprocess.run(f"gcloud storage rsync -r -q {self.dst}/step_{step} {dst}", shell=True, check=True)

    def extra(self, step):
        """extra з meta.json копії (None — не прочиталося)."""
        r = subprocess.run(f"gcloud storage cat {self.dst}/step_{step}/meta.json", shell=True, capture_output=True,
                           text=True, encoding="utf-8")
        try:
            return json.loads(r.stdout).get("extra") if r.returncode == 0 else None
        except Exception:                                               # noqa: BLE001
            return None


class FileStore(GcsStore):
    """Те саме на локальній теці (--ckpt_gcs file://...): перевірка kill+resume без хмари.
    CKPT_TEST_UPLOAD_DELAY — секунд паузи ПОСЕРЕДИНІ копіювання (завантаження «у польоті»)."""

    def __init__(self, dst):
        self.root = dst[len("file://"):]
        os.makedirs(self.root, exist_ok=True)

    def _d(self, step):
        return os.path.join(self.root, f"step_{step}")

    def is_complete(self, step):
        return os.path.exists(os.path.join(self._d(step), "COMPLETE"))

    def upload(self, local, step):
        tgt = self._d(step)
        shutil.rmtree(tgt, ignore_errors=True)
        files = sorted(os.path.relpath(os.path.join(r, f), local) for r, _, fs in os.walk(local) for f in fs)
        for n, f in enumerate(files):
            if n == len(files) // 2:
                if step >= int(os.environ.get("CKPT_TEST_DELAY_FROM", "0")):
                    time.sleep(float(os.environ.get("CKPT_TEST_UPLOAD_DELAY", "0")))
            os.makedirs(os.path.dirname(os.path.join(tgt, f)), exist_ok=True)
            shutil.copy2(os.path.join(local, f), os.path.join(tgt, f))
        with open(os.path.join(tgt, "COMPLETE"), "w") as fh:
            fh.write(str(step))
        return ""

    def steps(self):
        return sorted(int(n[5:]) for n in os.listdir(self.root) if n.startswith("step_") and n[5:].isdigit()
                      and os.path.exists(os.path.join(self.root, n, "COMPLETE")))

    def delete_older(self, step):
        # 15.11: лишати ще CKPT_KEEP−1 попередніх (добрий 35500 перезаписався зламаним 35750 при keep=1)
        keep = max(int(os.environ.get("CKPT_KEEP", "3")) - 1, 0)
        old = sorted(int(n[5:]) for n in os.listdir(self.root) if n.startswith("step_") and n[5:].isdigit()
                     and int(n[5:]) < step)
        for st in old[:len(old) - keep] if keep else old:
            shutil.rmtree(os.path.join(self.root, f"step_{st}"), ignore_errors=True)

    def fetch(self, step, dst):
        shutil.copytree(self._d(step), dst, dirs_exist_ok=True)

    def extra(self, step):
        return local_extra(self._d(step))


def make_store(dst):
    return None if not dst else (FileStore(dst) if dst.startswith("file://") else GcsStore(dst))


class Uploader:
    """Копія ckpt у сховище окремим потоком; цикл навчання ніколи не чекає (крім кінця прогону).
    Найновіший перемагає: поки йде копія, новий ckpt стає в чергу, замінюючи того, хто ще чекав.
    Старі копії видаляються лише після успішного маркера COMPLETE нової."""

    def __init__(self, store, on_change=None, run_id=None):
        self.store, self.on_change, self.run_id = store, on_change or (lambda: None), run_id
        self.lock = threading.Lock()
        self.pending, self.current, self.t = None, None, None

    def pinned(self):
        """Кроки, чиї локальні теки ще потрібні копіюванню (їх не можна видаляти)."""
        with self.lock:
            return {x[1] for x in (self.current, self.pending) if x is not None}

    def start(self, local, step):
        if self.store is None:
            return
        with self.lock:
            replaced = self.pending
            self.pending = (local, step)
            if self.t is None or not self.t.is_alive():
                self.t = threading.Thread(target=self._run, daemon=True)
                self.t.start()
        if replaced is not None:
            P(f"ckpt step {replaced[1]}: копію в сховище пропущено, бо є новіший")
            self.on_change()

    def _run(self):
        while True:
            with self.lock:
                self.current, self.pending = self.pending, None
                job = self.current
                if job is None:
                    self.t = None                                        # інакше start міг би побачити живий потік, що вже виходить
                    return
            local, step = job
            t0 = time.time()
            try:
                # готова копія з тим самим номером кроку наша лише при тому самому run_id; чужу перезаписуємо
                same = self.run_id is None or not hasattr(self.store, "extra") or (
                    extra_run_id(self.store.extra(step)) == self.run_id)
                err = "" if (self.store.is_complete(step) and same) else self.store.upload(local, step)
                if err:
                    P(f"ПОМИЛКА копії ckpt step {step}: {err}")
                else:
                    P(f"ckpt step {step} → сховище ({time.time() - t0:.0f} с)")
                    self.store.delete_older(step)
            except Exception as e:                                       # noqa: BLE001 — потік не валить навчання
                P(f"ПОМИЛКА копії ckpt step {step}: {e!r}")
            with self.lock:
                self.current = None
            self.on_change()

    def wait(self):
        """Лише для кінця прогону."""
        while True:
            with self.lock:
                t = self.t
            if t is None:
                return
            t.join()


class BackgroundSaver:
    """Запис знімка ckpt у теку окремим потоком. Один запис за раз: якщо попередній ще йде, новий пропускається
    ДО копіювання стану (цикл не чекає). Найновіша повна тека лишається валідною: стара видаляється після нової."""

    def __init__(self, ckpt_mod, root, uploader):
        self.ck, self.root, self.up, self.t = ckpt_mod, root, uploader, None
        self.lock = threading.Lock()
        self.skipped = 0
        uploader.on_change = self.cleanup

    def busy(self):
        return self.t is not None and self.t.is_alive()

    def cleanup(self):
        """Лишає найновіший і ті, що ще копіюються."""
        with self.lock:
            dirs = _ckpt_dirs(self.root)
            pin = self.up.pinned()
            for s in dirs[:-1]:
                if s not in pin:
                    shutil.rmtree(os.path.join(self.root, f"step_{s}"), ignore_errors=True)

    def submit(self, tree, step, data_state, extra):
        def run():
            t0 = time.perf_counter()
            dst = os.path.join(self.root, f"step_{step}")
            try:
                import jax                                                   # у модулі jax імпортується лише в main
                if any(isinstance(x, jax.Array) for x in jax.tree_util.tree_leaves(tree)):
                    tree2 = self.ck.host_tree_to_numpy(tree)             # CKPT_PINNED: перетворення вже у фоні
                else:
                    tree2 = tree
                self.ck.write(dst, tree2, data_state=data_state, extra=extra, backend="npz")
            except Exception as e:                                       # noqa: BLE001
                P(f"ПОМИЛКА запису ckpt step {step}: {e!r}")
                return
            P(f"ckpt step {step} → {dst} (запис у фоні {time.perf_counter() - t0:.1f} с)")
            self.up.start(dst, step)                                     # спершу pin, потім прибирання старих
            self.cleanup()
        self.t = threading.Thread(target=run, daemon=True)
        self.t.start()

    def wait(self):
        if self.t is not None:
            self.t.join()


def fetch_store_latest(store, local_root, rid=None, describe=None):
    """Найновіший ckpt з маркером COMPLETE зі сховища → local_root/step_N (None — немає).
    rid — брати лише копії цього прогону (чужі з більшим номером кроку пропускаються з рядком у лог)."""
    steps = store.steps()
    s = None
    for c in reversed(steps):
        ex = store.extra(c) if rid is not None and hasattr(store, "extra") else None
        if rid is None or not hasattr(store, "extra") or extra_run_id(ex) == rid:
            s = c
            break
        P(f"resume: копія step {c} у сховищі — з ІНШОГО прогону, пропускаю ({describe(ex) if describe else ''})")
    if s is None:
        return None
    dst = os.path.join(local_root, f"step_{s}")
    tmp = dst + ".tmp"
    shutil.rmtree(tmp, ignore_errors=True)
    os.makedirs(tmp, exist_ok=True)
    t0 = time.time()
    store.fetch(s, tmp)
    shutil.rmtree(dst, ignore_errors=True)
    os.replace(tmp, dst)
    P(f"ckpt step {s} завантажено зі сховища за {time.time() - t0:.0f} с")
    return dst


# ------------------------------------------------------------------ головне

def _recomp(ts):
    """Перекомпіляцій train_step після першої (розмір jit-кешу − 1); eval/норма ваг — окремі програми, не рахуються."""
    try:
        return ts._cache_size() - 1
    except Exception:                                                   # noqa: BLE001
        return -1


def _num(x):
    """Скаляр/масив/дерево стану Instinct → коротке число для логу (середнє листків)."""
    import numpy as np
    import jax
    try:
        v = [np.asarray(t, np.float64).mean() for t in jax.tree_util.tree_leaves(x)]
        return f"{float(np.mean(v)):.3g}" if v else "-"
    except Exception:                                                   # noqa: BLE001
        return "?"


def _snr_str(mm, spec):
    """--opt_snr: множник впевненості кроку для тензорів троса і середній (для журналу)."""
    import numpy as np
    if "snr_f" not in mm:
        return ""
    f = np.asarray(mm["snr_f"], np.float64).ravel()
    names = [e[0] for e in spec]
    pick = [("tros.qkv", "light_b.0.qkv"), ("tros.o", "light_b.0.o"), ("vhid.qkv", "light_a.0.qkv"), ("emb", "emb"), ("head", "head")]
    parts = [f"{lab} {f[names.index(n)]:.2f}" for lab, n in pick if n in names and names.index(n) < f.size]
    dmp = ""
    if "df" in mm:                                                      # fix21: найсильніше загальмований тензор кроку
        d = np.asarray(mm["df"], np.float64).ravel()
        j = int(d.argmin())
        dmp = f" | damp min {d[j]:.2f} {names[j] if j < len(names) else j} n<1 {int((d < 1).sum())}"
    return f" | snr сер {f.mean():.2f} " + " ".join(parts) + dmp


def _gain_str(mm, spec):
    """13.11: підсилювачі картки кроку (top-3 і фінальні шари) + top-3 rel (|крок|/|w|) — щоб бачити «запинку»."""
    import numpy as np
    if "g" not in mm:
        return ""
    names = [e[0] for e in spec]
    g = np.asarray(mm["g"], np.float64).ravel()
    out = " | g " + " ".join(f"{names[i]} {g[i]:.2f}" for i in np.argsort(-g)[:3] if i < len(names))
    fin = [i for i, n in enumerate(names) if n.startswith("final.") and i < g.size]
    if fin:
        out += f" fin max {g[fin].max():.2f}"
    if "rel" in mm:
        r = np.asarray(mm["rel"], np.float64).ravel()
        out += " | rel " + " ".join(f"{names[i]} {r[i]:.4f}" for i in np.argsort(-r)[:3] if i < len(names))
    if "mrl_bank" in mm:                                                 # fix15: |банк| відкладеного руху m, обрізаних тензорів
        out += f" | mrl bank {float(np.ravel(mm['mrl_bank'])[0]):.3g} cut {float(np.ravel(mm['mrl_cut'])[0]):.0f}" + (f" drop {float(np.ravel(mm['mrl_drop'])[0]):.3g}" if "mrl_drop" in mm else "")
    return out


def _rowsum_ratio(m):
    """|Σ рядків| / Σ|рядок| таблиці (V, C): для градієнта/інерції softmax-голови ≡ 0 (Σ_i (p_i − y_i) = 0)."""
    import jax.numpy as jnp
    m = m.astype(jnp.float32)
    return jnp.linalg.norm(m.sum(0)) / (jnp.sqrt((m * m).sum(1)).sum() + 1e-30)


def _nullspace(st):
    """16.11 (fable52g): здоров'я нуль-простору softmax голів — друкується з кожним eval. Причина зламів 15000 і 34014+:
    градієнт голови переставав бути градієнтом softmax (Σp ≠ 1 через неузгоджене округлення логітів, ce.CE_LOGITS) і
    розганяв спільний зсув логітів. Знаки (за чекпойнтами): сума рядків інерції l_head1 0.0005 (30000, здорово) →
    0.12 (34000) → 0.99 (35250); head 0.21 (15000) → 0.35 (15250), здорово 0.0002; l_head0 σ1/σ2 1.05 → 1.83 → 4.3."""
    import jax
    import jax.numpy as jnp
    import numpy as np
    out = []
    m = st.get("opt", {}).get("m", {}) if isinstance(st, dict) else {}
    for k in ("l_head1", "head"):
        if k in m:
            out.append(f"сума рядків інерції {k} {float(jax.jit(_rowsum_ratio)(m[k])):.4f}")
    pr = st.get("params", {}) if isinstance(st, dict) else {}
    if "l_head0" in pr:
        sv = np.linalg.svd(np.asarray(pr["l_head0"].astype(jnp.float32)), compute_uv=False)
        out.append(f"l_head0 σ1/σ2 {sv[0]:.2f}/{sv[1]:.2f} = {sv[0] / max(sv[1], 1e-9):.2f}")
    return " | ".join(out) + " (softmax ⇒ суми ≈ 0; > 0.05 або σ1/σ2 > 1.5 — тривога)"


def main(argv=None):
    args = _args(argv)
    if args.cpu8:
        os.environ["XLA_FLAGS"] = os.environ.get("XLA_FLAGS", "") + " --xla_force_host_platform_device_count=8"
        os.environ.setdefault("JAX_PLATFORMS", "cpu")
    import numpy as np
    import jax
    import jax.numpy as jnp
    from . import model, dist, instinct, data, ckpt, bench_tpu, diag
    diag.start()
    from . import step as S

    compiles = []
    jax.monitoring.register_event_duration_secs_listener(
        lambda ev, d, **kw: compiles.append(d) if ev == "/jax/core/compile/backend_compile_duration" else None)
    t_start = time.time()
    plat = jax.devices()[0].platform
    if args.attn is None:
        args.attn = "splash" if plat == "tpu" else "dense"
    cfg = bench_tpu._cfg(args)
    W = args.chips or len(jax.devices())
    cfg = dataclasses.replace(cfg, W=W)
    if getattr(args, "rope_doc_mask", False):
        _eos = args.eos
        if _eos is None:
            try:
                _eos = data._read_eos(args.data_root, next(iter(json.loads(args.mix) if args.mix else MIX)))
            except Exception:                                            # noqa: BLE001
                _eos = EOS_DEFAULT
        cfg = dataclasses.replace(cfg, rope_doc_eos=int(_eos))
    fish_kinds = None
    if getattr(args, "fish_n", 0) > 0:
        _wf = json.loads(args.mix) if args.mix else dict(MIX)
        fish_kinds = {k: {s: _wf[s] for s in srcs if s in _wf} for k, srcs in FISH_KINDS.items()}
        fish_kinds = {k: v for k, v in fish_kinds.items() if v}
        cfg = dataclasses.replace(cfg, fish_n=args.fish_n, fish_hf=args.fish_hf, fish_pool=args.fish_pool,
                                  fish_c=args.fish_c, fish_kinds=len(fish_kinds),
                                  fish_center=bool(args.fish_center))
    if getattr(args, "heavy_all", False):
        cfg = dataclasses.replace(cfg, heavy_all=True)
    elif getattr(args, "heavy_full_every", 0) > 0:
        cfg = dataclasses.replace(cfg, heavy_rotate=True)
    if args.vocab:
        cfg = dataclasses.replace(cfg, V=args.vocab, short_v=args.short_v or cfg.short_v)
    rem_on = args.remind > 0
    if rem_on:
        cfg = dataclasses.replace(cfg, remind=True, remind_w=args.remind_w, remind_q=args.remind_q,
                                  remind_use_w=args.remind_use_w, remind_blank=bool(args.remind_blank),
                                  remind_tok0=int(args.remind_tok0),
                                  remind_cand=bool(getattr(args, "remind_cand", False)),
                                  remind_mask_main=bool(getattr(args, "remind_mask_main", False)))
    if getattr(args, "d_model", 0) > 0:
        cfg = dataclasses.replace(cfg, d=int(args.d_model))
    if getattr(args, "attn_gain", 1.0) != 1.0 or getattr(args, "prev_in", False):
        cfg = dataclasses.replace(cfg, attn_gain=float(args.attn_gain), prev_in=bool(args.prev_in))
    cfg = dataclasses.replace(cfg, rope_pcut=float(getattr(args, "rope_pcut", 0.0)),
                              rope_anchor_top=bool(getattr(args, "rope_anchor_top", False)),
                              rope_frac=float(getattr(args, "rope_frac", 0.0)), rope_cap=float(getattr(args, "rope_cap", 0.0)),
                              rope_key_diag=bool(getattr(args, "rope_key_diag", False)))
    if cfg.rope_anchor_top or cfg.rope_pcut > 0 or cfg.rope_frac > 0 or cfg.rope_cap > 0 or cfg.rope_key_diag:
        P(f"трос (13.11): anchor_top={cfg.rope_anchor_top} pcut={cfg.rope_pcut} rope_frac={cfg.rope_frac or cfg.frac} "
          f"rope_cap={cfg.rope_cap or cfg.cap} key_diag={cfg.rope_key_diag}")
    mesh = dist.make_mesh(W)
    tok_step = W * cfg.B * cfg.T
    stop = min(args.stop_at or args.steps, args.steps)
    P(f"jax {jax.__version__} {plat} W={W} {jax.devices()[0].device_kind}; d={cfg.d} E={cfg.E} heavy={cfg.heavy} "
      f"V={cfg.V} B/чип={cfg.B} T={cfg.T} dtype={cfg.dtype} remat={args.remat} uroboros={cfg.uroboros}"
      f"{f' budget={cfg.uroboros_budget}' if cfg.uroboros else ''} MOE_UNROLL={os.environ.get('MOE_UNROLL', '0')} "
      f"MOE_UP_MID={os.environ.get('MOE_UP_MID', '-')}")
    P(f"розклад: steps={args.steps} stop_at={stop} lr_peak={args.lr} warmup={args.warmup} sched={args.sched}; "
      f"{tok_step:,} токенів/крок")

    consts, _, lt = bench_tpu._consts(cfg, args)
    remat = tuple({"True": True, "False": False}.get(x, x)
                  for x in args.remat.split(",")) if "," in args.remat else args.remat
    model.remat_parts(remat)
    # PALLAS_CPU=1 — лише для перевірок на ПК: ядра Instinct у режимі interpret (та сама математика)
    use_pallas = (plat == "tpu" or os.environ.get("PALLAS_CPU") == "1") and not args.no_pallas and cfg.dtype == "bf16"
    icfg = instinct.InstinctCfg(lr_peak=args.lr, use_pallas=use_pallas, sr=cfg.dtype == "bf16",
                                m_bf16=cfg.dtype == "bf16")
    if args.opt_scale is not None:
        icfg = dataclasses.replace(icfg, scale=args.opt_scale)
    if getattr(args, "opt_gnorm", False):
        icfg = dataclasses.replace(icfg, gnorm=True)
    if getattr(args, "opt_snr", False):
        icfg = dataclasses.replace(icfg, snr=True, snr_ref=args.opt_snr_ref, snr_beta=args.opt_snr_beta,
                                   snr_floor=args.opt_snr_floor,
                                   snr_only=tuple(x for x in args.opt_snr_only.split(",") if x))
        P(f"Instinct: впевненість кроку (--opt_snr) ref {args.opt_snr_ref} beta {args.opt_snr_beta} floor {args.opt_snr_floor}")
    if getattr(args, "opt_lr_mul", ""):                                    # fix18
        _lm = tuple((x.split(":")[0], float(x.split(":")[1])) for x in args.opt_lr_mul.split(",") if x)
        icfg = dataclasses.replace(icfg, lr_mul=_lm)
        P(f"fix18: множник кроку Instinct {_lm}")
    if getattr(args, "opt_card_kind", False):
        if not fish_kinds:
            raise SystemExit("--opt_card_kind потребує фішок (видів кроку)")
        icfg = dataclasses.replace(icfg, card_by_kind=True, cards=len(fish_kinds) + 1)
        P(f"Instinct: картка = вид кроку ({list(fish_kinds)} + синтетика = {icfg.cards} карток)")
    if getattr(args, "opt_stream_proj", False):
        icfg = dataclasses.replace(icfg, stream_proj=True)
        cfg = dataclasses.replace(cfg, stream_m=True)
        P("Instinct: K1 — без складової вздовж спільного вектора потоку (--opt_stream_proj)")
    if getattr(args, "norm_dump_proj", False):
        if not getattr(args, "opt_stream_proj", False):
            raise SystemExit("--norm_dump_proj потребує --opt_stream_proj (EMA-напрям U у стані Instinct)")
        cfg = dataclasses.replace(cfg, norm_dump_proj=True, stream_m=True)
        P("fix14 F1: фінальна norm без складової вздовж U (--norm_dump_proj)")
    if getattr(args, "opt_m_rate", False):
        icfg = dataclasses.replace(icfg, m_rate=True, m_rate_kappa=args.opt_m_rate_kappa, m_rate_warm=args.opt_m_rate_warm)
        cfg = dataclasses.replace(cfg, stream_m=True)
        P(f"fix15: швидкість виходу m обмежена (--opt_m_rate) κ {args.opt_m_rate_kappa} розігрів {args.opt_m_rate_warm}")
        if getattr(args, "opt_m_rate_gate", False):
            icfg = dataclasses.replace(icfg, m_rate_gate=True)
            P("fix15: відкладений рух m — лише де свіжий градієнт згоден (--opt_m_rate_gate)")
    if getattr(args, "opt_gnorm_rawrel", False):
        icfg = dataclasses.replace(icfg, gnorm_rawrel=True)
        P("fix14: rel/винні від сирого градієнта при gnorm (--opt_gnorm_rawrel)")
    if getattr(args, "opt_snr_dz", False):
        if not getattr(args, "opt_snr", False):
            raise SystemExit("--opt_snr_dz потребує --opt_snr")
        icfg = dataclasses.replace(icfg, snr_dz=True, snr_horizon=args.opt_snr_horizon)
        P(f"fix14: dead-zone впевненості (--opt_snr_dz) H {args.opt_snr_horizon}")
    if getattr(args, "opt_norm_anchor", False):
        icfg = dataclasses.replace(icfg, norm_anchor=True)
        P("Instinct: K2 — 1D norm-ваги light/final тягнуться до 1 (--opt_norm_anchor)")
    if getattr(args, "opt_safe16", False):
        icfg = dataclasses.replace(icfg, safe16=True)
        P("fix16: Instinct — проєкція інерції проти градієнта + cautious + відносний крок норм (light_a/light_b/final/norm)")
    if getattr(args, "opt_wd_norm", 0.0) > 0:
        icfg = dataclasses.replace(icfg, wd_norm=float(args.opt_wd_norm))
        P(f"Instinct: weight decay до 0 для 1D норм-ваг = {args.opt_wd_norm} (--opt_wd_norm)")
    if getattr(args, "opt_surge_thr", 0.0) > 0:                             # fix20
        icfg = dataclasses.replace(icfg, surge_thr=float(args.opt_surge_thr))
        P(f"fix20: гальмо несподіванки thr {args.opt_surge_thr}")
    if getattr(args, "opt_damp", 0.0) > 0:                                  # fix21
        icfg = dataclasses.replace(icfg, damp=float(args.opt_damp), damp_p=float(args.opt_damp_p))
        P(f"fix21: гасіння сплеску градієнта по тензору: поріг {args.opt_damp}× типової норми, степінь {args.opt_damp_p}")
    if getattr(args, "inst_word", False):                                     # 15.11 Instinct на словах
        cfg = dataclasses.replace(cfg, inst_word=True, iw_rewrite=True, iw_random=False)
        P(f"inst_word: переписана відповідь — норма ≥0.5 нічого; поза нормою: правильному +δ' (δ={cfg.iw_delta}), "
          f"top-{cfg.iw_k} конкурентам −δ'·p_i/Σp (δ'=min(δ,Σp), Σq=1)")
    if getattr(args, "opt_rollback", False):                                  # 15.11 запобіжник-відкатник
        _fn = int(getattr(cfg, "fish_n", 0) or 0)
        _fk = int(getattr(args, "fish_c", 2) or 2) if _fn > 0 else 0
        icfg = dataclasses.replace(icfg, rollback=True, rb_fish_n=_fn, rb_fish_k=_fk)
        if os.environ.get("RB_TEST_SPIKE"):                                   # перевірка: відкатник без розігріву
            icfg = dataclasses.replace(icfg, rb_warm=3)
        P(f"відкатник: копія лише ствола + {_fk} активних фішок з {_fn}")
        P(f"відкатник: loss > {icfg.rb_ratio}× звичного → відкат на стан до попереднього кроку, крок ×{icfg.rb_lr}")
    if getattr(args, "opt_err_step", False):                                  # 15.11 задум Стефана
        icfg = dataclasses.replace(icfg, err_step=True, damp=max(float(getattr(args, "opt_damp", 0.0)), 2.0),
                                   err_cap=float(getattr(args, "opt_err_cap", 3.0)))
        P(f"err_step: крок = lr·(похибка/очікувана)·(звична ||g||/||g||), ≤ {icfg.err_cap} (задум Стефана, 15.11)")
    if getattr(args, "opt_gearbox", False):                                   # 15.11 «механіка» Стефана
        icfg = dataclasses.replace(icfg, gearbox=True, err_step=False,
                                   damp=max(float(getattr(args, "opt_damp", 0.0)), 2.0))
        P(f"коробка передач: {instinct.GEARS}, передачі лише по одній: вниз при ||g||>{icfg.gb_down}× звичної, "
          f"вгору коли спокійно (<{icfg.gb_up}×) і розігнався до поточної; множник кроку плавно ≤{icfg.gb_acc} за крок; "
          f"камера: понад 1.0 і loss гірший за очікуваний → штраф, вниз; нуля (зчеплення) немає")
    if getattr(args, "opt_g_max", 3.0) != 3.0:
        icfg = dataclasses.replace(icfg, g_max=float(args.opt_g_max))
        P(f"fix16: підсилювачі карток ≤ {args.opt_g_max}")
    if getattr(args, "hn_guard_w", 0.0) > 0:
        cfg = dataclasses.replace(cfg, hn_guard_w=float(args.hn_guard_w), hn_guard_mean=float(args.hn_guard_mean),
                                  hn_guard_len=float(args.hn_guard_len))
        P(f"fix17: штраф стовбура w {args.hn_guard_w}: |сер. hn| > {args.hn_guard_mean}, |hn| > {args.hn_guard_len}")
    if getattr(args, "safe_norm_ratio", 0.0) > 0:
        cfg = dataclasses.replace(cfg, safe_norm_ratio=float(args.safe_norm_ratio), safe_norm_t0=int(args.safe_norm_t0))
        P(f"fix16: спільний вектор після фінальної norm ≤ {args.safe_norm_ratio}·сигнал (причинно, з позиції {args.safe_norm_t0})")
    if getattr(args, "opt_culprit", "abs") != "abs":
        icfg = dataclasses.replace(icfg, culprit=args.opt_culprit)
        P(f"Instinct: винні на поганій несподіванці — {args.opt_culprit} (перевищення власного середнього rel)")
    P(f"Instinct: scale={icfg.scale} wd={icfg.weight_decay} beta={icfg.beta} pallas={use_pallas}; "
      f"rope={cfg.rope} read={cfg.rope_read} pallas={cfg.rope_pallas} route={cfg.rope_route} layers={cfg.rope_layers} "
      f"frac={cfg.frac} cap={cfg.cap}")

    # --- ваги: одразу шардовані (як bench_tpu без --host_init)
    t0 = time.perf_counter()
    key = jax.random.PRNGKey(args.seed)
    shapes = jax.eval_shape(lambda k: model.init_params(k, cfg, lt), key)
    shard = jax.tree_util.tree_map(lambda s: dist.NamedSharding(mesh, s), dist.param_specs(shapes),
                                   is_leaf=lambda s: isinstance(s, dist.P))
    diag.phase("перед init_params")
    params = jax.jit(lambda k: model.init_params(k, cfg, lt), out_shardings=shard)(key)
    if diag.ON:
        jax.block_until_ready(params)
        diag.phase("init_params готові")
    n_par = sum(x.size for x in jax.tree_util.tree_leaves(params))
    spec = instinct.make_spec(params)
    ts_synth = None
    if rem_on and args.remind_synth > 0:
        ts, state, place, ts_synth = dist.make_step(cfg, consts, mesh, params, spec, icfg, remat=remat,
                                                    alt_cfg=dataclasses.replace(cfg, remind_only=True,
                                                                                remind_w=(args.remind_w_synth if args.remind_w_synth >= 0
                                                                                          else cfg.remind_w),
                                                                                remind_mean=bool(args.remind_pack),
                                                                                remind_strict=bool(args.remind_strict
                                                                                                   and args.remind_pack),
                                                                                remind_strict_w=args.remind_strict_w))
    else:
        ts, state, place = dist.make_step(cfg, consts, mesh, params, spec, icfg, remat=remat)
    if diag.ON:
        jax.block_until_ready(state)
        diag.phase("стан Instinct і крок готові (make_step)")
    # eval: лише fwd (та сама shard_map-втрата; tau/стан не змінюються)
    # 12.11: без фінальної фази (--fish_full_from < 0) eval іде тим самим числом фішок, що й тренування (fish_c);
    # раніше рядок ішов через увесь набір (17 фішок ланцюжком) — конфігурацію, якої модель ніколи не тренувала
    _ec = cfg.fish_c if getattr(cfg, "fish_n", 0) > 0 and args.fish_full_from < 0 else 0
    lf = dist.make_loss_fn(dataclasses.replace(cfg, fish_eval=True, fish_eval_c=_ec) if getattr(cfg, "fish_n", 0) > 0
                           else cfg, consts, mesh, params, remat)
    if getattr(cfg, "fish_n", 0) > 0:
        P(f"eval фішок: {'по ' + str(_ec) + ' фішки виду (як тренування)' if _ec else 'увесь набір виду (готова модель)'}")
    del params
    if rem_on:                                                          # та сама одна eval-програма + лічильники голок
        eval_fn = jax.jit(lambda p, xy, prep, tau: (lambda o: (o[1]["main"], o[2]["remind_hit"], o[2]["remind_n"],
                                                                         o[2]["remind_nll"]))(
            lf(p, xy, prep, tau)[1]))
    else:
        eval_fn = jax.jit(lambda p, xy, prep, tau: lf(p, xy, prep, tau)[1][1]["main"])
    eval_fn_all = None
    if getattr(cfg, "heavy_rotate", False) and not getattr(cfg, "heavy_all", False):
        # режим готової моделі: усі важкі шари на всіх словах (стрибок до фінальної фази видно з першого eval)
        cfg_all = dataclasses.replace(cfg, heavy_all=True)
        lf_all = dist.make_loss_fn(cfg_all, consts, mesh, state["params"], remat)
        if rem_on:
            eval_fn_all = jax.jit(lambda p, xy, prep, tau: (lambda o: (o[1]["main"], o[2]["remind_hit"], o[2]["remind_n"],
                                                                             o[2]["remind_nll"]))(lf_all(p, xy, prep, tau)[1]))
        else:
            eval_fn_all = jax.jit(lambda p, xy, prep, tau: lf_all(p, xy, prep, tau)[1][1]["main"])
    pnorm_fn = jax.jit(lambda p: jnp.sqrt(sum(jnp.sum(jnp.square(x.astype(jnp.float32)))
                                              for x in jax.tree_util.tree_leaves(p))))
    jax.block_until_ready(state)
    P(f"параметрів {n_par / 1e9:.3f}B, init {time.perf_counter() - t0:.1f} с")
    diag.phase("ініціалізація завершена")

    # --- дані
    weights = json.loads(args.mix) if args.mix else dict(MIX)
    rank, world = jax.process_index(), jax.process_count()
    eos = args.eos
    if eos is None:
        try:
            eos = data._read_eos(args.data_root, next(iter(weights)))
        except Exception:                                               # noqa: BLE001 — маніфестів у бакеті немає
            eos = EOS_DEFAULT
    # токени справжніх даних мають влазити у словник (small/tiny_k40 мають V=4096 — тоді --vocab 131328)
    assert eos < cfg.V, f"eos {eos} ≥ V {cfg.V}: справжні дані не влазять у словник моделі (дай --vocab 131328)"
    eval_cache = args.cache.rstrip("/") + "_eval"
    ho = Heldout(data, args.heldout_tok, eval_cache)
    ho.eos = eos
    for s in weights:
        ho.last[s] = data._ls(args.data_root, s)[-1]
    t0 = time.perf_counter()
    if fish_kinds:
        from . import fishki as _fk
        corpus = data.KindCorpus(fish_kinds, _fk.pools(cfg), cfg.fish_c, root=args.data_root, seq=cfg.T, seed=args.seed,
                                 rank=rank, world=world, cache=args.cache, eos=eos,
                                 doc_seed=(args.data_reseed if getattr(args, "data_reseed", -1) >= 0 else None))
        P(f"фішки: {cfg.fish_n} × {2 * cfg.d * cfg.fish_hf / 1e9:.3f}B, видів {cfg.fish_kinds} {list(fish_kinds)}, "
          f"набір {cfg.fish_pool}, за крок {cfg.fish_c}, фінальна фаза з кроку {args.fish_full_from}")
    else:
        corpus = data.Corpus(weights, root=args.data_root, seq=cfg.T, seed=args.seed, rank=rank, world=world,
                             cache=args.cache, eos=eos)
    rows = W * cfg.B // world                                           # рядків цього хоста
    prep_fn = data.induct_prep(cfg.V, consts["lock_bucket"])
    rem_x = None
    if rem_on:
        _L = 2 * args.remind_q + 2
        _mx = cfg.T + 1 - 2 * _L
        # найбільше вікно моделі: важкі й light_a — cfg.window, трос light_b — cfg.rope_win (≤ window)
        _w = max(cfg.window, getattr(cfg, "rope_win", 0))
        dmin = _w + _L                                                  # голка завжди за вікном (із запасом блоку)
        assert dmin <= _mx, f"T={cfg.T} замале: голка за вікном {_w} потребує відстані {dmin} > {_mx}"
        lad = None
        if args.remind_ladder == "auto":                                # ×1.5, ×2, ×4, … вікна, до кінця рядка
            lad = sorted({v for v in (int(1.5 * _w), 2 * _w, 4 * _w, 8 * _w, 16 * _w, 32 * _w, 64 * _w, 128 * _w)
                          if dmin < v < _mx} | {_mx})
        elif args.remind_ladder:
            lad = [int(v) for v in args.remind_ladder.split(",")]
        rem_x = data.Reminders(args.remind, cfg.T, n=args.remind_q, dist=args.remind_dist or dmin,
                               tok0=args.remind_tok0, seed=args.seed, V=cfg.V, p0=args.remind_p0,
                               warm=args.remind_warm, decay=args.remind_decay, ladder=lad, thr=args.remind_thr,
                               win=args.remind_win, lag=3 + 2,
                               dmin=0 if (lad and args.remind_ladder != "auto") else dmin, synth=args.remind_synth,
                               pack=args.remind_pack and args.remind_synth > 0,
                               pack_round=args.remind_pack_round, pack_mult=W,
                               doc_eos=getattr(cfg, "rope_doc_eos", -1),
                               near=args.remind_near, near_max=max(0, _w - _L),
                               strict=bool(args.remind_strict and args.remind_pack),
                               pack_frac=[float(v) for v in args.remind_pack_frac.split(",")] if args.remind_pack_frac
                               else None, win_ref=_w, blank=bool(args.remind_blank),
                               tail=bool(getattr(args, "remind_tail", False)))
        if args.remind_mix:
            rem_x.mix_fracs = [float(v) / 100.0 for v in args.remind_mix.split(",")]
            rem_x.mix_len = int(args.remind_mix_len)
            P(f"нагадування: плавний перехід після кроку {args.remind_synth}: синтетичних кроків "
              f"{args.remind_mix}% по {args.remind_mix_len} кр., далі {args.remind_mix.split(',')[-1]}% назавжди")
        if args.remind_p0 is not None or lad:
            P(f"нагадування: розклад p {args.remind_p0} ({args.remind_warm} кр.) → {args.remind} за {args.remind_decay}; "
              f"штраф троса {args.remind_use_w}; драбина {lad} поріг {args.remind_thr} вікно {args.remind_win} відповідей")
        P(f"нагадування: p={args.remind} пар {args.remind_q} відстань ≥{rem_x.dist} токени {rem_x.k0}..{rem_x.mark} "
          f"вага {args.remind_w}" + (f" (синтетичні кроки {args.remind_w_synth})" if args.remind_w_synth >= 0 else ""))
        _pf = prep_fn
        prep_fn = lambda x: dict(_pf(x), rem=np.zeros(x.shape, np.int8),    # eval-набори без голок: маска нулі
                                 **({"rem_y": np.zeros(x.shape, np.int32)} if args.remind_blank else {}))
        mx = cfg.T + 1 - 2 * rem_x.L
        rem_dists = ([int(v) for v in args.remind_eval.split(",")] if args.remind_eval else
                     sorted({min(v, mx) for v in (_w // 2, dmin, 2 * _w, 4 * _w, mx)}))
        assert all(0 <= v <= mx for v in rem_dists), (rem_dists, mx)
        P(f"eval пригадування на відстанях {rem_dists} (вікно {cfg.window}; ціль 95%+ за межами вікна)")
        if rem_x.pack:
            P(f"синтетика короткими рядками (--remind_pack): рівні (пар, сміття) {rem_x.plevels}; eval pack_fN: {args.remind_q} пар, "
              f"сміття {rem_x.pfill}")
            P("  справжнє сміття сходинок (після округлення до {}; вікно {}): {}".format(
                rem_x.pack_round, _w, [(pp, rem_x.real_fill(pp, f), f"{rem_x.real_fill(pp, f) / _w:.0%}")
                                       for pp, f in rem_x.plevels]))
    if rem_x is not None and getattr(args, "remind_same_fmt", False):
        rem_x.same_fmt = True
        P("нагадування: голки в тексті — той самий формат, що синтетика етапу 1"
          + (" (відповіді приховані)" if rem_x.blank else ""))
    if rem_x is not None and (getattr(args, "remind_per_row", 1) > 1 or getattr(args, "remind_far_log", -1.0) >= 0):
        assert rem_x.same_fmt, "--remind_per_row/--remind_far_log потребують --remind_same_fmt"
        rem_x.per_row = max(1, int(args.remind_per_row))
        rem_x.far_log = float(args.remind_far_log) if args.remind_far_log >= 0 else 1.0 - rem_x.near
        P(f"нагадування: у рядку тексту {rem_x.per_row} голок; {rem_x.far_log:.0%} за вікном (лог-рівномірно "
          f"{rem_x.dist}..{cfg.T + 1 - 2 * 2 * rem_x.n}), решта у вікні 0..{rem_x.near_max}")
    if rem_x is not None and getattr(args, "remind_far_frac", -1.0) >= 0:
        rem_x.far_frac, rem_x.far_win = float(args.remind_far_frac), int(cfg.window)
        P(f"нагадування: останній рівень — {rem_x.far_frac:.0%} батчів зі сміттям за вікном {cfg.window} (лише трос)")
    hl = data.HostLoader(corpus, rows, prep_fn=_pf if rem_on else prep_fn, depth=3, xform=rem_x)
    P(f"корпус: {len(weights)} джерел, eos={eos}, rank {rank}/{world}, суміш "
      + " ".join(f"{k}={v:g}" for k, v in weights.items()))

    _rb_test = int(os.environ.get("RB_TEST_SPIKE", "0"))   # 15.11: перевірка відкатника — lr ×300 на одному кроці

    def lr_of(i):
        _l = instinct.lr_at(i, args.lr, args.steps, warmup=args.warmup, sched=args.sched)
        return _l * float(os.environ.get("RB_TEST_MUL", "300")) if (_rb_test and i == _rb_test) else _l

    rot = bool(getattr(cfg, "heavy_rotate", False)) and not getattr(cfg, "heavy_all", False)

    def hfull_of(i):
        k = args.heavy_full_every
        return (i // k) % cfg.heavy if rot and i % k == 0 else -1

    def mk(xy, prep, lr, hf=-1):
        if getattr(cfg, "fish_n", 0) > 0 and "fish" not in prep:          # eval: вид вибирає роутер (fish_eval)
            prep = dict(prep, fish=np.zeros((xy.shape[0], 1), np.int32), kind=np.zeros((xy.shape[0], 1), np.int32))
        if rot:
            prep = dict(prep, hfull=np.full((xy.shape[0], 1), hf, np.int32))
        return place(dict(xy=xy, prep=prep, lr=np.float32(lr)))

    # held-out eval-набори: фіксовані батчі, підготовлені один раз
    evals = {}
    for nm, w in EVAL_SETS.items():
        w = {k: v for k, v in (w or weights).items() if k in weights}
        if not w:
            continue
        ec = data.Corpus(w, root=args.data_root, seq=cfg.T, seed=10_000 + args.seed, rank=0, world=1,
                         cache=eval_cache, eos=eos)
        for src in ec.sources:                                          # лише хвіст останнього шматка
            src.names = [ho.last[src.src]]
            src._order()
        evals[nm] = []
        for _ in range(args.eval_batches):
            xy = ec.batch(rows)
            evals[nm].append((xy, prep_fn(xy[:, :-1])))
            if rem_on and nm == next(iter(EVAL_SETS)):  # набори «remind_dN»: ті самі рядки, голка в кожному, відстань рівно N
                for dd in rem_dists:
                    if args.remind_synth > 0:                           # синтетика етапу 1 на тій самій відстані
                        xr, ex = rem_x(xy, p=1.0, exact=dd, synth=True)
                        evals.setdefault(f"synth_d{dd}", []).append((xr, data.mask_induct(dict(_pf(xr[:, :-1]), **ex))))
                    xr, ex = rem_x(xy, p=1.0, exact=dd)
                    if rem_x.pack and dd == rem_dists[0]:               # eval як torch RECALL: q пар, кожна довжина сміття
                        for ff in rem_x.pfill:
                            xp, exp_ = rem_x(xy, synth=True, pack=(args.remind_q, ff))
                            evals.setdefault(f"pack_f{ff}", []).append((xp, data.mask_induct(dict(_pf(xp[:, :-1]), **exp_))))
                    evals.setdefault(f"remind_d{dd}", []).append((xr, data.mask_induct(dict(_pf(xr[:, :-1]), **ex))))
    held = {s: ho.tails[(s, ho.last[s])][2] for s in weights if (s, ho.last[s]) in ho.tails}
    P(f"held-out: {sum(held.values()):,} токенів з {len(held)} джерел; eval-набори "
      + ", ".join(f"{k}×{len(v)}" for k, v in evals.items()) + f"; дані готові за {time.perf_counter() - t0:.1f} с")

    def run_eval(st, i):
        t0 = time.perf_counter()
        out = {}
        for nm, bs in evals.items():
            v = []
            hits = []
            for xy, prep in bs:
                b = mk(xy, prep, 0.0)
                r = eval_fn(st["params"], b["xy"], b["prep"], st["tau"])
                if rem_on:
                    r, hh, nn, nl = r
                    hits.append((float(hh), float(nn), float(nl)))
                v.append(float(r))
            out[nm] = float(np.mean(v))
            if nm == "mix" and getattr(args, "eval_nofish", False) and "fish" in st["params"]:   # 14.11: внесок фішок
                _pz = dict(st["params"]); _fz = dict(_pz["fish"]); _fz["down"] = jnp.zeros_like(_fz["down"]); _pz["fish"] = _fz
                _w = []
                for xy, prep in bs:
                    b = mk(xy, prep, 0.0)
                    r = eval_fn(_pz, b["xy"], b["prep"], st["tau"])
                    _w.append(float(r[0] if rem_on else r))
                out["mix_nofish"] = float(np.mean(_w))
                out["fish_gain"] = out["mix_nofish"] - out[nm]
            if nm.startswith(("remind_d", "synth_d", "pack_f")):                               # точність = p(значення)>0.5; nll на відповідь
                den = max(sum(h[1] for h in hits), 1e-9)
                out[nm + "_acc"] = sum(h[0] for h in hits) / den
                out[nm + "_ansnll"] = sum(h[2] for h in hits) / den
            if cfg.rope:
                # абляція пам'яті: поріг якорів троса 1e30 → жоден якір не відкривається, лишається тільки вікно
                # (перевірено _tmp/ready_check/reach_test.py). Та сама програма (tau — аргумент), без компіляції.
                # nomem − з пам'яттю > 0 означає, що модель справді користується далеким читанням.
                t_off = jax.device_put(jnp.asarray(np.asarray(jax.device_get(st["tau"]), np.float32) * np.array(
                    [1.0, 0.0] + [1.0] * (st["tau"].shape[0] - 2), np.float32) + np.array(
                    [0.0, 1e30] + [0.0] * (st["tau"].shape[0] - 2), np.float32)), st["tau"].sharding)
                w = [eval_fn(st["params"], mk(xy, prep, 0.0)["xy"], mk(xy, prep, 0.0)["prep"], t_off)
                     for xy, prep in bs]
                if rem_on:
                    if nm.startswith(("remind_d", "synth_d", "pack_f")):
                        out[nm + "_acc_nomem"] = sum(float(r[1]) for r in w) / max(sum(float(r[2]) for r in w), 1e-9)
                    w = [r[0] for r in w]
                out[nm + "_nomem"] = float(np.mean([float(r) for r in w]))
            if eval_fn_all is not None:                                 # готова модель: усі важкі шари на всіх словах
                w = []
                for xy, prep in bs:
                    b = mk(xy, prep, 0.0)
                    w.append(eval_fn_all(st["params"], b["xy"], {k: v for k, v in b["prep"].items() if k != "hfull"},
                                         st["tau"]))
                if rem_on:
                    if nm.startswith(("remind_d", "synth_d", "pack_f")):
                        out[nm + "_acc_allfull"] = sum(float(r[1]) for r in w) / max(sum(float(r[2]) for r in w), 1e-9)
                    w = [r[0] for r in w]
                out[nm + "_allfull"] = float(np.mean([float(r) for r in w]))
        P(f"eval step {i} " + " ".join(f"{k}={v:.4f}" for k, v in out.items())
          + f" ({time.perf_counter() - t0:.1f} с)")
        if rem_on:
            P(f"RECALL step {i} " + " | ".join(
                f"d{dd}: {out[f'remind_d{dd}_acc']:.1%}" + (f" (nomem {out[f'remind_d{dd}_acc_nomem']:.1%})"
                                                          if f"remind_d{dd}_acc_nomem" in out else "")
                for dd in rem_dists))
            if args.remind_synth > 0:
                P(f"RECALL_SYNTH step {i} " + " | ".join(
                    f"d{dd}: {out[f'synth_d{dd}_acc']:.1%}" + (f" (nomem {out[f'synth_d{dd}_acc_nomem']:.1%})"
                                                              if f"synth_d{dd}_acc_nomem" in out else "")
                    for dd in rem_dists))
            if rem_x.pack:
                P(f"RECALL_PACK step {i} " + " | ".join(
                    f"сміття {ff}: {out[f'pack_f{ff}_acc']:.1%}" + (f" (nomem {out[f'pack_f{ff}_acc_nomem']:.1%})"
                                                                   if f"pack_f{ff}_acc_nomem" in out else "")
                    for ff in rem_x.pfill))
        try:                                                    # 16.11 (fable52g): ранній знак зламу голів, див. _nullspace
            P(f"NULLSPACE step {i}: " + _nullspace(st))
        except Exception as e:                                  # діагностика не має права зупиняти тренування
            P(f"NULLSPACE step {i}: не пораховано ({type(e).__name__}: {str(e)[:160]})")
        return out

    # --- resume
    os.makedirs(args.ckpt_local, exist_ok=True)
    for n in os.listdir(args.ckpt_local):                               # недописані записи вбитого процесу
        if n.startswith("step_") and n.endswith(".tmp"):
            shutil.rmtree(os.path.join(args.ckpt_local, n), ignore_errors=True)
    store = make_store(args.ckpt_gcs)
    rid = run_id(vars(args), weights)
    P(f"run_id {rid}")
    describe = lambda ex: run_diff(ex, vars(args), weights)
    # Локальні ckpt, які цей прогін не має права взяти (чужий run_id; або старт БЕЗ --resume — тоді всі), відкладаємо
    # вбік: інакше resume взяв би чужі ваги, а cleanup лишав би чужий «найновіший» і стирав наші. Не видаляємо.
    for s in _ckpt_dirs(args.ckpt_local):
        src = os.path.join(args.ckpt_local, f"step_{s}")
        ex = local_extra(src)
        if args.resume and (extra_run_id(ex) == rid or args.resume_force):
            continue
        dst = os.path.join(args.ckpt_local, f"foreign_step_{s}_{extra_run_id(ex) or 'unknown'}_{int(time.time())}")
        os.replace(src, dst)
        P(f"УВАГА: локальний ckpt step {s} не цього прогону → відкладено в {dst} "
          f"({describe(ex) if args.resume else 'старт без --resume'}); видали вручну, якщо не потрібен")
    if store is not None and not args.resume and not args.resume_force:
        mine = [s for s in store.steps() if extra_run_id(store.extra(s)) == rid]
        if mine:
            raise SystemExit(f"[train] ВІДМОВА: у сховищі {args.ckpt_gcs} вже є ckpt цього самого прогону (кроки {mine}), "
                             f"а --resume не задано: після переривання resume взяв би старі ваги. "
                             f"Дай --resume або інший --ckpt_gcs.")
    up = Uploader(store, run_id=rid)
    saver = BackgroundSaver(ckpt, args.ckpt_local, up)
    P(f"стан ckpt: {ckpt.state_bytes(state) / 2**30:.2f} GiB")
    i0 = 0
    if args.resume:
        loc = _ckpt_dirs(args.ckpt_local)
        path = os.path.join(args.ckpt_local, f"step_{loc[-1]}") if loc else (
            fetch_store_latest(store, args.ckpt_local, None if args.resume_force else rid, describe) if store else None)
        if path is None:
            P("resume: ckpt немає — старт з нуля")
        else:
            t0 = time.perf_counter()
            state, ds, extra = restore_sharded(path, state, reset_cards=getattr(args, "opt_card_kind", False))
            # 12.11: і для --remind_pack (сходинки синтетики) — раніше лише при ladder, тож після resume сходинки
            # починались з рівня 0
            if rem_x is not None and (rem_x.ladder or rem_x.pack) and isinstance(ds, dict) and "_remind" in ds:
                rem_x.load(ds["_remind"])
                P(f"resume нагадувань: рівні {rem_x.changes}"
                  + (f", сходинки синтетики {rem_x.pchanges}" if rem_x.pack else ""))
            elif rem_x is not None and rem_x.pack and getattr(args, "remind_pack_level", -1) >= 0:
                _pl = min(int(args.remind_pack_level), len(rem_x.plevels) - 1)
                rem_x.pchanges = [(0, _pl)]
                P(f"resume нагадувань: стану сходинок у ckpt немає → --remind_pack_level {_pl} "
                  f"{rem_x.plevels[_pl]}")
            elif rem_x is not None and rem_x.pack:
                P("УВАГА: у ckpt немає стану сходинок синтетики — почнуться з рівня 0 "
                  "(задай --remind_pack_level, щоб продовжити з потрібного)")
            if getattr(args, "data_reseed", -1) >= 0 and isinstance(ds, dict) and "subs" in ds:
                ds = dict(ds, subs=None)
                P(f"resume: дані з нового зерна {args.data_reseed} (--data_reseed): розклад видів той самий, документи інші")
            hl.load_state_dict(ds)
            if getattr(args, "opt_reset_gain", False) and "opt" in state and "gain" in state["opt"]:
                _g = state["opt"]["gain"]
                state = dict(state, opt=dict(state["opt"], gain=jax.device_put(jnp.ones(_g.shape, _g.dtype), _g.sharding)))
                P("resume: підсилювачі карток Instinct скинуто до 1 (--opt_reset_gain)")
            # 15.11 (fable52f): OPT_RESET_EXPECT=1 — для resume з ckpt, записаного кодом з ІНШИМ складом втрати (напр.
            # step_34250 від main5_1115_1900: старий доданок iw ≈ −1.8 → expect і rb_lema занижені; main5_1115_2013 після
            # такого resume: s<0 щокроку і 17 відкатів за 130 кроків). Скидає очікування карток і звичну втрату
            # відкатника: перший крок кожної картки задає рівень заново. Для чистого step_34000 НЕ потрібно.
            if os.environ.get("OPT_RESET_EXPECT") and "opt" in state:
                _o = dict(state["opt"])
                for _k in ("expect", "seen", "rb_lema"):
                    if _k in _o:
                        _o[_k] = jax.device_put(jnp.zeros(_o[_k].shape, _o[_k].dtype), _o[_k].sharding)
                for _k in ("rb_cool", "rb_valid", "rb_rep"):
                    if _k in _o:
                        _o[_k] = jax.device_put(jnp.zeros(_o[_k].shape, bool), _o[_k].sharding)
                state = dict(state, opt=_o)
                P("resume: очікування карток і звичну втрату відкатника скинуто (OPT_RESET_EXPECT)")
            if getattr(args, "opt_gnorm_rawrel", False) and "opt" in state and "rel_avg" in state["opt"]:
                _r = state["opt"]["rel_avg"]
                state = dict(state, opt=dict(state["opt"], rel_avg=jax.device_put(jnp.zeros(_r.shape, _r.dtype), _r.sharding)))
                P("resume: rel_avg Instinct скинуто (--opt_gnorm_rawrel)")
            if getattr(args, "fish_reset", False) and "fish" in state["params"]:
                from . import fishki as _fk
                _fp = state["params"]["fish"]
                _sh = {k: v.sharding for k, v in _fp.items()}
                _new = jax.jit(lambda k: _fk.init(k, cfg), out_shardings=_sh)(jax.random.PRNGKey(args.seed + 1414))
                _par = dict(state["params"]); _par["fish"] = _new
                _opt = dict(state["opt"])
                for _key in ("m", "cs"):                                 # плоскі словники name → масив (instinct.init)
                    if isinstance(_opt.get(_key), dict):
                        _d = dict(_opt[_key]); _z = [n for n in _d if str(n).startswith("fish")]
                        for n in _z: _d[n] = jnp.zeros_like(_d[n])
                        _opt[_key] = _d
                        if _z: P(f"resume: стан Instinct фішок обнулено ({_key}: {_z})")
                state = dict(state, params=_par, opt=_opt)
                P("resume: фішки з початкових ваг (--fish_reset)")
            i0 = int(np.asarray(jax.device_get(state["step"])))
            P(f"resume з {path}: step {i0}, tau {float(np.ravel(state['tau'])[0]):.4f} ({time.perf_counter() - t0:.1f} с)")

    def save(st, i, final=False):
        """Цикл чекає лише копію стану на хост: наступний крок донатить буфери, тож без другої копії в HBM інакше не можна.
        Запис і завантаження — у фоні. Якщо попередній запис ще триває, цей ckpt пропускається (крім останнього)."""
        if final:
            saver.wait()
        elif saver.busy():
            saver.skipped += 1
            P(f"ckpt step {i}: пропущено, попередній запис ще триває (усього пропусків {saver.skipped})")
            return
        t0 = time.perf_counter()
        tree = ckpt.snapshot_pinned(st) if os.environ.get("CKPT_PINNED") == "1" else None
        if tree is None:
            tree = ckpt.snapshot(st)
        ds = hl.state_dict()
        if rem_x is not None and (rem_x.ladder or rem_x.pack):
            ds = dict(ds, _remind=rem_x.state())
        saver.submit(tree, i, ds, dict(args=vars(args), mix=weights, run_id=rid))
        P(f"ckpt step {i}: цикл стояв {time.perf_counter() - t0:.2f} с (знімок на хост)")

    if getattr(args, "eval_at_resume", False) and i0 > 0 and not getattr(args, "eval_only", False):
        P(f"eval чекпойнта step {i0} перед навчанням (--eval_at_resume)")
        run_eval(state, i0)
    if getattr(args, "eval_only", False):                                # 12.11: правильний eval старого ckpt
        P(f"eval_only: ckpt step {i0}, без тренування")
        run_eval(state, i0)
        P("RESULT " + json.dumps(dict(status="EVAL_ONLY", step=i0)))
        hl.close()
        return True
    # --- цикл
    lossf = open(args.loss_log, "a") if args.loss_log else None
    times, ema, n_comp0 = [], None, len(compiles)
    win_t, win_tok, win_wait, win_n = 0.0, 0, 0.0, 0
    syn_far, syn_near = [0.0, 0.0], [0.0, 0.0]                         # синтетика: влучання/відповіді за вікном і у вікні
    first_ms = None
    nan_at = int(os.environ.get("TRAIN_NAN_AT", "0"))
    i = i0
    status = "OK"
    _n_text = 0
    while i < stop:
        tw = time.perf_counter()
        if fish_kinds and args.fish_full_from >= 0 and i == args.fish_full_from and not corpus.full:
            corpus.full = True                                           # фінальна фаза: увесь набір виду
            P(f"step {i}: фішки — фінальна фаза (увесь набір виду, {cfg.fish_pool} фішок)")
        xy, prep = hl.next()
        is_syn = ts_synth is not None and rem_x is not None and rem_x.synth_at(i)
        _lr = lr_of(i)
        if getattr(args, "text_lr_ramp", 0) > 0 and ts_synth is not None and not is_syn:
            _n_text += 1                                                # ПРОПОЗИЦІЯ: м'який вхід тексту після етапу 1
            _lr = _lr * min(1.0, _n_text / args.text_lr_ramp)
        batch = mk(xy, prep, _lr, hfull_of(i))
        t_wait = time.perf_counter() - tw
        t0 = time.perf_counter()
        if is_syn:                                                      # етап 1 / синтетичний крок плавного переходу
            state, m = ts_synth(state, batch)
        else:
            if ts_synth is not None and i == args.remind_synth:
                P(f"step {i}: етап 1 (синтетика) завершено → текст з нагадуваннями"
                  + (f"; плавний перехід --remind_mix {args.remind_mix} по {args.remind_mix_len} кр." if rem_x.mix_fracs
                     else ""))
            state, m = ts(state, batch)
        if rem_x is not None and (rem_x.ladder or rem_x.pack):          # та сама синхронізація + 2 скаляри
            loss, rh, rn = (float(v) for v in jax.device_get((m["loss"], m["stats"]["remind_hit"],
                                                                m["stats"]["remind_n"])))
            if rem_x.pack and is_syn:                                   # короткі рядки: відповідей у батчі — з маски
                _na = float(np.asarray(prep["rem"]).sum())
                msg = rem_x.observe(i, rh / max(rn, 1e-9) * _na, _na)
                _pp, _fr = rem_x.pack_fill_at(i)                        # відстань питання→ключ ≥ сміття+2
                _acc = syn_far if _fr + 2 > cfg.window else syn_near
                _acc[0] += rh / max(rn, 1e-9) * _na
                _acc[1] += _na
            else:
                msg = rem_x.observe(i, rh * W * cfg.B * cfg.remind_q, rn * W * cfg.B * cfg.remind_q)
            if msg:
                P(f"step {i + 1}: {msg}")
        else:
            loss = float(m["loss"])                                     # синхронізація: мс/крок — справжні
        dt = time.perf_counter() - t0
        i += 1
        diag.step(i)
        if diag.ON and i == i0 + 1:
            diag.phase("перший крок (з компіляцією)")
        if nan_at and i == nan_at:                                      # перевірка запобіжника (env TRAIN_NAN_AT)
            loss = float("nan")
        if lossf:
            lossf.write(json.dumps(dict(step=i, loss=loss)) + "\n")
            lossf.flush()
        if first_ms is None:
            first_ms = dt * 1e3
            P(f"крок {i}: {dt:.1f} с (з компіляцією), loss {loss:.4f}, компіляцій {len(compiles) - n_comp0}")
        else:                                                           # крок компіляції — не у швидкості
            times.append(dt)
            win_t += dt + t_wait
            win_tok += tok_step
            win_wait += t_wait
            win_n += 1
        if not math.isfinite(loss):
            P(f"NAN step {i}: loss={loss} — зупинка (стан НЕ зберігаю)")
            status = "NAN"
            break
        if ema is not None and i > 50 and loss > args.spike * ema:
            P(f"SPIKE step {i}: loss {loss:.4f} > {args.spike}×EMA {ema:.4f}")
        ema = loss if ema is None else 0.98 * ema + 0.02 * loss
        if i % args.log_every == 0 or i == stop:
            mm = jax.device_get(m)
            tr, sts = mm["terms"], mm["stats"]
            steady = times[1:] or times                                 # крок 2 інколи ще з дорозкладкою
            mean_ms = float(np.mean(steady)) * 1e3 if steady else first_ms
            el = time.time() - t_start
            eta = (stop - i) * mean_ms / 1e3
            pc = np.asarray(mm.get("per_chip", [loss]))
            hp = (f" uroboros corr {float(sts['uroboros_corr']):.3f} eaten {float(sts['uroboros_eaten']):.3f} "
                  f"sel {float(sts['uroboros_nll_sel']):.3f} skip {float(sts['uroboros_nll_skip']):.3f}"
                  if "uroboros_corr" in sts else (f" anch {float(sts['anchors']):.3f}" if "anchors" in sts else ""))
            if "hn_mean" in sts:                                        # fix17: дрейф стовбура після фінальної norm
                hp += (f" | hn mean {float(sts['hn_mean']):.1f} len {float(sts['hn_len']):.1f}"
                       + (f" guard {float(tr['hn_guard']):.4f}" if "hn_guard" in tr else ""))
            if "rope_share" in sts:                                     # вагонетка: частка уваги, якорі, втрачені слова
                ov = float(sts.get("rope_overflow", 0.0))
                hp += (f" | rope share {float(sts['rope_share']):.4f} anch {float(sts['rope_anchors']):.4f} "
                       f"overflow {ov:.5f}"
                       + (f" klink {float(sts['rope_key_link']):.3f} kanch {float(sts['rope_key_anch']):.3f}"
                          if "rope_key_link" in sts else ""))
                if ov > 0:
                    P(f"УВАГА step {i}: вагонетка губить слова — rope_overflow {ov:.5f} "
                      f"(зібраних слів понад --rope_gather_cap); підніми --rope_gather_cap")
            if "remind_n" in sts:                                       # нагадування в train-батчі (рядків-голок мало)
                rn = float(sts["remind_n"])
                hp += (f" | remind nll {float(sts['remind_nll']) / max(rn, 1e-9):.3f} "
                       f"acc {float(sts['remind_hit']) / max(rn, 1e-9):.3f} rows {rn * cfg.B * W:.0f} "
                       + (f"rope@q {float(sts['remind_rope_share']) / max(rn, 1e-9):.4f} " if "remind_rope_share" in sts else "")
                       + f"p {rem_x.p_at(i - 1):.3f}" + (f" рівень {rem_x.level_at(i - 1)} {rem_x.range_at(i - 1)}"
                                                       if rem_x.ladder else "")
                       + (f" синт.рівень {rem_x.plevel_at(i - 1)}/{len(rem_x.plevels) - 1} "
                          f"{rem_x.plevels[rem_x.plevel_at(i - 1)]}" if rem_x.pack and (i <= args.remind_synth or is_syn)
                          else ""))
            if "iw_ok_frac" in sts:                                    # 15.11 Instinct на словах
                hp += (f" | слова: норма {float(sts['iw_ok_frac']):.3f} розмаз {float(sts['iw_spread_frac']):.3f} "
                       f"впевн.хибно {float(sts['iw_cw_frac']):.3f}")
            if "rb_n" in mm:                                           # 15.11 відкатник
                hp += f" | відкатів {int(np.asarray(mm['rb_n']))}"
            if "remind_main_masked" in sts:                            # fix19: частка слів, вилучених з main
                hp += f" | main-маска голок {float(sts['remind_main_masked']):.4f}"
            if syn_far[1] + syn_near[1] > 0:                           # 12.11: синтетика з останнього журналу — окремо
                hp += (" | синт. за вікном " + (f"{syn_far[0] / syn_far[1]:.3f} ({syn_far[1]:.0f})" if syn_far[1] else "—")
                       + " у вікні " + (f"{syn_near[0] / syn_near[1]:.3f} ({syn_near[1]:.0f})" if syn_near[1] else "—"))
                syn_far[:], syn_near[:] = [0.0, 0.0], [0.0, 0.0]
            if "remind_hit_q1" in sts:                                 # етап 1: точність по питаннях 1..4 рядка
                hp += " | q1..4 " + " ".join(
                    f"{float(sts[f'remind_hit_q{qk}']) / max(float(sts[f'remind_n_q{qk}']), 1e-9):.3f}" for qk in range(1, 5))
            if "remind_strict_keep" in sts:
                hp += (f" | strict keep {float(sts['remind_strict_keep']):.3f} pen {float(sts['remind_strict_pen']):.3f} "
                       f"P(інші) {float(sts['remind_strict_pother']):.3f} rare_drop {float(sts['remind_strict_rare_drop']):.3f}")
            moe_l = ""
            if "moe_active_l" in sts:                                   # MOE_LAYER_STATS=1: живих експертів по шарах
                ac = np.ravel(np.asarray(sts["moe_active_l"]))
                moe_l = f" active min {ac.min():.1f} mean {ac.mean():.1f} з {cfg.E}"
            if "rope_share" in sts:
                hp += f" rtau {float(np.ravel(mm['tau'])[1]):.3f}"
            P(f"step {i}/{stop} loss {loss:.4f} main {float(tr['main']):.4f} light {float(tr['light']):.4f} "
              f"aux {float(tr['aux']):.3f} cost {float(tr['cost']):.3f}"
              + (f" uroborosL {float(tr['uroboros']):.3f}" if "uroboros" in tr else "")
              + f" | tok/s {win_tok / max(win_t, 1e-9):,.0f} (сер {tok_step / (mean_ms / 1e3):,.0f}) {dt * 1e3:.0f} ms "
              f"(сер {mean_ms:.0f}) дані {win_wait / max(win_n, 1) * 1e3:.0f} ms | lr {lr_of(i - 1):.3e} "
              f"tau {float(np.ravel(mm['tau'])[0]):.4f}{hp} | moe fill {float(sts.get('moe_fill', np.nan)):.3f} "
              f"drop {float(sts.get('moe_drop', np.nan)):.4f}{moe_l} chip loss {pc.min():.3f}..{pc.max():.3f} "
              f"| |w| {float(pnorm_fn(state['params'])):.1f} k {_num(mm['k'])} s {_num(mm['s'])} "
              f"stake {_num(mm['stake'])}{_snr_str(mm, spec)}{_gain_str(mm, spec)} | {el / 60:.1f} хв, ETA {eta / 60:.1f} хв "
              f"| рекомп кроку {_recomp(ts)}")
            win_t, win_tok, win_wait, win_n = 0.0, 0, 0.0, 0
        if args.eval_every and i % args.eval_every == 0:
            run_eval(state, i)
        if args.ckpt_every and i % args.ckpt_every == 0 and i < stop:
            save(state, i)
    if status == "OK" and i == stop and i > i0:
        if args.eval_every and i % args.eval_every:                     # на кратному кроці eval уже був у циклі
            run_eval(state, i)
        if not args.no_final_ckpt and (not args.stop_at or args.stop_at >= args.steps):
            save(state, i, final=True)
    saver.wait()
    up.wait()
    saver.cleanup()
    if lossf:
        lossf.close()
    steady = times[1:] or times
    res = dict(status=status, step=i, steps=args.steps, ms_step=float(np.median(steady)) * 1e3 if steady else None,
               tok_s=tok_step / float(np.median(steady)) if steady else None, first_step_s=(first_ms or 0) / 1e3,
               recompiles=_recomp(ts), compiles_all=len(compiles) - n_comp0, minutes=(time.time() - t_start) / 60)
    if args.compare:
        a = {}
        with open(args.compare) as f:
            for line in f:
                r = json.loads(line)
                a[r["step"]] = r["loss"]
        b = {}
        with open(args.loss_log) as f:
            for line in f:
                r = json.loads(line)
                if r["step"] > i0:
                    b[r["step"]] = r["loss"]
        common = sorted(set(a) & set(b))
        diffs = [abs(a[s] - b[s]) for s in common]
        ok = bool(common) and max(diffs) <= args.cmp_tol and len(common) == stop - i0
        bitwise = bool(common) and all(a[s] == b[s] for s in common)
        P(f"RESUME_TEST {'PASS' if ok else 'FAIL'} кроки {i0 + 1}–{stop}: порівняно {len(common)}, "
          f"max|Δ| {max(diffs) if diffs else float('nan'):.3e}, побітово {bitwise}")
        for s in common:
            P(f"  cmp step {s}: A {a[s]!r} B {b[s]!r}")
        res["resume_test"] = "PASS" if ok else "FAIL"
    P("RESULT " + json.dumps(res))
    hl.close()
    return status == "OK" and res.get("resume_test", "PASS") == "PASS"


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
