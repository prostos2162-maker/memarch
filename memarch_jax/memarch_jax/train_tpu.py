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
    a.add_argument("--fish_full_from", type=int, default=-1, help="з цього кроку — фінальна фаза: увесь набір виду")
    a.add_argument("--heavy_full_every", type=int, default=0,
                   help="тасування (ідея Стефана 11.11): кожен K-й крок один важкий шар (по колу 0..heavy-1) працює на ВСІХ словах; 0 — вимкнено")
    a.add_argument("--heavy_all", action="store_true",
                   help="усі важкі шари на всіх словах (фінальна фаза перед SFT / режим готової моделі)")
    a.add_argument("--remind_near", type=float, default=0.0,
                   help="частка голок справжнього тексту У ВІКНІ (відстань 0..вікно, трос відкритий): лікує заважання троса близьким фактам")
    a.add_argument("--remind_dist", type=int, default=0, help="мін. відстань факти→питання (0 = cfg.window)")
    a.add_argument("--remind_tok0", type=int, default=1000, help="перший токен голок: ключі +0..511, значення +512..575, мітка +576")
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
    a.add_argument("--remind_pack_round", type=int, default=1,
                   help="--remind_pack: довжина короткого рядка кратна цьому числу (ядра Pallas троса/уваги на TPU "
                        "потребують кратності блоку, напр. 256); 1 — точно як torch (4·пар + сміття − 1)")
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
                         "loss_log", "compare", "cmp_tol", "cache", "log_every", "eval_every", "eval_batches", "spike"))


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


def restore_sharded(path, template):
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
    assert len(lv) == len(vals) and all(d["path"] == p for d, p in zip(lv, paths)), "структура ckpt ≠ стан моделі"
    leaves, treedef = jax.tree_util.tree_flatten(tree)
    new = []
    for i, (d, x) in enumerate(zip(lv, leaves)):
        a = np.load(os.path.join(path, "arrays", f"{i}.npy"), mmap_mode="r")
        if d["dtype"] == "bfloat16":
            a = a.view(jnp.bfloat16)
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
        ls = subprocess.run(f"gcloud storage ls {self.dst}/", shell=True, capture_output=True, text=True).stdout
        for d in ls.split():
            nm = d.rstrip("/").split("/")[-1]
            if nm.startswith("step_") and nm[5:].isdigit() and int(nm[5:]) < step:
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
        for n in os.listdir(self.root):
            if n.startswith("step_") and n[5:].isdigit() and int(n[5:]) < step:
                shutil.rmtree(os.path.join(self.root, n), ignore_errors=True)

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
                                  fish_c=args.fish_c, fish_kinds=len(fish_kinds))
    if getattr(args, "heavy_all", False):
        cfg = dataclasses.replace(cfg, heavy_all=True)
    elif getattr(args, "heavy_full_every", 0) > 0:
        cfg = dataclasses.replace(cfg, heavy_rotate=True)
    if args.vocab:
        cfg = dataclasses.replace(cfg, V=args.vocab, short_v=args.short_v or cfg.short_v)
    rem_on = args.remind > 0
    if rem_on:
        cfg = dataclasses.replace(cfg, remind=True, remind_w=args.remind_w, remind_q=args.remind_q,
                                  remind_use_w=args.remind_use_w)
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
    use_pallas = plat == "tpu" and not args.no_pallas and cfg.dtype == "bf16"
    icfg = instinct.InstinctCfg(lr_peak=args.lr, use_pallas=use_pallas, sr=cfg.dtype == "bf16",
                                m_bf16=cfg.dtype == "bf16")
    if args.opt_scale is not None:
        icfg = dataclasses.replace(icfg, scale=args.opt_scale)
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
                                                                                remind_mean=bool(args.remind_pack)))
    else:
        ts, state, place = dist.make_step(cfg, consts, mesh, params, spec, icfg, remat=remat)
    if diag.ON:
        jax.block_until_ready(state)
        diag.phase("стан Instinct і крок готові (make_step)")
    # eval: лише fwd (та сама shard_map-втрата; tau/стан не змінюються)
    lf = dist.make_loss_fn(dataclasses.replace(cfg, fish_eval=True) if getattr(cfg, "fish_n", 0) > 0 else cfg,
                           consts, mesh, params, remat)
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
                                 rank=rank, world=world, cache=args.cache, eos=eos)
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
                               near=args.remind_near, near_max=max(0, _w - _L))
        if args.remind_p0 is not None or lad:
            P(f"нагадування: розклад p {args.remind_p0} ({args.remind_warm} кр.) → {args.remind} за {args.remind_decay}; "
              f"штраф троса {args.remind_use_w}; драбина {lad} поріг {args.remind_thr} вікно {args.remind_win} відповідей")
        P(f"нагадування: p={args.remind} пар {args.remind_q} відстань ≥{rem_x.dist} токени {rem_x.k0}..{rem_x.mark} "
          f"вага {args.remind_w}")
        _pf = prep_fn
        prep_fn = lambda x: dict(_pf(x), rem=np.zeros(x.shape, np.int8))    # eval-набори без голок: маска нулі
        mx = cfg.T + 1 - 2 * rem_x.L
        rem_dists = ([int(v) for v in args.remind_eval.split(",")] if args.remind_eval else
                     sorted({min(v, mx) for v in (_w // 2, dmin, 2 * _w, 4 * _w, mx)}))
        assert all(0 <= v <= mx for v in rem_dists), (rem_dists, mx)
        P(f"eval пригадування на відстанях {rem_dists} (вікно {cfg.window}; ціль 95%+ за межами вікна)")
        if rem_x.pack:
            P(f"синтетика короткими рядками (--remind_pack): рівні (пар, сміття) {rem_x.plevels}; eval pack_fN: {args.remind_q} пар, "
              f"сміття {rem_x.pfill}")
    hl = data.HostLoader(corpus, rows, prep_fn=_pf if rem_on else prep_fn, depth=3, xform=rem_x)
    P(f"корпус: {len(weights)} джерел, eos={eos}, rank {rank}/{world}, суміш "
      + " ".join(f"{k}={v:g}" for k, v in weights.items()))

    def lr_of(i):
        return instinct.lr_at(i, args.lr, args.steps, warmup=args.warmup, sched=args.sched)

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
            state, ds, extra = restore_sharded(path, state)
            if rem_x is not None and rem_x.ladder and isinstance(ds, dict) and "_remind" in ds:
                rem_x.load(ds["_remind"])
                P(f"resume нагадувань: рівні {rem_x.changes}")
            hl.load_state_dict(ds)
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
        if rem_x is not None and rem_x.ladder:
            ds = dict(ds, _remind=rem_x.state())
        saver.submit(tree, i, ds, dict(args=vars(args), mix=weights, run_id=rid))
        P(f"ckpt step {i}: цикл стояв {time.perf_counter() - t0:.2f} с (знімок на хост)")

    # --- цикл
    lossf = open(args.loss_log, "a") if args.loss_log else None
    times, ema, n_comp0 = [], None, len(compiles)
    win_t, win_tok, win_wait, win_n = 0.0, 0, 0.0, 0
    first_ms = None
    nan_at = int(os.environ.get("TRAIN_NAN_AT", "0"))
    i = i0
    status = "OK"
    while i < stop:
        tw = time.perf_counter()
        if fish_kinds and args.fish_full_from >= 0 and i == args.fish_full_from and not corpus.full:
            corpus.full = True                                           # фінальна фаза: увесь набір виду
            P(f"step {i}: фішки — фінальна фаза (увесь набір виду, {cfg.fish_pool} фішок)")
        xy, prep = hl.next()
        batch = mk(xy, prep, lr_of(i), hfull_of(i))
        t_wait = time.perf_counter() - tw
        t0 = time.perf_counter()
        if ts_synth is not None and i < args.remind_synth:              # етап 1 (batch i — синтетика, див. Reminders)
            state, m = ts_synth(state, batch)
        else:
            if ts_synth is not None and i == args.remind_synth:
                P(f"step {i}: етап 1 (синтетика) завершено → звичайний текст з нагадуваннями")
            state, m = ts(state, batch)
        if rem_x is not None and (rem_x.ladder or rem_x.pack):          # та сама синхронізація + 2 скаляри
            loss, rh, rn = (float(v) for v in jax.device_get((m["loss"], m["stats"]["remind_hit"],
                                                                m["stats"]["remind_n"])))
            if rem_x.pack and i < args.remind_synth:                    # короткі рядки: відповідей у батчі — з маски
                _na = float(np.asarray(prep["rem"]).sum())
                msg = rem_x.observe(i, rh / max(rn, 1e-9) * _na, _na)
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
            if "rope_share" in sts:                                     # вагонетка: частка уваги, якорі, втрачені слова
                ov = float(sts.get("rope_overflow", 0.0))
                hp += (f" | rope share {float(sts['rope_share']):.4f} anch {float(sts['rope_anchors']):.4f} "
                       f"overflow {ov:.5f}")
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
                          f"{rem_x.plevels[rem_x.plevel_at(i - 1)]}" if rem_x.pack and i <= args.remind_synth else ""))
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
              f"stake {_num(mm['stake'])} | {el / 60:.1f} хв, ETA {eta / 60:.1f} хв "
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
