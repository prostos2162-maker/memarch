"""Режим тотальної діагностики тренування (DIAG=1; Стефан 12.11 після зависання на v6e-8 без жодного сліду).
- чорна скринька: якщо DIAG_STALE с (180) не було нового рядка логу — стек УСІХ потоків (faulthandler) + стан хоста й
  пам'яті чипів; повторює кожні DIAG_REPEAT с (120), поки висить;
- фази старту: час від старту і від попередньої фази, RSS/CPU хоста, пам'ять кожного чипа (зайнято / пік);
- кожна компіляція XLA з часом (jax_log_compiles);
- кожні DIAG_EVERY кроків (50) — пам'ять чипів; профіль кроків DIAG_PROF..+3 (jax.profiler) у DIAG_DIR.
Без DIAG=1 нічого не робить (touch/phase — порожні виклики)."""
import faulthandler
import logging
import os
import sys
import threading
import time

ON = os.environ.get("DIAG", "0") == "1"
STALE = float(os.environ.get("DIAG_STALE", "180"))
REPEAT = float(os.environ.get("DIAG_REPEAT", "120"))
EVERY = int(os.environ.get("DIAG_EVERY", "50"))
PROF = int(os.environ.get("DIAG_PROF", "20"))
DIR = os.environ.get("DIAG_DIR", "/content/ms/tr_diag")
_t0 = time.time()
_last = [time.time()]
_phase_t = [time.time()]
_prof_on = [False]


def _p(*a):
    print("[diag]", *a, flush=True)


def touch():
    _last[0] = time.time()


def host():
    try:
        rss = int(open("/proc/self/status").read().split("VmRSS:")[1].split()[0]) / 2**20
        la = open("/proc/loadavg").read().split()[:3]
        mem = open("/proc/meminfo").read().split("MemAvailable:")[1].split()[0]
        return f"RSS {rss:.1f} GiB, load {' '.join(la)}, вільно RAM {int(mem) / 2**20:.0f} GiB"
    except Exception as e:                       # noqa: BLE001 — не Linux
        return f"хост: {e!r}"


def devmem():
    try:
        import jax
        out = []
        for d in jax.local_devices():
            s = d.memory_stats() or {}
            out.append(f"{d.id}:{s.get('bytes_in_use', 0) / 2**30:.1f}/{s.get('peak_bytes_in_use', 0) / 2**30:.1f}"
                       f"/{s.get('bytes_limit', 0) / 2**30:.0f}")
        return "HBM GiB (зайнято/пік/межа) " + " ".join(out)
    except Exception as e:                       # noqa: BLE001
        return f"HBM: {e!r}"


def _watch():
    while True:
        time.sleep(15)
        idle = time.time() - _last[0]
        if idle > STALE:
            _p(f"ЗАВИСАННЯ? {idle:.0f} с без нових рядків логу (від старту {time.time() - _t0:.0f} с). {host()}")
            _p(devmem())
            _p("стек усіх потоків:")
            faulthandler.dump_traceback(file=sys.stdout, all_threads=True)
            sys.stdout.flush()
            _last[0] = time.time() - STALE + REPEAT


def start():
    if not ON:
        return
    faulthandler.enable(file=sys.stderr, all_threads=True)
    logging.basicConfig(stream=sys.stdout, level=logging.WARNING, format="[diag-log] %(message)s")
    try:
        import jax
        jax.config.update("jax_log_compiles", True)
        logging.getLogger("jax._src.dispatch").setLevel(logging.WARNING)
        logging.getLogger("jax._src.interpreters.pxla").setLevel(logging.WARNING)
    except Exception as e:                       # noqa: BLE001
        _p(f"jax_log_compiles: {e!r}")
    threading.Thread(target=_watch, daemon=True).start()
    _p(f"увімкнено: зависання > {STALE:.0f} с → стек; пам'ять кожні {EVERY} кроків; профіль кроків {PROF}..{PROF + 3} → {DIR}")
    phase("старт")


def phase(name):
    if not ON:
        return
    now = time.time()
    _p(f"фаза «{name}»: +{now - _phase_t[0]:.1f} с (від старту {now - _t0:.1f} с) | {host()} | {devmem()}")
    _phase_t[0] = now
    touch()


def step(i):
    """Після кроку i (номер наступного): пам'ять раз на EVERY кроків, профіль кроків PROF..PROF+3."""
    if not ON:
        return
    if i % EVERY == 0:
        _p(f"крок {i}: {devmem()} | {host()}")
    try:
        import jax
        if i == PROF and not _prof_on[0]:
            os.makedirs(DIR, exist_ok=True)
            jax.profiler.start_trace(DIR)
            _prof_on[0] = True
            _p(f"профіль: старт на кроці {i} → {DIR}")
        elif i == PROF + 3 and _prof_on[0]:
            jax.profiler.stop_trace()
            _prof_on[0] = False
            _p(f"профіль: стоп на кроці {i}")
    except Exception as e:                       # noqa: BLE001
        _p(f"профіль: {e!r}")
