# Тест масштабу 1B на G4: трансформер Td2048L16 (1.28B) проти нашої Fv2moeK8h8d2048 (2.25B / ~1B на якорі),
# той самий цикл, дані, оптимізатор (інстинкт + гомеостаз); ~30 хв чистого тренування кожна.
import re, subprocess, os
subprocess.run("cd /content && rm -f r1b_done.txt r1b_plan.txt r1b_T.log r1b_ours.log r1b_speed_*.log", shell=True)
E = ("cd /content && export PYTHONUNBUFFERED=1 PYTHONIOENCODING=utf-8 FUSED=ours BF16_EXPERTS=1 MOE_ACT=relu2 LR_SCHED=wsd "
     "OPT_KW='{\"norm\": \"hs\", \"scale\": 0.12, \"m_bf16\": true, \"rel_every\": 8}' WSD_FRAC=0.5 WARMUP=50 WARM_COMPILE=1 && ")
OURS = ("PHRASE_INIT=_tmp/phr/bigram_d2048.pt PHRASE3=_tmp/phr/trigram_d2048_k262144.pt SURP_PHRASE=1 NO_MEM=1 "
        "INDUCT=lock LOCK_TRUST=_tmp/phr/lock_trust.pt INDUCT_IN=1 INDUCT_MIX=1 WAVE_WARM=1000000 ")
def sh(c, log):
    with open(log, "a") as f:
        subprocess.run(c, shell=True, stdout=f, stderr=subprocess.STDOUT)
if not os.path.exists("/content/_tmp/phr/trigram_d2048_k262144.pt"):
    sh("cd /content && python _tmp/phr/bigram_init.py 2048 100000000 && python _tmp/phr/lse2048.py && python _tmp/phr/tri2048.py 262144", "/content/r1b_prep.log")
for tag, model, extra in [("T", "Td2048L16", ""), ("ours", "Fv2moeK8h8d2048", OURS)]:
    sh(E + extra + f"python -u fastloop.py speed --models {model} --stages ABX --batch 32 --micro 32 --warmup 10 --n 10", f"/content/r1b_speed_{tag}.log")
    txt = open(f"/content/r1b_speed_{tag}.log").read()
    m = re.findall(r"SPEED\s+\S+\s+\S+\s+([\d.]+) ms", txt)
    ms = float(m[-1]) if m else 1000.0
    steps = max(500, int(1500 / (ms / 1000)) // 50 * 50)
    with open("/content/r1b_plan.txt", "a") as f:
        f.write(f"{tag} {model} {ms:.1f} ms/крок -> {steps} кроків\n")
    sh(E + extra + f"python -u fastloop.py train --models {model} --stage ABX --steps {steps} --batch 32 --micro 32 "
       f"--eval_every {max(50, steps // 12 // 50 * 50)} --lr 6e-4", f"/content/r1b_{tag}.log")
open("/content/r1b_done.txt", "w").write("DONE")
