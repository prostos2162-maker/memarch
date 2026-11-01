#!/bin/bash
# Гонка на ПК (наша збірка Fv2moeK8h6d512, інстинкт + гомеостаз). Аргументи: назва:BLK:WARMUP:LR; env SCALE, STEPS, BATCH, MICRO, M
cd "$(dirname "$0")/.."
export PYTHONIOENCODING=utf-8 PYTHONUNBUFFERED=1 TRITON_DEFAULT_FP_FUSION=1 FUSED=ours BF16_EXPERTS=1 MOE_ACT=relu2 LR_SCHED=wsd
export TORCHINDUCTOR_CACHE_DIR=./.cache/inductor TRITON_CACHE_DIR=./.cache/triton TMP=./_tmp TEMP=./_tmp
M=${M:-Fv2moeK8h6d512}
for v in "$@"; do
  [ -f _tmp/race_stop ] && break
  IFS=: read name blk warm lr <<< "$v"
  echo "=== $name BLK=$blk WARMUP=${warm:-200} LR=${lr:-6e-4} NORM=${NORM:-hs}" >> _tmp/race_pc.log
  OPT_KW="{\"norm\": \"${NORM:-hs}\", \"scale\": ${SCALE:-0.9}, \"m_bf16\": true, \"rel_every\": 8}" BLK="$blk" WARMUP=${warm:-200} \
    python fastloop.py train --models $M --stage ABX --steps ${STEPS:-300} --batch ${BATCH:-32} --micro ${MICRO:-16} \
    --eval_every 100 --lr ${lr:-6e-4} 2>&1 | grep --line-buffered -E "крок|Error|error|Traceback" >> _tmp/race_pc.log
done
echo DONE >> _tmp/race_pc.log
