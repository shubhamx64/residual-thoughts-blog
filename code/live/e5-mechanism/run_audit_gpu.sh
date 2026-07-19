#!/bin/bash
# Audit follow-ups (a) + (b): seed-matched updnorm splices, 2 extra selector seeds
cd /c/Users/shubh/Downloads/s2path || exit 1
PY=.venv/Scripts/python.exe
echo "== (a) seed-matched updnorm + random splices, seeds 1-4 =="
$PY e5-mechanism/src/rollback.py --model tinyllama-1.1b --grid --seeds 1,2,3,4 \
    --buckets updnorm,random --out-suffix _updnorm_seedmatched
echo "== (b) selector seeds 5,6 =="
for seed in 5 6; do
  for arm in fisher curvK2; do
    log="e4-continual/results/log_B_${arm}_s${seed}.jsonl"
    if [ -f "$log" ] && [ "$(grep -c '"step"' "$log")" -ge 6 ]; then
      echo "skip ${arm}_s${seed}"; continue
    fi
    $PY e4-continual/src/train_e4.py --phase B --arm "$arm" --model tinyllama-1.1b \
        --seed "$seed" --init e4-continual/results/ckpt_A.pt --tag-suffix "_s${seed}" \
        --eval-split-file e5-mechanism/results/eval_math_outcome.jsonl
  done
done
echo "AUDIT GPU BATCH DONE"
