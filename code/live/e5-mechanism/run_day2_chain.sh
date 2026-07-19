#!/bin/bash
# Day-2 GPU chain (each stage idempotent/resumable):
#   1. remaining K2 sketches (unit_s0, raw_s1)
#   2. curvK2 selector E4 arm, data seeds 1-4, outcome-half logging
#   3. comparator final-ckpt re-evals on the outcome half
#   4. Qwen protocol-repair baseline B-run with --save-ckpt (E-M3 addendum)
cd /c/Users/shubh/Downloads/s2path || exit 1
PY=.venv/Scripts/python.exe

echo "== stage 1: sketches =="
$PY e5-mechanism/src/curvature_run.py --model tinyllama-1.1b --sketch || exit 1

echo "== stage 2: curvK2 selector runs =="
for seed in 1 2 3 4; do
  log="e4-continual/results/log_B_curvK2_s${seed}.jsonl"
  if [ -f "$log" ] && [ "$(grep -c '"step"' "$log")" -ge 6 ]; then
    echo "skip curvK2_s${seed} (complete)"; continue
  fi
  $PY e4-continual/src/train_e4.py --phase B --arm curvK2 --model tinyllama-1.1b \
      --seed "$seed" --init e4-continual/results/ckpt_A.pt \
      --tag-suffix "_s${seed}" \
      --eval-split-file e5-mechanism/results/eval_math_outcome.jsonl
done

echo "== stage 3: comparator outcome-half re-evals =="
[ -f e5-mechanism/results/outcome_half_evals.json ] || \
  $PY e5-mechanism/src/eval_outcome_half.py

echo "== stage 4: Qwen repair baseline B-run (E-M3 addendum) =="
if [ ! -f e4-continual/results/ckpt_B_baseline_qwen_repair_rb.pt ]; then
  $PY e4-continual/src/train_e4.py --phase B --arm baseline --model qwen2.5-1.5b \
      --seed 0 --init e4-continual/results/ckpt_A_qwen_repair.pt \
      --steps 200 --lr 2e-6 --train-file repair_train_code.jsonl \
      --tag-suffix _repair_rb --save-ckpt --ckpt-name ckpt_B_baseline_qwen_repair_rb.pt
fi
echo "DAY2 CHAIN DONE"
