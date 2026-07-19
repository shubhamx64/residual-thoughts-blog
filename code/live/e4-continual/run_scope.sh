#!/bin/bash
# E-M2 interface-localization grid (PREREG v1, plan rev 3).
# 23 runs: {down20,gate20,up20,read20} x seeds 0-4  +  read10 x seeds 0-2.
# Idempotent: a completed log (>=6 eval rows) is skipped, so re-launching resumes.
cd /c/Users/shubh/Downloads/s2path || exit 1
PY=.venv/Scripts/python.exe
CKPT=e4-continual/results/ckpt_A.pt

run_one() {
  local scope=$1 name=$2 seed=$3 maskargs=$4
  local log="e4-continual/results/log_B_weights_scope-${name}_s${seed}.jsonl"
  if [ -f "$log" ] && [ "$(grep -c '"step"' "$log")" -ge 6 ]; then
    echo "skip scope-${name}_s${seed} (complete)"; return
  fi
  echo ">>> scope-${name}_s${seed}"
  $PY e4-continual/src/train_e4.py --phase B --arm weights --model tinyllama-1.1b \
      --seed "$seed" --init "$CKPT" --mask-scope "$scope" \
      --tag-suffix "_scope-${name}_s${seed}" $maskargs
  echo "=== done scope-${name}_s${seed} exit $? ==="
}

for seed in 0 1 2 3 4; do
  for scope in down gate up read; do
    run_one "$scope" "${scope}20" "$seed" ""
  done
done
for seed in 0 1 2; do
  run_one read read10 "$seed" "--mask-file e4-continual/data/mask_weights10.npz"
done
echo "ALL SCOPE RUNS DONE"
