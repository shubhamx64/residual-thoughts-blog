#!/bin/bash
# Multi-seed E4 phase-B sweep (PREREG_ROBUSTNESS.md Part A).
# From the fixed per-family after-A checkpoint, run seeds 1-4 x 6 arms x 2 families.
# Seed 0 = the existing runs. Idempotent: a completed log (>=6 eval rows) is skipped,
# so re-launching after any kill resumes.
cd /c/Users/shubh/Downloads/s2path || exit 1
PY=.venv/Scripts/python.exe
ARMS="baseline random weights footprint join fisher"

run_family() {
  local model=$1 ckpt=$2 maskdir=$3 tagbase=$4
  for seed in 1 2 3 4; do
    for arm in $ARMS; do
      local suf="${tagbase}_s${seed}"
      local log="e4-continual/results/log_B_${arm}${suf}.jsonl"
      if [ -f "$log" ] && [ "$(grep -c '"step"' "$log")" -ge 6 ]; then
        echo "skip B_${arm}${suf} (complete)"; continue
      fi
      local maskargs=""
      if [ "$arm" = "random" ]; then
        maskargs="--mask-file ${maskdir}/mask_random_s${seed}.npz"
      elif [ "$arm" != "baseline" ]; then
        maskargs="--mask-file ${maskdir}/mask_${arm}.npz"
      fi
      echo ">>> B_${arm}${suf}"
      $PY e4-continual/src/train_e4.py --phase B --arm "$arm" --model "$model" --seed "$seed" \
          --init "$ckpt" --tag-suffix "$suf" $maskargs
      echo "=== done B_${arm}${suf} exit $? ==="
    done
  done
}

run_family tinyllama-1.1b e4-continual/results/ckpt_A.pt e4-continual/data ""
run_family qwen2.5-1.5b e4-continual/results/ckpt_A_qwen.pt e4-continual/data/qwen2.5-1.5b "_qwen"
echo "MULTISEED SWEEP DONE"
