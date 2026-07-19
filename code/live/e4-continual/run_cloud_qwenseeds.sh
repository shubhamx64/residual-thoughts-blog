#!/bin/bash
# Lane 1 (cloud): Qwen2.5-1.5B multi-seed E4. Self-bootstraps on the pod:
#   phase A (math) -> recompute Fisher mask from the pod after-A ckpt -> phase B
#   seeds 0-4 x 6 arms (5 mutually-consistent seeds from one checkpoint).
# Idempotent; pin to a GPU via CUDA_VISIBLE_DEVICES in the dispatcher.
cd /workspace/s2 || exit 1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=python
ARMS="baseline random weights footprint join fisher"
CKPT=e4-continual/results/ckpt_A_qwen.pt
MD=e4-continual/data/qwen2.5-1.5b
mkdir -p e4-continual/results

if [ ! -f "$CKPT" ]; then
  echo ">>> LANE1 phase A (qwen, seed 0)"
  $PY e4-continual/src/train_e4.py --phase A --model qwen2.5-1.5b --tag-suffix _qwen || exit 1
fi
if [ ! -f "${MD}/.fisher_pod" ]; then
  echo ">>> LANE1 fisher mask from pod ckpt"
  $PY e4-continual/src/prep_qwen_masks.py --fisher && touch "${MD}/.fisher_pod"
fi

for seed in ${SEEDS:-0 1 2 3 4}; do
  for arm in $ARMS; do
    log="e4-continual/results/log_B_${arm}_qwen_s${seed}.jsonl"
    if [ -f "$log" ] && [ "$(grep -c '"step"' "$log")" -ge 6 ]; then echo "skip ${arm} s${seed}"; continue; fi
    mask=""
    if [ "$arm" = "random" ]; then
      if [ "$seed" = "0" ]; then mask="--mask-file ${MD}/mask_random.npz"
      else mask="--mask-file ${MD}/mask_random_s${seed}.npz"; fi
    elif [ "$arm" != "baseline" ]; then mask="--mask-file ${MD}/mask_${arm}.npz"; fi
    echo ">>> LANE1 qwen seed ${seed} arm ${arm}"
    $PY e4-continual/src/train_e4.py --phase B --arm "$arm" --model qwen2.5-1.5b --seed "$seed" \
        --init "$CKPT" --tag-suffix "_qwen_s${seed}" $mask
  done
done
echo "LANE1 QWEN_SEEDS_DONE"
