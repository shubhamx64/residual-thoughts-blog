#!/bin/bash
# Lane 3/4 (cloud): E4-causal at scale. Usage: run_cloud_e4scale.sh <model_key> <tag>
#   e.g. run_cloud_e4scale.sh qwen2.5-3b 3b   /   qwen2.5-7b 7b
# Builds masks -> phase A -> fisher mask -> phase B x6 arms. Idempotent.
cd /workspace/s2 || exit 1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HOME=/root/hf
export E4_CKPT_DIR=/root/ckpts   # big phase-A ckpt on reliable local disk, not the network volume
PY=python
MODEL="$1"; TAG="$2"
MD="e4-continual/data/${MODEL}"
CKPT="/root/ckpts/ckpt_A_${TAG}.pt"
ARMS="baseline random weights footprint join fisher"
mkdir -p e4-continual/results /root/ckpts

if [ ! -f "${MD}/mask_join.npz" ]; then
  echo ">>> E4SCALE ${TAG} build masks"
  $PY escale/src/cloud_e4prep.py --model "${MODEL}" || exit 1
fi
if [ ! -f "$CKPT" ]; then
  echo ">>> E4SCALE ${TAG} phase A"
  $PY e4-continual/src/train_e4.py --phase A --model "${MODEL}" --tag-suffix "_${TAG}" || exit 1
fi
if [ ! -f "${MD}/mask_fisher.npz" ]; then
  echo ">>> E4SCALE ${TAG} fisher mask"
  $PY escale/src/cloud_e4prep.py --model "${MODEL}" --fisher --ckpt "$CKPT" || exit 1
fi
for arm in $ARMS; do
  log="e4-continual/results/log_B_${arm}_${TAG}.jsonl"
  if [ -f "$log" ] && [ "$(grep -c '"step"' "$log")" -ge 6 ]; then echo "skip ${arm}"; continue; fi
  mask=""
  [ "$arm" != "baseline" ] && mask="--mask-file ${MD}/mask_${arm}.npz"
  echo ">>> E4SCALE ${TAG} arm ${arm}"
  $PY e4-continual/src/train_e4.py --phase B --arm "$arm" --model "${MODEL}" \
      --init "$CKPT" --tag-suffix "_${TAG}" $mask
done
echo "E4SCALE ${TAG} DONE"
