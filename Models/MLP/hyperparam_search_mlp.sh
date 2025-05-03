#!/usr/bin/env bash
# Hyperparameter sweep for PyTorch MLP chord classifier (POP909)

set -euo pipefail

mkdir -p logs

LRS=(1e-4 2e-4 5e-4)
BATCH_SIZES=(64 128)
EPOCHS_LIST=(20 30)
HIDDENS=("256,128" "512,256")

DEVICE="cuda:0"

for lr in "${LRS[@]}"; do
  for bs in "${BATCH_SIZES[@]}"; do
    for ep in "${EPOCHS_LIST[@]}"; do
      for h in "${HIDDENS[@]}"; do
        run="mlp_lr${lr}_bs${bs}_ep${ep}_h${h//,/x}"
        echo -e "\n===== Running ${run} ====="
        python mlp.py \
          --lr "$lr" \
          --batch-size "$bs" \
          --epochs "$ep" \
          --hidden "$h" \
          --device "$DEVICE" \
        2>&1 | tee "logs/${run}.log"
      done
    done
  done
done
