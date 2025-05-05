#!/usr/bin/env bash
# Hyperparameter sweep for PyTorch MLP chord classifier (POP909)

set -euo pipefail

mkdir -p logs

#!/bin/bash
set -euo pipefail
mkdir -p logs

# 超参数网格
LRS=(1e-4 2e-4 5e-4 1e-3)
BATCH_SIZES=(32 64 128)
EPOCHS_LIST=(10 20 30)
HIDDENS=("256,128" "512,256" "512,256,128")
DEVICE="cuda"

# 网格搜索循环
for lr in "${LRS[@]}"; do
  for bs in "${BATCH_SIZES[@]}"; do
    for ep in "${EPOCHS_LIST[@]}"; do
      for hidden in "${HIDDENS[@]}"; do
        run="lr${lr}_bs${bs}_ep${ep}_h${hidden//,/x}"
        echo -e "\n===== Running ${run} ====="
        python mlp.py \
          --lr "$lr" \
          --batch-size "$bs" \
          --epochs "$ep" \
          --hidden "$hidden" \
          --device "$DEVICE" \
        2>&1 | tee "logs/${run}.log"
      done
    done
  done
done

