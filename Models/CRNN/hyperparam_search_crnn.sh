#!/usr/bin/env bash
# Simplified hyperparameter sweep: only lr, batch-size, and hidden
set -euo pipefail
mkdir -p logs

# Hyperparameter grids (example values—adjust as needed)
LRS=(1e-4 2e-4 4e-4 8e-4)
BATCHES=(256 512 1024)
HIDDENS=(256 512 768)

# LRS=(1e-4)
# BATCHES=(1024)
# HIDDENS=(1024)

EPOCHS=15
DEVICE="cuda:0"

for lr in "${LRS[@]}"; do
  for bs in "${BATCHES[@]}"; do
    for hid in "${HIDDENS[@]}"; do
      run="lr${lr}_bs${bs}_hid${hid}"
      echo "==== $run ===="
      python crnn_train.py \
        --lr ${lr} \
        --batch-size ${bs} \
        --hidden ${hid} \
        --epochs ${EPOCHS} \
        --device ${DEVICE} \
        2>&1 | tee logs/${run}.log
    done
  done
done
