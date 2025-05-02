#!/usr/bin/env bash
# Hyper-parameter sweep: only lr, batch-size, hidden, heads

set -euo pipefail

mkdir -p logs

# ---- Search space -----------------------------------------------------------
LRS=(1e-4 2e-4 4e-4 8e-4)
BATCH_SIZES=(256 512 1024)
HIDDENS=(256 512 768)


EPOCHS=15
DEVICE="cuda:1"  

# ---- Sweep loop -------------------------------------------------------------
for lr in "${LRS[@]}"; do
  for bs in "${BATCH_SIZES[@]}"; do
    for hd in "${HIDDENS[@]}"; do
      # for head in "${HEADS[@]}"; do
        run="lr${lr}_bs${bs}_h${hd}"
        echo -e "\n===== Running ${run} ====="
        python rnn_train.py \
          --batch-size "${bs}" \
          --lr "${lr}" \
          --hidden "${hd}" \
          --epochs  "${EPOCHS}" \
          --device  "${DEVICE}" \
        2>&1 | tee "logs/${run}.log"
      # done
    done
  done
done


