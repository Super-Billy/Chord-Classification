#!/usr/bin/env bash
# Hyper-parameter sweep script for the BiLSTM-Attention chord classifier.
# -----------------------------------------------------------------------------
# - Only sweeps learning rate (LR), batch size (BS) and hidden dimension.
# - Runs sequentially to avoid exhausting GPU VRAM.
# - Logs for each run are written to ./logs/ under a descriptive filename.
# -----------------------------------------------------------------------------

set -euo pipefail

# ensure logs directory exists
mkdir -p logs

# ---- Search space definition -----------------------------------------------
LRS=(1e-4 2e-4 4e-4 8e-4)              # learning rates to try
BATCH_SIZES=(256 512 1024)       # batch sizes to try
HIDDENS=(256 512 768)             # hidden layer sizes to try

EPOCHS=15                         # fixed number of epochs
DEVICE="cuda:0"                   # GPU device

# ---- Sweep loop ------------------------------------------------------------
for lr in "${LRS[@]}"; do
  for bs in "${BATCH_SIZES[@]}"; do
    for hd in "${HIDDENS[@]}"; do
      run="lr${lr}_bs${bs}_h${hd}"
      echo -e "\n===== Running ${run} ====="
      
      python lstm_train.py \
        --batch-size "${bs}" \
        --lr         "${lr}" \
        --hidden     "${hd}" \
        --epochs     "${EPOCHS}" \
        --device     "${DEVICE}" \
      2>&1 | tee "logs/${run}.log"
    done
  done
done

# After completion, parse ./logs/ to collect validation metrics.
