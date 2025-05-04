#!/usr/bin/env bash
# Grid‑search 18 (=3×3×2) CNN configs for chord classification.
# Results are written to ./logs/{tag}.log  (one log per run).

set -euo pipefail
mkdir -p logs

# ────────── Hyper‑parameter grids ──────────
LR_LIST=(3e-4 5e-4 8e-4)          # learning‑rate
BASE_CH_LIST=(48 64 96)           # width of CNN (affects capacity)
DROPOUT_LIST=(0.3 0.4 0.5)            # regularisation strength

EPOCHS=15
BATCH=1024
DEVICE="cuda:1"

# ────────── Sweep ──────────
for lr in "${LR_LIST[@]}"; do
  for ch in "${BASE_CH_LIST[@]}"; do
    for drop in "${DROPOUT_LIST[@]}"; do

      tag="lr${lr}_ch${ch}_do${drop}"
      echo "===== Running $tag ====="

      python cnn_train.py \
        --lr "$lr" \
        --base-channels "$ch" \
        --dropout "$drop" \
        --epochs "$EPOCHS" \
        --batch-size "$BATCH" \
        --device "$DEVICE" \
        > "logs/${tag}.log" 2>&1
    done
  done
done
