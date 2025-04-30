#!/usr/bin/env bash
# Hyper‑parameter sweep script for the BiLSTM‑Attention chord classifier.
# -----------------------------------------------------------------------------
# ‑ Each run is executed sequentially to avoid exhausting the GPU’s 48 GB VRAM.
# ‑ Adjust the search space below if you need a finer / broader sweep.
# ‑ Logs are stored under ./logs/ with the run name as file name.
# -----------------------------------------------------------------------------

set -euo pipefail

# Create a directory for logs if it does not exist
mkdir -p logs

# ---- Search space definition -------------------------------------------------
LRS=(1e-4 2e-4 5e-4)
WEIGHT_DECAYS=(0 1e-2 3e-2)
BATCH_SIZES=(512 1024 2048)          # 2048 may exceed 15 GB runtime usage → excluded
LABEL_SMOOTHS=(0.0 0.1 0.15)
DROPOUTS=(0.3 0.4 0.5)
HIDDENS=(256 512 768)
HEADS=(2 4 8)
OPTIMS=(adam adamw)
SCHEDS=(cosine step)

EPOCHS=25
DEVICE="cuda:0"                 # change if you prefer another GPU index

# ---- Sweep loop --------------------------------------------------------------
for lr in "${LRS[@]}"; do
  for wd in "${WEIGHT_DECAYS[@]}"; do
    for bs in "${BATCH_SIZES[@]}"; do
      for ls in "${LABEL_SMOOTHS[@]}"; do
        for dr in "${DROPOUTS[@]}"; do
          for hd in "${HIDDENS[@]}"; do
            for head in "${HEADS[@]}"; do
              for opt in "${OPTIMS[@]}"; do
                for sch in "${SCHEDS[@]}"; do
                  run="lr${lr}_wd${wd}_bs${bs}_ls${ls}_do${dr}_h${hd}_heads${head}_${opt}_${sch}"
                  echo "\n===== Running $run ====="
                  python rnn_train.py \
                    --batch-size "${bs}" \
                    --lr "${lr}" \
                    --weight-decay "${wd}" \
                    --label-smoothing "${ls}" \
                    --dropout "${dr}" \
                    --hidden "${hd}" \
                    --heads "${head}" \
                    --optimizer "${opt}" \
                    --scheduler "${sch}" \
                    --epochs "${EPOCHS}" \
                    --device "${DEVICE}" \
                    2>&1 | tee "logs/${run}.log"
                done
              done
            done
          done
        done
      done
    done
  done
done

# After the loop, you can parse the logs directory to collect metrics.
