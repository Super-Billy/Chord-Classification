#!/usr/bin/env bash
# Grid-search 36 (=3×2×3×3) Transformer configs (adding batch size sweep).
# Results are written to ./logs/{tag}.log  (one log per run).

set -e
mkdir -p logs

# Hyper-parameter grids
LR_LIST=(1e-4 2e-4 3e-4)
D_MODEL_LIST=(256 512 768)       # must be divisible by --heads (8)
LAYERS_LIST=(2 4)
BATCH_SIZES=(64 128 256 512)  # batch sizes to try

for lr in "${LR_LIST[@]}"; do
  for dmodel in "${D_MODEL_LIST[@]}"; do
    for layers in "${LAYERS_LIST[@]}"; do
      for bs in "${BATCH_SIZES[@]}"; do

        tag="lr${lr}_d${dmodel}_L${layers}_bs${bs}"
        echo "===== Running ${tag} ====="

        python transformer_train.py \
          --lr         "$lr" \
          --hidden     "$dmodel" \
          --ff-dim     $((dmodel * 4)) \
          --layers     "$layers" \
          --heads      8 \
          --batch-size "$bs" \
          --epochs     30 \
          --optimizer  adamw \
          --scheduler  cosine \
          --device     cuda:2 \
        | tee "logs/${tag}.log"

      done
    done
  done
done
