set -euo pipefail

mkdir -p logs

# -------- Sweep Space --------
LRS=(1e-4 2e-4 5e-4 1e-3)
BATCH_SIZES=(32 64 128)
EPOCHS_LIST=(10 20 30)

DEVICE="cuda:0"

# -------- Sweep Loop --------
for lr in "${LRS[@]}"; do
  for bs in "${BATCH_SIZES[@]}"; do
    for ep in "${EPOCHS_LIST[@]}"; do
      run="lr${lr}_bs${bs}_ep${ep}"
      echo -e "\n===== Running ${run} ====="
      python logistic.py \
        --lr "${lr}" \
        --batch-size "${bs}" \
        --epochs "${ep}" \
        --device "${DEVICE}" \
      2>&1 | tee "logs/${run}.log"
    done
  done
done
