#!/usr/bin/env bash
# Hyperparameter sweep for cuML SVM chord classifier (POP909)

set -euo pipefail

mkdir -p logs

CS=(0.1 1.0 10.0)
GAMMAS=("scale" "auto")
SUBSAMPLES=(2000)  # smaller size recommended for SVM

for c in "${CS[@]}"; do
  for g in "${GAMMAS[@]}"; do
    for n in "${SUBSAMPLES[@]}"; do
      run="svm_c${c}_g${g}_n${n}"
      echo -e "\n===== Running ${run} ====="
      python svm.py \
        --C "$c" \
        --gamma "$g" \
        --subsample "$n" \
      2>&1 | tee "logs/${run}.log"
    done
  done
done
