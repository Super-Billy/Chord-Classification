#!/usr/bin/env bash
# Hyper‑parameter sweep for CRNN chord classifier.
# Each run is sequential to respect a 48 GB GPU; typical peak ~12 GB @ batch=512.

set -euo pipefail
mkdir -p crnn_logs

LRS=(1e-4 2e-4 5e-5)
BATCHES=(256 512 1024)
BASE_CHANS=(32 64 96)
KERNELS=(3 5)
HIDDENS=(384 512 768)
DROPS=(0.2 0.3 0.4)
WEIGHT_DECAYS=(0 5e-3 1e-2)

EPOCHS=25
DEVICE="cuda:0"

for lr in "${LRS[@]}"; do
  for bs in "${BATCHES[@]}"; do
    for bc in "${BASE_CHANS[@]}"; do
      for k in "${KERNELS[@]}"; do
        for hid in "${HIDDENS[@]}"; do
          for dr in "${DROPS[@]}"; do
            for wd in "${WEIGHT_DECAYS[@]}"; do
              run="lr${lr}_bs${bs}_bc${bc}_k${k}_h${hid}_dr${dr}_wd${wd}"
              echo "==== $run ===="
              python crnn_train.py \
                --batch-size ${bs} \
                --lr ${lr} \
                --base-channels ${bc} \
                --kernel-size ${k} \
                --hidden ${hid} \
                --dropout ${dr} \
                --weight-decay ${wd} \
                --epochs ${EPOCHS} \
                --device ${DEVICE} \
                2>&1 | tee crnn_logs/${run}.log
            done
          done
        done
      done
    done
  done
done
