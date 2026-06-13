#!/usr/bin/env bash
# Reproduce the full restoration-aware deblur+segmentation pipeline
# (Table 4 and the "Proposed + DeBlur" row of Table 6), over five seeds (0-4).
set -euo pipefail
cd "$(dirname "$0")/.."

for seed in 0 1 2 3 4; do
  # Stage 1: pre-train the NAFNet deblurring front-end.
  python -m src.train_deblur --config configs/default.yaml \
    --seed "$seed" --out "runs/nafnet_seed${seed}"

  # Stage 2: restoration-aware fine-tuning of deblur + UNet-ResNet50.
  python -m src.train_restoration_aware --config configs/default.yaml \
    --nafnet "runs/nafnet_seed${seed}/nafnet_best.pt" \
    --encoder resnet50 --seed "$seed" \
    --out "runs/pipeline_resnet50_seed${seed}"
done

# Complexity analysis (Table 8).
python -m src.complexity --config configs/default.yaml --encoder resnet50
