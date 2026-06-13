#!/usr/bin/env bash
# Reproduce Table 2 / Table 3 (encoder comparison) over five seeds (0-4).
set -euo pipefail
cd "$(dirname "$0")/.."

for enc in resnet18 resnet34 resnet50 resnet101; do
  for seed in 0 1 2 3 4; do
    python -m src.train_seg \
      --config configs/default.yaml \
      --encoder "$enc" --seed "$seed" \
      --out "runs/${enc}_seed${seed}"
  done
done
