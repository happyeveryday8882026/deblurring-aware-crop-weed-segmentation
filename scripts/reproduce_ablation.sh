#!/usr/bin/env bash
# Reproduce Table 5 (ablation study) with ResNet-50, over five seeds (0-4).
set -euo pipefail
cd "$(dirname "$0")/.."

for seed in 0 1 2 3 4; do
  # Full model
  python -m src.train_seg --config configs/default.yaml --encoder resnet50 \
    --seed "$seed" --out "runs/ablation/full_seed${seed}"
  # w/o ImageNet pretraining
  python -m src.train_seg --config configs/default.yaml --encoder resnet50 \
    --no-pretrain --seed "$seed" --out "runs/ablation/no_pretrain_seed${seed}"
  # w/o skip connections
  python -m src.train_seg --config configs/default.yaml --encoder resnet50 \
    --no-skip --seed "$seed" --out "runs/ablation/no_skip_seed${seed}"
  # w/o dropout
  python -m src.train_seg --config configs/default.yaml --encoder resnet50 \
    --dropout 0.0 --seed "$seed" --out "runs/ablation/no_dropout_seed${seed}"
  # single-conv decoder
  python -m src.train_seg --config configs/default.yaml --encoder resnet50 \
    --decoder-convs 1 --seed "$seed" --out "runs/ablation/single_conv_seed${seed}"
  # 3 decoder blocks
  python -m src.train_seg --config configs/default.yaml --encoder resnet50 \
    --decoder-blocks 3 --seed "$seed" --out "runs/ablation/three_blocks_seed${seed}"
done

# The "w/o DeBlur Module" row is the segmentation-only pipeline evaluated on the
# blurred subset; compare runs/ablation/full_* against the full deblur+seg
# pipeline from scripts/reproduce_pipeline.sh.
