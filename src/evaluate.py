"""Evaluate a trained checkpoint on the hold-out test set.

Supports both a segmentation-only checkpoint (from ``train_seg.py``) and a full
deblur+segmentation pipeline checkpoint (from ``train_restoration_aware.py``).
Reports mDS / mIoU / mPrecision / mRecall and per-class Dice, matching Tables
2-4 of the paper.

Example
-------
    python -m src.evaluate --config configs/default.yaml \
        --checkpoint runs/resnet50_seed0/best.pt --kind seg
    python -m src.evaluate --config configs/default.yaml \
        --checkpoint runs/pipeline_resnet50_seed0/pipeline_best.pt --kind pipeline
"""
from __future__ import annotations

import argparse
import json

import torch

from .data.dataset import CLASS_NAMES
from .metrics import SegMetricAccumulator
from .models.nafnet import build_deblur_model
from .models.unet_resnet import build_segmentation_model
from .utils import build_dataloaders, get_device, load_config, set_seed


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--kind", choices=["seg", "pipeline"], default="seg")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--json-out", default=None)
    return p.parse_args()


@torch.no_grad()
def main():
    args = parse_args()
    cfg = load_config(args.config)
    set_seed(args.seed)
    device = get_device(cfg)
    state = torch.load(args.checkpoint, map_location=device)
    if "config" in state:
        cfg = state["config"]

    _, _, test_loader = build_dataloaders(cfg, seed=args.seed)
    num_classes = cfg["model"].get("num_classes", 3)
    acc = SegMetricAccumulator(num_classes)

    seg = build_segmentation_model(cfg).to(device).eval()
    deblur = None
    if args.kind == "pipeline":
        seg.load_state_dict(state["seg"])
        deblur = build_deblur_model(cfg).to(device).eval()
        deblur.load_state_dict(state["deblur"])
    else:
        seg.load_state_dict(state["model"])

    for blurry, sharp, mask in test_loader:
        mask = mask.to(device)
        if args.kind == "pipeline":
            x = deblur(blurry.to(device))
        else:
            x = sharp.to(device)
        acc.update(seg(x), mask)

    res = acc.compute()
    print(f"mDS={res['mDice']:.4f} mIoU={res['mIoU']:.4f} "
          f"mPrecision={res['mPrecision']:.4f} mRecall={res['mRecall']:.4f}")
    for name, d in zip(CLASS_NAMES, res["dice_per_class"]):
        print(f"  Dice[{name:10s}] = {float(d):.4f}")

    if args.json_out:
        out = {k: (v.tolist() if hasattr(v, "tolist") else v) for k, v in res.items()}
        with open(args.json_out, "w") as f:
            json.dump(out, f, indent=2)


if __name__ == "__main__":
    main()
