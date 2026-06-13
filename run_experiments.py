"""Driver: run the real three-class UNet-ResNet experiments and dump results.

Trains each configuration on the actual DeBlurWeedSeg three-class data (MPS/CPU),
evaluates on the hold-out test set, and appends structured results to a JSON
file so the paper tables can be filled with genuine numbers.

Usage:
    python run_experiments.py encoder   # Table: encoder comparison
    python run_experiments.py ablation  # Table: ablation study
"""
from __future__ import annotations

import copy
import json
import os
import sys
import time


def _hb(msg):
    print(msg, flush=True)

import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR

from src.losses import SegmentationLoss
from src.metrics import SegMetricAccumulator
from src.models.unet_resnet import build_segmentation_model
from src.utils import build_dataloaders, get_device, load_config, set_seed

RESULTS = "results_real.json"


def _save(tag, rec):
    data = {}
    if os.path.exists(RESULTS):
        with open(RESULTS) as f:
            data = json.load(f)
    data[tag] = rec
    with open(RESULTS, "w") as f:
        json.dump(data, f, indent=2)


@torch.no_grad()
def _evaluate(model, loader, device, k):
    model.eval()
    acc = SegMetricAccumulator(k)
    for blurry, sharp, mask in loader:
        acc.update(model(sharp.to(device)), mask.to(device))
    return acc.compute()


def train_one(base_cfg, overrides, epochs, seed=0):
    cfg = copy.deepcopy(base_cfg)
    cfg.setdefault("model", {}).update(overrides)
    set_seed(seed)
    device = get_device(cfg)
    k = cfg["model"].get("num_classes", 3)

    model = build_segmentation_model(cfg).to(device)
    train_loader, val_loader, test_loader = build_dataloaders(cfg, seed=seed)
    tr = cfg["train"]
    crit = SegmentationLoss()
    opt = Adam(model.parameters(), lr=tr["lr"], betas=(0.9, 0.999), weight_decay=tr["weight_decay"])
    sched = CosineAnnealingLR(opt, T_max=epochs, eta_min=tr["min_lr"])

    best_mds, best_state = -1.0, None
    for ep in range(epochs):
        model.train()
        for blurry, sharp, mask in train_loader:
            sharp, mask = sharp.to(device), mask.to(device)
            opt.zero_grad()
            loss = crit(model(sharp), mask)
            loss.backward()
            opt.step()
        sched.step()
        val = _evaluate(model, val_loader, device, k)
        if val["mDice"] > best_mds:
            best_mds = val["mDice"]
            best_state = {kk: v.detach().cpu().clone() for kk, v in model.state_dict().items()}
        if (ep + 1) % 20 == 0 or ep == 0:
            _hb(f"    epoch {ep + 1}/{epochs} val_mDS={val['mDice']:.4f}")

    model.load_state_dict(best_state)
    test = _evaluate(model, test_loader, device, k)
    return {
        "dice_per_class": [round(float(x), 4) for x in test["dice_per_class"]],
        "mDS": round(test["mDice"], 4),
        "mIoU": round(test["mIoU"], 4),
        "mPrecision": round(test["mPrecision"], 4),
        "mRecall": round(test["mRecall"], 4),
        "best_val_mDS": round(best_mds, 4),
    }


def main():
    suite = sys.argv[1] if len(sys.argv) > 1 else "encoder"
    epochs = int(sys.argv[2]) if len(sys.argv) > 2 else 200
    cfg = load_config("configs/default.yaml")
    print(f"suite={suite} epochs={epochs} device={get_device(cfg)}")

    if suite == "encoder":
        configs = [
            ("ResNet-18", {"encoder": "resnet18"}),
            ("ResNet-34", {"encoder": "resnet34"}),
            ("ResNet-50", {"encoder": "resnet50"}),
            ("ResNet-101", {"encoder": "resnet101"}),
        ]
    elif suite == "ablation":
        configs = [
            ("Full Model", {"encoder": "resnet34"}),
            ("w/o ImageNet Pretrain", {"encoder": "resnet34", "pretrained": False}),
            ("w/o Skip Connections", {"encoder": "resnet34", "use_skip": False}),
            ("w/o Dropout", {"encoder": "resnet34", "dropout": 0.0}),
            ("Single Conv Decoder", {"encoder": "resnet34", "decoder_convs": 1}),
            ("3 Decoder Blocks", {"encoder": "resnet34", "num_decoder_blocks": 3}),
        ]
    else:
        raise SystemExit(f"unknown suite '{suite}'")

    for name, ov in configs:
        t = time.time()
        rec = train_one(cfg, ov, epochs)
        rec["minutes"] = round((time.time() - t) / 60, 1)
        _save(f"{suite}:{name}", rec)
        print(f"[{name}] {rec}", flush=True)


if __name__ == "__main__":
    main()
