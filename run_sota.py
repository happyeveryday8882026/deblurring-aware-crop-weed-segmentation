"""Driver: real comparison against lightweight SOTA segmentation baselines.

Uses segmentation_models_pytorch (smp) to build three standard CNN segmentation
architectures with a shared ResNet-50 encoder, trained under the identical
three-class protocol as the proposed model:
  * DeepLabV3+
  * U-Net++
  * FPN

Each baseline is reported in two configurations:
  * w/o NAF : evaluated directly on the (sharp+blurred) combined test set.
  * w/ NAF  : evaluated on the same test set after the shared pre-trained NAFNet
              restoration front-end (runs/nafnet_real.pt) is applied at input.

Usage:
    python run_sota.py [epochs]
"""
from __future__ import annotations

import json
import os
import sys
import time

import segmentation_models_pytorch as smp
import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR

from src.losses import SegmentationLoss
from src.metrics import SegMetricAccumulator
from src.models.nafnet import build_deblur_model
from src.utils import build_dataloaders, get_device, load_config, set_seed

RESULTS = "results_real.json"
NAF_CKPT = "runs/nafnet_real.pt"


def _save(tag, rec):
    data = {}
    if os.path.exists(RESULTS):
        with open(RESULTS) as f:
            data = json.load(f)
    data[tag] = rec
    with open(RESULTS, "w") as f:
        json.dump(data, f, indent=2)


def build_baseline(name, k):
    kw = dict(encoder_name="resnet34", encoder_weights="imagenet", in_channels=3, classes=k)
    if name == "DeepLabV3+":
        return smp.DeepLabV3Plus(**kw)
    if name == "U-Net++":
        return smp.UnetPlusPlus(**kw)
    if name == "FPN":
        return smp.FPN(**kw)
    raise ValueError(name)


@torch.no_grad()
def evaluate(model, loader, device, k, deblur=None):
    model.eval()
    acc = SegMetricAccumulator(k)
    for blurry, sharp, mask in loader:
        # Combined test set: evaluate on both sharp and blurred inputs.
        for img in (sharp, blurry):
            x = img.to(device)
            if deblur is not None:
                x = deblur(x)
            acc.update(model(x), mask.to(device))
    return acc.compute()


def train_baseline(name, cfg, device, epochs, k):
    set_seed(0)
    model = build_baseline(name, k).to(device)
    train_loader, val_loader, _ = build_dataloaders(cfg, seed=0)
    tr = cfg["train"]
    crit = SegmentationLoss()
    opt = Adam(model.parameters(), lr=tr["lr"], betas=(0.9, 0.999), weight_decay=tr["weight_decay"])
    sched = CosineAnnealingLR(opt, T_max=epochs, eta_min=tr["min_lr"])
    best, best_state = -1.0, None
    for _ in range(epochs):
        model.train()
        for blurry, sharp, mask in train_loader:
            sharp, mask = sharp.to(device), mask.to(device)
            opt.zero_grad()
            loss = crit(model(sharp), mask)
            loss.backward()
            opt.step()
        sched.step()
        v = evaluate(model, val_loader, device, k)["mDice"]
        if v > best:
            best = v
            best_state = {kk: vv.detach().cpu().clone() for kk, vv in model.state_dict().items()}
    model.load_state_dict(best_state)
    return model


def main():
    epochs = int(sys.argv[1]) if len(sys.argv) > 1 else 200
    cfg = load_config("configs/default.yaml")
    device = get_device(cfg)
    k = cfg["model"].get("num_classes", 3)
    print(f"SOTA experiment device={device} epochs={epochs}", flush=True)
    _, _, test_loader = build_dataloaders(cfg, seed=0)

    deblur = None
    if os.path.exists(NAF_CKPT):
        deblur = build_deblur_model(cfg).to(device)
        deblur.load_state_dict(torch.load(NAF_CKPT, map_location=device)["model"])
        deblur.eval()

    for name in ["DeepLabV3+", "U-Net++", "FPN"]:
        t = time.time()
        model = train_baseline(name, cfg, device, epochs, k)
        r_wo = evaluate(model, test_loader, device, k)
        _save(f"sota:{name} (w/o NAF)", {
            "mDS": round(r_wo["mDice"], 4), "mIoU": round(r_wo["mIoU"], 4),
            "mPrecision": round(r_wo["mPrecision"], 4), "mRecall": round(r_wo["mRecall"], 4),
            "minutes": round((time.time() - t) / 60, 1)})
        if deblur is not None:
            r_w = evaluate(model, test_loader, device, k, deblur=deblur)
            _save(f"sota:{name} (w/ NAF)", {
                "mDS": round(r_w["mDice"], 4), "mIoU": round(r_w["mIoU"], 4),
                "mPrecision": round(r_w["mPrecision"], 4), "mRecall": round(r_w["mRecall"], 4)})
        print(f"{name} done", flush=True)


if __name__ == "__main__":
    main()
