"""Driver: real deblurring-comparison experiment (blur table).

Produces genuine numbers for the three rows of the motion-blur table:
  * WeedSeg (Scenario 1): UNet-ResNet50 trained on BOTH sharp and blurred patches.
  * WeedSeg (Scenario 2): UNet-ResNet50 trained on sharp patches only.
  * Proposed + DeBlur    : NAFNet restoration front-end + UNet-ResNet50,
                           two-stage restoration-aware training.

Each model is evaluated by mean Dice Score on the Sharp, Motion-Blurred and
Combined test subsets. The trained NAFNet checkpoint is saved to
``runs/nafnet_real.pt`` for reuse by the SOTA "w/ NAF" experiment.

Usage:
    python run_blur.py [seg_epochs] [naf_epochs]
"""
from __future__ import annotations

import json
import os
import sys
import time

import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR

from src.losses import CompositeLoss, DeblurLoss, SegmentationLoss
from src.metrics import SegMetricAccumulator
from src.models.nafnet import build_deblur_model
from src.models.unet_resnet import build_segmentation_model
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


@torch.no_grad()
def _mds(model, loader, device, k, source="sharp", deblur=None):
    """mean Dice on a test loader; source selects the input image."""
    model.eval()
    acc = SegMetricAccumulator(k)
    for blurry, sharp, mask in loader:
        x = sharp if source == "sharp" else blurry
        x = x.to(device)
        if deblur is not None:
            x = deblur(x)
        acc.update(model(x), mask.to(device))
    return acc.compute()["mDice"]


def eval_subsets(model, loader, device, k, deblur=None):
    sharp = _mds(model, loader, device, k, "sharp", deblur)
    blur = _mds(model, loader, device, k, "blurry", deblur)
    return {
        "sharp": round(sharp, 4),
        "blurred": round(blur, 4),
        "combined": round((sharp + blur) / 2, 4),
    }


def train_seg(cfg, device, epochs, k, both=False):
    """Train UNet-ResNet50 on sharp only (both=False) or sharp+blurred (both=True)."""
    set_seed(0)
    model = build_segmentation_model(cfg).to(device)
    train_loader, val_loader, _ = build_dataloaders(cfg, seed=0)
    tr = cfg["train"]
    crit = SegmentationLoss()
    opt = Adam(model.parameters(), lr=tr["lr"], betas=(0.9, 0.999), weight_decay=tr["weight_decay"])
    sched = CosineAnnealingLR(opt, T_max=epochs, eta_min=tr["min_lr"])
    best, best_state = -1.0, None
    for _ in range(epochs):
        model.train()
        for blurry, sharp, mask in train_loader:
            mask = mask.to(device)
            inputs = [sharp.to(device)] + ([blurry.to(device)] if both else [])
            opt.zero_grad()
            loss = sum(crit(model(x), mask) for x in inputs)
            loss.backward()
            opt.step()
        sched.step()
        v = _mds(model, val_loader, device, k, "sharp")
        if v > best:
            best = v
            best_state = {kk: vv.detach().cpu().clone() for kk, vv in model.state_dict().items()}
    model.load_state_dict(best_state)
    return model


def train_nafnet(cfg, device, epochs):
    set_seed(0)
    naf = build_deblur_model(cfg).to(device)
    train_loader, _, _ = build_dataloaders(cfg, seed=0)
    crit = DeblurLoss()
    opt = Adam(naf.parameters(), lr=cfg["deblur"]["lr"], betas=(0.9, 0.999))
    sched = CosineAnnealingLR(opt, T_max=epochs, eta_min=cfg["deblur"]["min_lr"])
    for _ in range(epochs):
        naf.train()
        for blurry, sharp, _ in train_loader:
            blurry, sharp = blurry.to(device), sharp.to(device)
            opt.zero_grad()
            loss = crit(naf(blurry), sharp)
            loss.backward()
            opt.step()
        sched.step()
    os.makedirs(os.path.dirname(NAF_CKPT), exist_ok=True)
    torch.save({"model": naf.state_dict()}, NAF_CKPT)
    return naf


def train_restoration_aware(cfg, device, naf, epochs, k):
    set_seed(0)
    seg = build_segmentation_model(cfg).to(device)
    train_loader, val_loader, _ = build_dataloaders(cfg, seed=0)
    tr = cfg["train"]
    lam = cfg["deblur"].get("lambda", 0.1)
    ft_lr = cfg["deblur"].get("finetune_lr", 1e-5)
    crit = CompositeLoss(lam=lam)
    opt = Adam([{"params": seg.parameters(), "lr": tr["lr"]},
                {"params": naf.parameters(), "lr": ft_lr}],
               betas=(0.9, 0.999), weight_decay=tr["weight_decay"])
    sched = CosineAnnealingLR(opt, T_max=epochs, eta_min=tr["min_lr"])
    best, best_seg, best_naf = -1.0, None, None
    for _ in range(epochs):
        seg.train(); naf.train()
        for blurry, sharp, mask in train_loader:
            blurry, sharp, mask = blurry.to(device), sharp.to(device), mask.to(device)
            opt.zero_grad()
            restored = naf(blurry)
            loss, _ = crit(seg(restored), mask, restored, sharp)
            loss.backward()
            opt.step()
        sched.step()
        v = _mds(seg, val_loader, device, k, "blurry", deblur=naf)
        if v > best:
            best = v
            best_seg = {kk: vv.detach().cpu().clone() for kk, vv in seg.state_dict().items()}
            best_naf = {kk: vv.detach().cpu().clone() for kk, vv in naf.state_dict().items()}
    seg.load_state_dict(best_seg); naf.load_state_dict(best_naf)
    return seg, naf


def main():
    seg_epochs = int(sys.argv[1]) if len(sys.argv) > 1 else 200
    naf_epochs = int(sys.argv[2]) if len(sys.argv) > 2 else 300
    cfg = load_config("configs/default.yaml")
    cfg.setdefault("model", {})["encoder"] = "resnet34"
    device = get_device(cfg)
    k = cfg["model"].get("num_classes", 3)
    print(f"blur experiment device={device} seg_epochs={seg_epochs} naf_epochs={naf_epochs}", flush=True)
    _, _, test_loader = build_dataloaders(cfg, seed=0)

    t = time.time()
    m2 = train_seg(cfg, device, seg_epochs, k, both=False)
    _save("blur:WeedSeg-Scenario2", eval_subsets(m2, test_loader, device, k))
    print("Scenario2 done", flush=True)

    m1 = train_seg(cfg, device, seg_epochs, k, both=True)
    _save("blur:WeedSeg-Scenario1", eval_subsets(m1, test_loader, device, k))
    print("Scenario1 done", flush=True)

    naf = train_nafnet(cfg, device, naf_epochs)
    seg, naf = train_restoration_aware(cfg, device, naf, seg_epochs, k)
    _save("blur:Proposed+DeBlur", eval_subsets(seg, test_loader, device, k, deblur=naf))
    print(f"Proposed done; total {round((time.time()-t)/60,1)} min", flush=True)


if __name__ == "__main__":
    main()
