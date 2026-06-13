"""Re-train the proposed NAFNet + UNet-ResNet34 pipeline correctly.

Fixes a bug in run_blur.py where the restoration-aware fine-tuning fed the
segmentation network only NAFNet(blurry), so it failed on sharp inputs. Here
the segmentation network is trained on the restored versions of BOTH the sharp
and the blurred patches, so the proposed model handles both conditions.

Re-uses the pre-trained NAFNet from runs/nafnet_real.pt (no 150-epoch
re-pretrain). Writes:
  * blur:Proposed+DeBlur   -> {sharp, blurred, combined} mDS  (blur table)
  * sota:Proposed          -> full metrics on the combined set (SOTA table)

Usage: python run_proposed.py [epochs]
"""
from __future__ import annotations

import json
import os
import sys

import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR

from src.losses import CompositeLoss
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
def _eval(seg, naf, loader, device, k, source):
    seg.eval(); naf.eval()
    acc = SegMetricAccumulator(k)
    for blurry, sharp, mask in loader:
        srcs = {"sharp": [sharp], "blurred": [blurry], "combined": [sharp, blurry]}[source]
        for img in srcs:
            acc.update(seg(naf(img.to(device))), mask.to(device))
    return acc.compute()


def main():
    epochs = int(sys.argv[1]) if len(sys.argv) > 1 else 100
    cfg = load_config("configs/default.yaml")
    cfg.setdefault("model", {})["encoder"] = "resnet34"
    device = get_device(cfg)
    k = cfg["model"].get("num_classes", 3)
    set_seed(0)
    print(f"proposed re-train device={device} epochs={epochs}", flush=True)

    naf = build_deblur_model(cfg).to(device)
    naf.load_state_dict(torch.load(NAF_CKPT, map_location=device)["model"])
    seg = build_segmentation_model(cfg).to(device)

    train_loader, val_loader, test_loader = build_dataloaders(cfg, seed=0)
    tr = cfg["train"]
    crit = CompositeLoss(lam=cfg["deblur"].get("lambda", 0.1))
    opt = Adam([{"params": seg.parameters(), "lr": tr["lr"]},
                {"params": naf.parameters(), "lr": cfg["deblur"].get("finetune_lr", 1e-5)}],
               betas=(0.9, 0.999), weight_decay=tr["weight_decay"])
    sched = CosineAnnealingLR(opt, T_max=epochs, eta_min=tr["min_lr"])

    best, best_seg, best_naf = -1.0, None, None
    for ep in range(epochs):
        seg.train(); naf.train()
        for blurry, sharp, mask in train_loader:
            blurry, sharp, mask = blurry.to(device), sharp.to(device), mask.to(device)
            # Train on the restored versions of BOTH sharp and blurred inputs.
            for img in (sharp, blurry):
                opt.zero_grad()
                restored = naf(img)
                loss, _ = crit(seg(restored), mask, restored, sharp)
                loss.backward()
                opt.step()
        sched.step()
        v = _eval(seg, naf, val_loader, device, k, "combined")["mDice"]
        if v > best:
            best = v
            best_seg = {kk: vv.detach().cpu().clone() for kk, vv in seg.state_dict().items()}
            best_naf = {kk: vv.detach().cpu().clone() for kk, vv in naf.state_dict().items()}
        if (ep + 1) % 20 == 0 or ep == 0:
            print(f"    epoch {ep+1}/{epochs} val_combined_mDS={v:.4f}", flush=True)
    seg.load_state_dict(best_seg); naf.load_state_dict(best_naf)

    sharp = _eval(seg, naf, test_loader, device, k, "sharp")["mDice"]
    blur = _eval(seg, naf, test_loader, device, k, "blurred")["mDice"]
    comb = _eval(seg, naf, test_loader, device, k, "combined")
    _save("blur:Proposed+DeBlur", {"sharp": round(sharp, 4), "blurred": round(blur, 4),
                                   "combined": round(comb["mDice"], 4)})
    _save("sota:Proposed", {"mDS": round(comb["mDice"], 4), "mIoU": round(comb["mIoU"], 4),
                            "mPrecision": round(comb["mPrecision"], 4), "mRecall": round(comb["mRecall"], 4)})
    torch.save({"seg": best_seg, "naf": best_naf}, "runs/proposed_real.pt")
    print(f"DONE sharp={sharp:.4f} blurred={blur:.4f} combined={comb['mDice']:.4f}", flush=True)


if __name__ == "__main__":
    main()
