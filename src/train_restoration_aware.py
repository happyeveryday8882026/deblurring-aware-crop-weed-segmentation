"""Stage 2: restoration-aware fine-tuning of the deblur+segmentation pipeline.

Section 3 ("Integration with the Segmentation Network") and Section 4
("Restoration-aware training"). The blurry input is restored by the pre-trained
NAFNet and then segmented by the UNet-ResNet network. The segmentation network
is trained with the composite objective L_total = L_seg + lambda * L_deblur
(lambda = 0.1); the restoration parameters are jointly updated with a reduced
learning rate (1e-5) so the restored representation is adapted for downstream
segmentability without overwriting the deblurring prior.

Example
-------
    python -m src.train_restoration_aware --config configs/default.yaml \
        --nafnet runs/nafnet/nafnet_best.pt --encoder resnet50 \
        --seed 0 --out runs/pipeline_resnet50_seed0
"""
from __future__ import annotations

import argparse
import os

import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR

from .losses import CompositeLoss
from .metrics import SegMetricAccumulator
from .models.nafnet import build_deblur_model
from .models.unet_resnet import build_segmentation_model
from .utils import build_dataloaders, get_device, load_config, save_checkpoint, set_seed


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--nafnet", required=True, help="path to the Stage-1 NAFNet checkpoint")
    p.add_argument("--encoder", default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="runs/pipeline")
    p.add_argument("--epochs", type=int, default=None)
    return p.parse_args()


@torch.no_grad()
def evaluate_pipeline(deblur, seg, loader, device, num_classes, use_deblur=True):
    deblur.eval()
    seg.eval()
    acc = SegMetricAccumulator(num_classes)
    for blurry, sharp, mask in loader:
        blurry, mask = blurry.to(device), mask.to(device)
        x = deblur(blurry) if use_deblur else blurry
        acc.update(seg(x), mask)
    return acc.compute()


def main():
    args = parse_args()
    cfg = load_config(args.config)
    if args.encoder:
        cfg.setdefault("model", {})["encoder"] = args.encoder
    set_seed(args.seed)
    device = get_device(cfg)

    deblur = build_deblur_model(cfg).to(device)
    deblur.load_state_dict(torch.load(args.nafnet, map_location=device)["model"])
    seg = build_segmentation_model(cfg).to(device)

    train_loader, val_loader, test_loader = build_dataloaders(cfg, seed=args.seed)
    tr = cfg["train"]
    epochs = args.epochs or tr["epochs"]
    lam = cfg["deblur"].get("lambda", 0.1)
    ft_lr = cfg["deblur"].get("finetune_lr", 1e-5)

    criterion = CompositeLoss(lam=lam)
    optimizer = Adam(
        [
            {"params": seg.parameters(), "lr": tr["lr"]},
            {"params": deblur.parameters(), "lr": ft_lr},
        ],
        betas=(0.9, 0.999),
        weight_decay=tr["weight_decay"],
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=tr["min_lr"])

    num_classes = cfg["model"].get("num_classes", 3)
    best_mds, best_path = -1.0, os.path.join(args.out, "pipeline_best.pt")

    for epoch in range(epochs):
        deblur.train()
        seg.train()
        running = 0.0
        for blurry, sharp, mask in train_loader:
            blurry, sharp, mask = blurry.to(device), sharp.to(device), mask.to(device)
            optimizer.zero_grad()
            restored = deblur(blurry)
            logits = seg(restored)
            loss, _ = criterion(logits, mask, restored, sharp)
            loss.backward()
            optimizer.step()
            running += float(loss.detach()) * blurry.size(0)
        scheduler.step()

        val = evaluate_pipeline(deblur, seg, val_loader, device, num_classes)
        print(f"[epoch {epoch + 1:3d}/{epochs}] loss={running / len(train_loader.dataset):.4f} "
              f"val_mDS={val['mDice']:.4f}")
        if val["mDice"] > best_mds:
            best_mds = val["mDice"]
            save_checkpoint({"deblur": deblur.state_dict(), "seg": seg.state_dict(),
                             "config": cfg, "val_mDice": best_mds, "seed": args.seed}, best_path)

    state = torch.load(best_path, map_location=device)
    deblur.load_state_dict(state["deblur"])
    seg.load_state_dict(state["seg"])
    test = evaluate_pipeline(deblur, seg, test_loader, device, num_classes)
    print("=== TEST (deblur + seg) ===")
    print(f"mDS={test['mDice']:.4f} mIoU={test['mIoU']:.4f} "
          f"mPrecision={test['mPrecision']:.4f} mRecall={test['mRecall']:.4f}")


if __name__ == "__main__":
    main()
