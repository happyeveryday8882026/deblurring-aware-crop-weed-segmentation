"""Stage: train the UNet-ResNet segmentation network on sharp patches.

Reproduces Table 2 (encoder comparison) and Table 5 (ablations). Optimiser and
schedule follow Section 4 ("Segmentation training"): Adam (beta1=0.9,
beta2=0.999), initial lr 1e-3, weight decay 1e-4, cosine-annealing to 1e-6,
200 epochs, batch size 16, pixel-wise cross-entropy. The checkpoint with the
highest validation mDS is retained.

Example
-------
    python -m src.train_seg --config configs/default.yaml \
        --encoder resnet50 --seed 0 --out runs/resnet50_seed0

Ablations (override model fields):
    --no-pretrain         w/o ImageNet pretraining
    --no-skip             w/o skip connections
    --dropout 0.0         w/o dropout
    --decoder-convs 1     single-conv decoder
    --decoder-blocks 3    3 decoder blocks
"""
from __future__ import annotations

import argparse
import os

import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR

from .losses import SegmentationLoss
from .metrics import SegMetricAccumulator
from .models.unet_resnet import build_segmentation_model
from .utils import (
    build_dataloaders,
    get_device,
    load_config,
    save_checkpoint,
    set_seed,
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--encoder", default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="runs/seg")
    p.add_argument("--no-pretrain", action="store_true")
    p.add_argument("--no-skip", action="store_true")
    p.add_argument("--dropout", type=float, default=None)
    p.add_argument("--decoder-convs", type=int, default=None)
    p.add_argument("--decoder-blocks", type=int, default=None)
    p.add_argument("--epochs", type=int, default=None)
    return p.parse_args()


def apply_overrides(cfg, args):
    m = cfg.setdefault("model", {})
    if args.encoder:
        m["encoder"] = args.encoder
    if args.no_pretrain:
        m["pretrained"] = False
    if args.no_skip:
        m["use_skip"] = False
    if args.dropout is not None:
        m["dropout"] = args.dropout
    if args.decoder_convs is not None:
        m["decoder_convs"] = args.decoder_convs
    if args.decoder_blocks is not None:
        m["num_decoder_blocks"] = args.decoder_blocks
    if args.epochs is not None:
        cfg["train"]["epochs"] = args.epochs
    return cfg


@torch.no_grad()
def evaluate_loader(model, loader, device, num_classes):
    model.eval()
    acc = SegMetricAccumulator(num_classes)
    for blurry, sharp, mask in loader:
        sharp, mask = sharp.to(device), mask.to(device)
        logits = model(sharp)
        acc.update(logits, mask)
    return acc.compute()


def main():
    args = parse_args()
    cfg = apply_overrides(load_config(args.config), args)
    set_seed(args.seed)
    device = get_device(cfg)

    model = build_segmentation_model(cfg).to(device)
    train_loader, val_loader, test_loader = build_dataloaders(cfg, seed=args.seed)

    tr = cfg["train"]
    epochs = tr["epochs"]
    criterion = SegmentationLoss()
    optimizer = Adam(model.parameters(), lr=tr["lr"], betas=(0.9, 0.999),
                     weight_decay=tr["weight_decay"])
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=tr["min_lr"])

    num_classes = cfg["model"].get("num_classes", 3)
    best_mds, best_path = -1.0, os.path.join(args.out, "best.pt")

    for epoch in range(epochs):
        model.train()
        running = 0.0
        for blurry, sharp, mask in train_loader:
            sharp, mask = sharp.to(device), mask.to(device)
            optimizer.zero_grad()
            logits = model(sharp)
            loss = criterion(logits, mask)
            loss.backward()
            optimizer.step()
            running += float(loss.detach()) * sharp.size(0)
        scheduler.step()

        val = evaluate_loader(model, val_loader, device, num_classes)
        train_loss = running / len(train_loader.dataset)
        print(f"[epoch {epoch + 1:3d}/{epochs}] train_loss={train_loss:.4f} "
              f"val_mDS={val['mDice']:.4f} val_mIoU={val['mIoU']:.4f}")

        if val["mDice"] > best_mds:
            best_mds = val["mDice"]
            save_checkpoint(
                {"model": model.state_dict(), "config": cfg, "val_mDice": best_mds,
                 "seed": args.seed},
                best_path,
            )

    # Final test with the best checkpoint.
    state = torch.load(best_path, map_location=device)
    model.load_state_dict(state["model"])
    test = evaluate_loader(model, test_loader, device, num_classes)
    print("=== TEST (best val checkpoint) ===")
    print(f"mDS={test['mDice']:.4f} mIoU={test['mIoU']:.4f} "
          f"mPrecision={test['mPrecision']:.4f} mRecall={test['mRecall']:.4f}")
    print("per-class Dice:", [round(float(x), 4) for x in test["dice_per_class"]])


if __name__ == "__main__":
    main()
