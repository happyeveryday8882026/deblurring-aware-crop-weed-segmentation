"""Stage 1: pre-train the NAFNet deblurring front-end on paired patches.

Section 4 ("Restoration-aware training"): NAFNet is pre-trained for 300 epochs
on paired sharp/blurry patches using the L1 deblurring loss, with Adam and an
initial learning rate of 1e-3. The blurry patch is the input and the sharp
patch is the target.

Example
-------
    python -m src.train_deblur --config configs/default.yaml --out runs/nafnet
"""
from __future__ import annotations

import argparse
import os

import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR

from .losses import DeblurLoss
from .models.nafnet import build_deblur_model
from .utils import build_dataloaders, get_device, load_config, save_checkpoint, set_seed


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="runs/nafnet")
    p.add_argument("--epochs", type=int, default=None)
    return p.parse_args()


@torch.no_grad()
def _psnr(restored, sharp, eps=1e-8):
    mse = torch.mean((restored - sharp) ** 2).item()
    return 10.0 * torch.log10(torch.tensor(1.0 / (mse + eps))).item()


def main():
    args = parse_args()
    cfg = load_config(args.config)
    set_seed(args.seed)
    device = get_device(cfg)

    model = build_deblur_model(cfg).to(device)
    train_loader, val_loader, _ = build_dataloaders(cfg, seed=args.seed)

    epochs = args.epochs or cfg["deblur"]["epochs"]
    criterion = DeblurLoss()
    optimizer = Adam(model.parameters(), lr=cfg["deblur"]["lr"], betas=(0.9, 0.999))
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=cfg["deblur"]["min_lr"])

    best_psnr, best_path = -1.0, os.path.join(args.out, "nafnet_best.pt")
    for epoch in range(epochs):
        model.train()
        running = 0.0
        for blurry, sharp, _ in train_loader:
            blurry, sharp = blurry.to(device), sharp.to(device)
            optimizer.zero_grad()
            restored = model(blurry)
            loss = criterion(restored, sharp)
            loss.backward()
            optimizer.step()
            running += float(loss.detach()) * blurry.size(0)
        scheduler.step()

        model.eval()
        psnrs = []
        with torch.no_grad():
            for blurry, sharp, _ in val_loader:
                blurry, sharp = blurry.to(device), sharp.to(device)
                psnrs.append(_psnr(model(blurry), sharp))
        val_psnr = sum(psnrs) / max(1, len(psnrs))
        print(f"[epoch {epoch + 1:3d}/{epochs}] l1={running / len(train_loader.dataset):.4f} "
              f"val_PSNR={val_psnr:.2f}dB")
        if val_psnr > best_psnr:
            best_psnr = val_psnr
            save_checkpoint({"model": model.state_dict(), "config": cfg,
                             "val_psnr": best_psnr}, best_path)

    print(f"Best validation PSNR: {best_psnr:.2f} dB -> {best_path}")


if __name__ == "__main__":
    main()
