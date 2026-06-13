"""Parameter count and MACs analysis (Table 8 of the paper).

Both metrics are measured at the standard 128x128 input resolution with the
``thop`` library, matching Section 4 ("Computational Complexity Analysis").
Reports the segmentation-only model, the NAFNet front-end, and the full
deblur+segmentation pipeline.

Example
-------
    python -m src.complexity --config configs/default.yaml --encoder resnet50
"""
from __future__ import annotations

import argparse

import torch

from .models.nafnet import build_deblur_model
from .models.unet_resnet import build_segmentation_model
from .utils import load_config


def _profile(model, x):
    from thop import profile  # imported lazily so the rest of the package runs without thop

    macs, params = profile(model, inputs=(x,), verbose=False)
    return macs, params


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--encoder", default=None)
    p.add_argument("--size", type=int, default=128)
    return p.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config)
    if args.encoder:
        cfg.setdefault("model", {})["encoder"] = args.encoder

    x = torch.randn(1, 3, args.size, args.size)
    seg = build_segmentation_model(cfg).eval()
    deblur = build_deblur_model(cfg).eval()

    seg_macs, seg_params = _profile(seg, x)
    deb_macs, deb_params = _profile(deblur, x)

    def fmt(macs, params):
        return f"Params={params / 1e6:6.2f} M | MACs={macs / 1e9:6.2f} G"

    enc = cfg["model"].get("encoder", "resnet50")
    print(f"Input: {args.size}x{args.size}")
    print(f"NAFNet (deblur only)        : {fmt(deb_macs, deb_params)}")
    print(f"Proposed Seg (UNet-{enc})   : {fmt(seg_macs, seg_params)}")
    print(f"Proposed (Deblur + Seg)     : {fmt(seg_macs + deb_macs, seg_params + deb_params)}")


if __name__ == "__main__":
    main()
