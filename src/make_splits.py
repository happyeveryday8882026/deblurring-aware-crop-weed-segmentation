"""Generate stratified 70/15/15 train/val/test splits at the patch level.

Section 4 of the paper: "the dataset is split into training, validation, and
test subsets in a stratified 70/15/15 ratio at the patch level, ensuring that
the class distribution is preserved across all splits".

Patches are stratified by their dominant minority class (the rarest class
present in the base 3-class mask), then split deterministically with a fixed
seed so the partition is fully reproducible.

Example
-------
    python -m src.make_splits \
        --image-dir "../Data .../data/gt" --out data_splits --seed 0
"""
from __future__ import annotations

import argparse
import csv
import os

import numpy as np

from .data.dataset import _encode_mask, _imread


def _dominant_minority(mask: np.ndarray) -> int:
    """Stratification key: rarest non-background class present, else background."""
    present = [c for c in (2, 1, 0) if (mask == c).any()]
    return present[0] if present else 0


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--image-dir", required=True)
    p.add_argument("--out", default="data_splits")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--ratios", nargs=3, type=float, default=[0.70, 0.15, 0.15])
    args = p.parse_args()

    ids = sorted(
        os.path.splitext(f)[0] for f in os.listdir(args.image_dir) if f.endswith(".png")
    )
    keys = {}
    for sid in ids:
        img = _imread(os.path.join(args.image_dir, f"{sid}.png"))
        keys[sid] = _dominant_minority(_encode_mask(img[136:264, 4:132, :3]))

    rng = np.random.default_rng(args.seed)
    train, val, test = [], [], []
    for cls in sorted(set(keys.values())):
        group = [s for s in ids if keys[s] == cls]
        rng.shuffle(group)
        n = len(group)
        n_tr = int(round(args.ratios[0] * n))
        n_val = int(round(args.ratios[1] * n))
        train += group[:n_tr]
        val += group[n_tr:n_tr + n_val]
        test += group[n_tr + n_val:]

    os.makedirs(args.out, exist_ok=True)
    for name, group in [("train", train), ("val", val), ("test", test)]:
        with open(os.path.join(args.out, f"{name}.csv"), "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["filename"])
            for s in sorted(group):
                w.writerow([s])
    print(f"train={len(train)} val={len(val)} test={len(test)} -> {args.out}")


if __name__ == "__main__":
    main()
