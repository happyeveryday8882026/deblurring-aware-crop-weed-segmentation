"""On-line data augmentation (Section 4, "Data preparation").

The training pipeline applies random horizontal/vertical flips, random 90-degree
rotations, and mild brightness/contrast jitter (+/-20%). The *identical*
geometric transform is applied to the sharp image, the blurry image and the
mask so that the paired sharp/blurry/label correspondence is preserved (this is
required for the restoration-aware training stage). No augmentation is used at
validation or test time.

A NumPy implementation is used to avoid a hard dependency on a specific
augmentation library; the operations match the description in the manuscript.
"""
from __future__ import annotations

import numpy as np
import torch


class TrainTransform:
    def __init__(self, brightness=0.2, contrast=0.2, seed=None):
        self.brightness = brightness
        self.contrast = contrast
        self.rng = np.random.default_rng(seed)

    def _photometric(self, img):
        # img: float array in [0, 1]; applied to image only, not the mask.
        b = 1.0 + self.rng.uniform(-self.brightness, self.brightness)
        c = 1.0 + self.rng.uniform(-self.contrast, self.contrast)
        mean = img.mean(axis=(0, 1), keepdims=True)
        img = (img - mean) * c + mean * b
        return np.clip(img, 0.0, 1.0)

    def __call__(self, sharp, blurry, mask):
        sharp = sharp.astype(np.float32) / 255.0
        blurry = blurry.astype(np.float32) / 255.0

        if self.rng.random() < 0.5:  # horizontal flip
            sharp, blurry, mask = sharp[:, ::-1], blurry[:, ::-1], mask[:, ::-1]
        if self.rng.random() < 0.5:  # vertical flip
            sharp, blurry, mask = sharp[::-1], blurry[::-1], mask[::-1]
        k = int(self.rng.integers(0, 4))  # random 90-degree rotation
        if k:
            sharp = np.rot90(sharp, k)
            blurry = np.rot90(blurry, k)
            mask = np.rot90(mask, k)

        sharp = self._photometric(np.ascontiguousarray(sharp))
        blurry = self._photometric(np.ascontiguousarray(blurry))
        mask = np.ascontiguousarray(mask)

        sharp = torch.from_numpy(sharp).permute(2, 0, 1).float()
        blurry = torch.from_numpy(blurry).permute(2, 0, 1).float()
        mask = torch.from_numpy(mask.astype(np.int64)).long()
        return sharp, blurry, mask


class EvalTransform:
    def __call__(self, sharp, blurry, mask):
        sharp = torch.from_numpy(sharp).permute(2, 0, 1).float() / 255.0
        blurry = torch.from_numpy(blurry).permute(2, 0, 1).float() / 255.0
        mask = torch.from_numpy(mask.astype(np.int64)).long()
        return sharp, blurry, mask
