"""Three-class plant segmentation dataset (background, crop/sorghum, weed).

The imagery is the publicly available DeBlurWeedSeg dataset of Genze et al.
(Mendeley Data, https://doi.org/10.17632/k4gvsjv4t3.1): 1,300 paired
sharp/blurry 128x128 patches with three-class segmentation masks. Each
``gt/<id>.png`` is a composite grid in which sub-images are separated by a
4-pixel border:

    sharp image  : rows   4:132, cols   4:132
    blurry image : rows   4:132, cols 136:264
    sharp mask   : rows 136:264, cols   4:132   (RGB-coded)
    blurry mask  : rows 136:264, cols 136:264

The masks encode three classes via RGB colours:
    [199,199,199] -> background, [31,119,180] -> crop (sorghum), [255,127,14] -> weed.

These are the only classes present in the public data; no other annotation
layer is used.
"""
from __future__ import annotations

import csv
import os

import numpy as np
import torch
from torch.utils.data import Dataset

try:  # skimage is optional; fall back to PIL if unavailable.
    from skimage import io as skio

    def _imread(path):
        return skio.imread(path)
except Exception:  # pragma: no cover
    from PIL import Image

    def _imread(path):
        return np.array(Image.open(path))


# RGB colours of the three classes in the composite masks.
_BASE_COLORS = np.array(
    [[199, 199, 199], [31, 119, 180], [255, 127, 14]], dtype=np.uint8
)

CLASS_NAMES = ["background", "crop", "weed"]
NUM_CLASSES = 3


def read_split(csv_path: str):
    """Read a split csv (a single ``filename`` column) into a list of ids."""
    ids = []
    with open(csv_path, newline="") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        for row in reader:
            if not row:
                continue
            ids.append(row[0].strip())
    # If the file had no header row, keep the first value too.
    if header and header[0].strip().lower() != "filename":
        ids = [header[0].strip()] + ids
    return ids


def _encode_mask(rgb_mask: np.ndarray) -> np.ndarray:
    label = np.zeros(rgb_mask.shape[:2], dtype=np.uint8)
    for idx, color in enumerate(_BASE_COLORS):
        label[(rgb_mask == color).all(axis=2)] = idx
    return label


class PlantSegDataset(Dataset):
    """Returns ``(blurry, sharp, mask)`` tensors for one patch.

    ``blurry`` and ``sharp`` are float tensors in ``[0, 1]`` of shape
    ``(3, 128, 128)``; ``mask`` is a long tensor of shape ``(128, 128)`` with
    values in ``{0, 1, 2}``.
    """

    def __init__(self, ids, image_dir: str, transform=None, cache: bool = True):
        self.ids = list(ids)
        self.image_dir = image_dir
        self.transform = transform
        # The full dataset is tiny (~64 MB); decoding the PNGs once and keeping
        # the cropped uint8 arrays in RAM removes the per-epoch disk-I/O
        # bottleneck (otherwise every epoch re-reads and re-decodes 1300 PNGs).
        self._cache = None
        if cache:
            self._cache = [self._load_pair(sid) for sid in self.ids]

    def __len__(self):
        return len(self.ids)

    def _load_pair(self, sample_id: str):
        img = _imread(os.path.join(self.image_dir, f"{sample_id}.png"))
        sharp = img[4:132, 4:132, :3]
        blurry = img[4:132, 136:264, :3]
        mask = _encode_mask(img[136:264, 4:132, :3])
        return sharp.copy(), blurry.copy(), mask.copy()

    def __getitem__(self, idx):
        if self._cache is not None:
            sharp, blurry, mask = (a.copy() for a in self._cache[idx])
        else:
            sharp, blurry, mask = self._load_pair(self.ids[idx])

        if self.transform is not None:
            sharp, blurry, mask = self.transform(sharp, blurry, mask)
        else:
            sharp = torch.from_numpy(sharp).permute(2, 0, 1).float() / 255.0
            blurry = torch.from_numpy(blurry).permute(2, 0, 1).float() / 255.0
            mask = torch.from_numpy(mask).long()

        return blurry, sharp, mask


# Backwards-compatible alias.
PlantHealthDataset = PlantSegDataset
