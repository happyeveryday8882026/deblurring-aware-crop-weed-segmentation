"""Shared utilities: config loading, seeding, device, checkpoints, dataloaders."""
from __future__ import annotations

import os
import random

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from .data.dataset import PlantSegDataset, read_split
from .data.transforms import EvalTransform, TrainTransform


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def set_seed(seed: int):
    """Seed Python, NumPy and PyTorch for reproducible runs (seeds 0-4 in paper)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device(cfg: dict) -> torch.device:
    want = cfg.get("device", "auto")
    if want in ("cuda", "auto") and torch.cuda.is_available():
        return torch.device("cuda")
    if want in ("mps", "auto") and getattr(torch.backends, "mps", None) is not None \
            and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def build_dataloaders(cfg: dict, seed: int = 0):
    data = cfg["data"]
    image_dir = data["image_dir"]
    splits_dir = data["splits_dir"]
    bs = cfg["train"]["batch_size"]
    nw = data.get("num_workers", 4)

    train_ids = read_split(os.path.join(splits_dir, "train.csv"))
    val_ids = read_split(os.path.join(splits_dir, "val.csv"))
    test_ids = read_split(os.path.join(splits_dir, "test.csv"))

    train_ds = PlantSegDataset(train_ids, image_dir, TrainTransform(seed=seed))
    val_ds = PlantSegDataset(val_ids, image_dir, EvalTransform())
    test_ds = PlantSegDataset(test_ids, image_dir, EvalTransform())

    g = torch.Generator()
    g.manual_seed(seed)
    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True, num_workers=nw,
                              pin_memory=True, generator=g, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=bs, shuffle=False, num_workers=nw, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=bs, shuffle=False, num_workers=nw, pin_memory=True)
    return train_loader, val_loader, test_loader


def save_checkpoint(state: dict, path: str):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    torch.save(state, path)


def load_checkpoint(path: str, map_location="cpu"):
    return torch.load(path, map_location=map_location)
