"""Segmentation metrics: per-class Dice, IoU, Precision, Recall and their means.

Implements the per-class Dice/IoU/Precision/Recall equations of the paper for
the three semantic classes (background, crop, weed). Statistics are accumulated over the
whole evaluation set via confusion-matrix-style counters (intersection, union,
predicted-positive, ground-truth-positive) and reduced at the end, which is
equivalent to computing the metrics on the concatenation of all patches.
"""
from __future__ import annotations

import numpy as np
import torch


class SegMetricAccumulator:
    def __init__(self, num_classes: int = 3):
        self.k = num_classes
        self.inter = np.zeros(num_classes, dtype=np.float64)
        self.pred = np.zeros(num_classes, dtype=np.float64)
        self.gt = np.zeros(num_classes, dtype=np.float64)
        self.union = np.zeros(num_classes, dtype=np.float64)

    @torch.no_grad()
    def update(self, logits_or_pred, target):
        if logits_or_pred.dim() == 4:
            pred = logits_or_pred.argmax(dim=1)
        else:
            pred = logits_or_pred
        pred = pred.flatten().cpu().numpy()
        target = target.flatten().cpu().numpy()
        for c in range(self.k):
            p = pred == c
            g = target == c
            inter = np.logical_and(p, g).sum()
            self.inter[c] += inter
            self.pred[c] += p.sum()
            self.gt[c] += g.sum()
            self.union[c] += np.logical_or(p, g).sum()

    def compute(self, eps: float = 1e-7):
        dice = 2 * self.inter / (self.pred + self.gt + eps)
        iou = self.inter / (self.union + eps)
        precision = self.inter / (self.pred + eps)
        recall = self.inter / (self.gt + eps)
        return {
            "dice_per_class": dice,
            "iou_per_class": iou,
            "precision_per_class": precision,
            "recall_per_class": recall,
            "mDice": float(dice.mean()),
            "mIoU": float(iou.mean()),
            "mPrecision": float(precision.mean()),
            "mRecall": float(recall.mean()),
        }
