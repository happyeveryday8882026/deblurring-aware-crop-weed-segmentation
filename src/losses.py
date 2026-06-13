"""Loss functions (Section 3).

* segmentation: standard pixel-wise multi-class cross-entropy (Eq. 17 uses CE
  for L_seg). No explicit class-balancing or focal weighting is applied, matching
  the paper; an optional class-weight vector is supported for completeness.
* deblurring: L1 between the restored output and the sharp ground truth (Eq. 11).
* composite restoration-aware objective: L_seg + lambda * L_deblur (Eq. 16).
"""
from __future__ import annotations

import torch.nn as nn


class SegmentationLoss(nn.Module):
    def __init__(self, class_weights=None):
        super().__init__()
        self.ce = nn.CrossEntropyLoss(weight=class_weights)

    def forward(self, logits, target):
        return self.ce(logits, target)


class DeblurLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.L1Loss()

    def forward(self, restored, sharp):
        return self.l1(restored, sharp)


class CompositeLoss(nn.Module):
    """L_total = L_seg(S(restored), Y) + lambda * L_deblur(restored, sharp)."""

    def __init__(self, lam: float = 0.1, class_weights=None):
        super().__init__()
        self.seg = SegmentationLoss(class_weights)
        self.deblur = DeblurLoss()
        self.lam = lam

    def forward(self, logits, target, restored, sharp):
        l_seg = self.seg(logits, target)
        l_deblur = self.deblur(restored, sharp)
        return l_seg + self.lam * l_deblur, {
            "loss_seg": float(l_seg.detach()),
            "loss_deblur": float(l_deblur.detach()),
        }
