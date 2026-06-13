"""ResNet encoder backbone for the UNet-ResNet segmentation network.

The four supported variants (ResNet-18/34/50/101) wrap the corresponding
torchvision models so that ImageNet-pretrained weights can be loaded directly,
as described in Section 3 ("ResNet Encoder Backbone Architecture") of the paper.

The encoder exposes the multi-scale feature pyramid

    {F0, F1, F2, F3, F4}

where ``F0`` is the stem feature *before* max-pooling (stride-2, 64 channels,
H/2 resolution) and ``F1..F4`` are the outputs of the four residual stages
(``layer1``..``layer4``). ``F0..F3`` are forwarded to the decoder as skip
connections and ``F4`` is the bottleneck.
"""
from __future__ import annotations

import torch.nn as nn
import torchvision


# name -> (constructor, torchvision weights enum, [stem, l1, l2, l3, l4] channels)
_RESNET_FACTORY = {
    "resnet18": (
        torchvision.models.resnet18,
        torchvision.models.ResNet18_Weights.IMAGENET1K_V1,
        [64, 64, 128, 256, 512],
    ),
    "resnet34": (
        torchvision.models.resnet34,
        torchvision.models.ResNet34_Weights.IMAGENET1K_V1,
        [64, 64, 128, 256, 512],
    ),
    "resnet50": (
        torchvision.models.resnet50,
        torchvision.models.ResNet50_Weights.IMAGENET1K_V2,
        [64, 256, 512, 1024, 2048],
    ),
    "resnet101": (
        torchvision.models.resnet101,
        torchvision.models.ResNet101_Weights.IMAGENET1K_V2,
        [64, 256, 512, 1024, 2048],
    ),
}


class ResNetEncoder(nn.Module):
    """torchvision ResNet wrapped as a multi-scale feature extractor."""

    def __init__(self, name: str = "resnet50", pretrained: bool = True):
        super().__init__()
        if name not in _RESNET_FACTORY:
            raise ValueError(
                f"Unknown encoder '{name}'. Choose from {list(_RESNET_FACTORY)}."
            )
        ctor, weights, channels = _RESNET_FACTORY[name]
        net = ctor(weights=weights if pretrained else None)

        # Stem: 7x7 conv (stride 2) + BN + ReLU -> H/2, 64 channels.
        self.stem = nn.Sequential(net.conv1, net.bn1, net.relu)
        self.maxpool = net.maxpool  # -> H/4
        self.layer1 = net.layer1    # -> H/4
        self.layer2 = net.layer2    # -> H/8
        self.layer3 = net.layer3    # -> H/16
        self.layer4 = net.layer4    # -> H/32 (bottleneck)

        self.name = name
        self.pretrained = pretrained
        # Channels of [F0, F1, F2, F3, F4].
        self.out_channels = list(channels)

    def forward(self, x):
        f0 = self.stem(x)             # H/2, 64
        f1 = self.layer1(self.maxpool(f0))  # H/4
        f2 = self.layer2(f1)          # H/8
        f3 = self.layer3(f2)          # H/16
        f4 = self.layer4(f3)          # H/32
        return [f0, f1, f2, f3, f4]
