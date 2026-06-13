"""UNet segmentation network with a ResNet encoder (Section 3 of the paper).

Assembles :class:`ResNetEncoder` and :class:`UNetDecoder` into the three-class
plant-health segmentation model. Decoder channel widths follow the paper:

* ResNet-18/34 (basic block): (512,256,256) (256,128,128) (128,64,64)
  (64,64,32) (32,0,16) -> decoder output widths [256,128,64,32,16].
* ResNet-50/101 (bottleneck): (2048,1024,256) (256,512,128) (128,256,64)
  (64,64,32) (32,0,16) -> decoder output widths [256,128,64,32,16].

Both share the same decoder output widths because the per-block input widths
are determined by the encoder; only the skip-connection channel counts differ,
which is handled automatically from ``encoder.out_channels``.
"""
from __future__ import annotations

import torch.nn as nn

from .encoder import ResNetEncoder
from .decoder import UNetDecoder

DEFAULT_DECODER_CHANNELS = (256, 128, 64, 32, 16)


class UNetResNet(nn.Module):
    def __init__(
        self,
        encoder: str = "resnet50",
        num_classes: int = 3,
        pretrained: bool = True,
        use_skip: bool = True,
        decoder_convs: int = 2,
        num_decoder_blocks: int = 5,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.encoder = ResNetEncoder(encoder, pretrained=pretrained)
        self.decoder = UNetDecoder(
            encoder_channels=self.encoder.out_channels,
            num_classes=num_classes,
            decoder_channels=DEFAULT_DECODER_CHANNELS,
            use_skip=use_skip,
            num_convs=decoder_convs,
            num_blocks=num_decoder_blocks,
            dropout=dropout,
        )
        self.config = dict(
            encoder=encoder,
            num_classes=num_classes,
            pretrained=pretrained,
            use_skip=use_skip,
            decoder_convs=decoder_convs,
            num_decoder_blocks=num_decoder_blocks,
            dropout=dropout,
        )

    def forward(self, x):
        out_size = x.shape[-2:]
        features = self.encoder(x)
        return self.decoder(features, out_size)


def build_segmentation_model(cfg: dict) -> UNetResNet:
    """Build the segmentation model from a config dict (see configs/default.yaml)."""
    m = cfg.get("model", {})
    return UNetResNet(
        encoder=m.get("encoder", "resnet50"),
        num_classes=m.get("num_classes", 3),
        pretrained=m.get("pretrained", True),
        use_skip=m.get("use_skip", True),
        decoder_convs=m.get("decoder_convs", 2),
        num_decoder_blocks=m.get("num_decoder_blocks", 5),
        dropout=m.get("dropout", 0.1),
    )
