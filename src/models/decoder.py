"""UNet decoder and segmentation head.

Implements the decoder pathway described in Section 3 ("Decoder Architecture
and Segmentation Head"). Each decoder block performs nearest-neighbour
upsampling (factor 2), optional channel-wise concatenation with the encoder
skip connection, and one or two Conv(3x3)-BN-ReLU sub-blocks. Five blocks are
stacked by default, progressively restoring the full input resolution; the
final segmentation head is Dropout(p) followed by a 3x3 convolution.

The block count, the use of skip connections, the number of convolutions per
block and the dropout rate are all configurable so that the ablations reported
in Table 5 of the paper can be reproduced from a single implementation.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _conv_bn_relu(in_ch: int, out_ch: int) -> nn.Sequential:
    # Bias is disabled because the following BN provides the learnable shift.
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    )


class DecoderBlock(nn.Module):
    def __init__(
        self,
        in_ch: int,
        skip_ch: int,
        out_ch: int,
        use_skip: bool = True,
        num_convs: int = 2,
    ):
        super().__init__()
        self.use_skip = use_skip and skip_ch > 0
        first_in = in_ch + (skip_ch if self.use_skip else 0)

        layers = [_conv_bn_relu(first_in, out_ch)]
        for _ in range(max(0, num_convs - 1)):
            layers.append(_conv_bn_relu(out_ch, out_ch))
        self.convs = nn.Sequential(*layers)

    def forward(self, x, skip=None):
        x = F.interpolate(x, scale_factor=2.0, mode="nearest")
        if self.use_skip and skip is not None:
            x = torch.cat([x, skip], dim=1)
        return self.convs(x)


class UNetDecoder(nn.Module):
    """Cascaded decoder + lightweight segmentation head.

    Parameters
    ----------
    encoder_channels : list[int]
        ``[F0, F1, F2, F3, F4]`` channel counts from the encoder.
    num_classes : int
        Number of output semantic classes (3 in this study).
    decoder_channels : list[int]
        Output channels for each decoder block (default ``[256,128,64,32,16]``).
    use_skip : bool
        If ``False`` all skip connections are removed ("w/o skip" ablation).
    num_convs : int
        Convolutions per decoder block (1 reproduces the "single conv" ablation).
    num_blocks : int
        Number of decoder blocks; 3 reproduces the "3 decoder blocks" ablation.
    dropout : float
        Dropout probability in the segmentation head.
    """

    def __init__(
        self,
        encoder_channels,
        num_classes: int = 3,
        decoder_channels=(256, 128, 64, 32, 16),
        use_skip: bool = True,
        num_convs: int = 2,
        num_blocks: int = 5,
        dropout: float = 0.1,
    ):
        super().__init__()
        decoder_channels = list(decoder_channels)[:num_blocks]

        # Skip features fed to blocks 1..4 come from F3, F2, F1, F0; the last
        # block (full resolution) has no skip.
        skip_channels = [
            encoder_channels[3],  # F3
            encoder_channels[2],  # F2
            encoder_channels[1],  # F1
            encoder_channels[0],  # F0 (stem)
            0,                    # last block: no skip
        ]

        blocks = []
        in_ch = encoder_channels[4]  # bottleneck F4
        for i in range(num_blocks):
            out_ch = decoder_channels[i]
            skip_ch = skip_channels[i] if i < len(skip_channels) else 0
            blocks.append(
                DecoderBlock(in_ch, skip_ch, out_ch, use_skip=use_skip, num_convs=num_convs)
            )
            in_ch = out_ch
        self.blocks = nn.ModuleList(blocks)
        self.num_blocks = num_blocks

        self.head = nn.Sequential(
            nn.Dropout2d(p=dropout),
            nn.Conv2d(in_ch, num_classes, kernel_size=3, padding=1),
        )

        # Skip features consumed by blocks 1..4 (reverse order, excl. bottleneck).
        self._skip_indices = [3, 2, 1, 0]
        self._init_weights()

    def _init_weights(self):
        for m in self.blocks.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight, mode="fan_in", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
        # Segmentation head uses Xavier uniform initialisation.
        nn.init.xavier_uniform_(self.head[1].weight)
        if self.head[1].bias is not None:
            nn.init.zeros_(self.head[1].bias)

    def forward(self, features, out_size):
        x = features[4]  # bottleneck
        for i, block in enumerate(self.blocks):
            skip = None
            if i < len(self._skip_indices):
                skip = features[self._skip_indices[i]]
            x = block(x, skip)

        logits = self.head(x)
        if logits.shape[-2:] != tuple(out_size):
            logits = F.interpolate(logits, size=out_size, mode="nearest")
        return logits
