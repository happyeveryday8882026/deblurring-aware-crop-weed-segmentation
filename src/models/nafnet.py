"""NAFNet deblurring front-end (Section 3, "NAFNet Deblurring Module").

A compact, self-contained re-implementation of the Nonlinear Activation Free
Network (Chen et al., ECCV 2022, https://github.com/megvii-research/NAFNet).
Each NAF block replaces conventional activations with a SimpleGate operator and
a Simplified Channel Attention (SCA) module, as described by Eqs. (12)-(13) of
the manuscript.

The default width/block configuration approximates the ~17.1 M-parameter /
2.78 GMACs budget reported in Table 8; the exact figures for a given config can
be recomputed with ``src/complexity.py``.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class LayerNorm2d(nn.Module):
    """Channel-wise LayerNorm for NCHW tensors."""

    def __init__(self, channels: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(channels))
        self.bias = nn.Parameter(torch.zeros(channels))
        self.eps = eps

    def forward(self, x):
        mu = x.mean(dim=1, keepdim=True)
        var = (x - mu).pow(2).mean(dim=1, keepdim=True)
        x = (x - mu) / torch.sqrt(var + self.eps)
        return x * self.weight[None, :, None, None] + self.bias[None, :, None, None]


class SimpleGate(nn.Module):
    """Split along channels into two halves and multiply element-wise."""

    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2


class NAFBlock(nn.Module):
    def __init__(self, c: int, dw_expand: int = 2, ffn_expand: int = 2):
        super().__init__()
        dw_channel = c * dw_expand
        self.norm1 = LayerNorm2d(c)
        self.conv1 = nn.Conv2d(c, dw_channel, 1)
        self.conv2 = nn.Conv2d(
            dw_channel, dw_channel, 3, padding=1, groups=dw_channel
        )
        self.sg = SimpleGate()
        # Simplified Channel Attention on the gated feature (dw_channel // 2).
        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dw_channel // 2, dw_channel // 2, 1),
        )
        self.conv3 = nn.Conv2d(dw_channel // 2, c, 1)

        ffn_channel = c * ffn_expand
        self.norm2 = LayerNorm2d(c)
        self.conv4 = nn.Conv2d(c, ffn_channel, 1)
        self.conv5 = nn.Conv2d(ffn_channel // 2, c, 1)

        self.beta = nn.Parameter(torch.zeros(1, c, 1, 1))
        self.gamma = nn.Parameter(torch.zeros(1, c, 1, 1))

    def forward(self, inp):
        x = self.conv1(self.norm1(inp))
        x = self.conv2(x)
        x = self.sg(x)
        x = x * self.sca(x)
        x = self.conv3(x)
        y = inp + x * self.beta

        x = self.conv4(self.norm2(y))
        x = self.sg(x)
        x = self.conv5(x)
        return y + x * self.gamma


class NAFNet(nn.Module):
    """Four-level hierarchical encoder-decoder built from NAF blocks."""

    def __init__(
        self,
        img_channel: int = 3,
        width: int = 36,
        enc_blk_nums=(1, 1, 2, 8),
        middle_blk_num: int = 4,
        dec_blk_nums=(1, 1, 1, 1),
    ):
        super().__init__()
        self.intro = nn.Conv2d(img_channel, width, 3, padding=1)
        self.ending = nn.Conv2d(width, img_channel, 3, padding=1)

        self.encoders = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.ups = nn.ModuleList()

        chan = width
        for n in enc_blk_nums:
            self.encoders.append(nn.Sequential(*[NAFBlock(chan) for _ in range(n)]))
            self.downs.append(nn.Conv2d(chan, 2 * chan, 2, stride=2))
            chan *= 2

        self.middle_blks = nn.Sequential(*[NAFBlock(chan) for _ in range(middle_blk_num)])

        for n in dec_blk_nums:
            self.ups.append(
                nn.Sequential(
                    nn.Conv2d(chan, chan * 2, 1, bias=False),
                    nn.PixelShuffle(2),
                )
            )
            chan //= 2
            self.decoders.append(nn.Sequential(*[NAFBlock(chan) for _ in range(n)]))

        self.padder_size = 2 ** len(enc_blk_nums)

    def _check_pad(self, x):
        _, _, h, w = x.size()
        mod = self.padder_size
        ph = (mod - h % mod) % mod
        pw = (mod - w % mod) % mod
        if ph or pw:
            x = nn.functional.pad(x, (0, pw, 0, ph))
        return x, h, w

    def forward(self, inp):
        x, h, w = self._check_pad(inp)
        inp_p = x

        x = self.intro(x)
        skips = []
        for enc, down in zip(self.encoders, self.downs):
            x = enc(x)
            skips.append(x)
            x = down(x)

        x = self.middle_blks(x)

        for dec, up, skip in zip(self.decoders, self.ups, skips[::-1]):
            x = up(x)
            x = x + skip
            x = dec(x)

        x = self.ending(x)
        x = x + inp_p
        return x[:, :, :h, :w]


def build_deblur_model(cfg: dict) -> NAFNet:
    n = cfg.get("nafnet", {})
    return NAFNet(
        img_channel=3,
        width=n.get("width", 36),
        enc_blk_nums=tuple(n.get("enc_blk_nums", [1, 1, 2, 8])),
        middle_blk_num=n.get("middle_blk_num", 4),
        dec_blk_nums=tuple(n.get("dec_blk_nums", [1, 1, 1, 1])),
    )
