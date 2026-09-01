"""Canonical U-Net used as the observable baseline for Wavegazer Net.

Two modes, both 23 convolutional layers and four skip levels:

- ``paper``: Ronneberger et al. 2015. Valid (unpadded) 3x3 convolutions,
  ReLU, 2x2 max-pool, 2x2 transposed conv, cropped skip concatenation,
  He/Kaiming initialization. Input 572x572 yields output 388x388.
- ``modern``: same topology with same-padding so input spatial size equals
  output spatial size. Optional batch-norm. This is what nearly every
  open-source PyTorch U-Net actually ships (milesial/Pytorch-UNet,
  nnU-Net PlainConvUNet).

Neither mode is the project formula. They exist so a later equivalent
network can be compared against a frozen, published architecture.
"""

from __future__ import annotations

from typing import List, Literal, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

Mode = Literal["paper", "modern"]

# Encoder channel schedule from the original figure: 64, 128, 256, 512, 1024.
FEATURE_CHANNELS: Tuple[int, ...] = (64, 128, 256, 512, 1024)

# Paper figure 1: 572 input, four 2x2 pools, valid 3x3 convs -> 388 output.
PAPER_INPUT_SIZE = 572
PAPER_OUTPUT_SIZE = 388


def _he_init(module: nn.Module) -> None:
    """He/Kaiming normal init as specified in the U-Net paper (sqrt(2/N))."""
    if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
        nn.init.kaiming_normal_(module.weight, mode="fan_in", nonlinearity="relu")
        if module.bias is not None:
            nn.init.zeros_(module.bias)


class DoubleConv(nn.Module):
    """Two 3x3 convolutions, each followed by ReLU (and optional BatchNorm)."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        padding: int,
        batch_norm: bool,
        bias: bool,
    ) -> None:
        super().__init__()
        layers: List[nn.Module] = [
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=padding, bias=bias),
        ]
        if batch_norm:
            layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.ReLU(inplace=True))
        layers.append(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=padding, bias=bias)
        )
        if batch_norm:
            layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.ReLU(inplace=True))
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class BaselineUNet(nn.Module):
    """U-Net with four down/up stages and skip concatenation.

    Parameters
    ----------
    in_channels:
        Input image channels (1 for the original EM/microscopy setup, 3 for RGB).
    n_classes:
        Number of output class logits. The paper used 2 for membrane vs cell.
    mode:
        ``paper`` or ``modern``. See module docstring.
    bilinear:
        Modern-mode only. If True, upsample with bilinear interpolation plus a
        1x1 channel projection instead of a learned 2x2 transposed convolution.
        The paper uses transposed convolution (bilinear=False).
    """

    def __init__(
        self,
        in_channels: int = 1,
        n_classes: int = 2,
        mode: Mode = "modern",
        bilinear: bool = False,
    ) -> None:
        super().__init__()
        if mode not in ("paper", "modern"):
            raise ValueError(f"mode must be 'paper' or 'modern', got {mode!r}")
        self.in_channels = in_channels
        self.n_classes = n_classes
        self.mode = mode
        self.bilinear = bilinear and mode == "modern"

        padding = 0 if mode == "paper" else 1
        batch_norm = mode == "modern"
        # Paper Caffe convs had bias. Modern BN variants drop bias.
        bias = not batch_norm
        ch = FEATURE_CHANNELS

        def conv(ic: int, oc: int) -> DoubleConv:
            return DoubleConv(ic, oc, padding=padding, batch_norm=batch_norm, bias=bias)

        self.inc = conv(in_channels, ch[0])
        self.down_convs = nn.ModuleList(
            [conv(ch[i], ch[i + 1]) for i in range(len(ch) - 1)]
        )
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.up_ops = nn.ModuleList()
        self.up_convs = nn.ModuleList()
        for i in range(len(ch) - 1, 0, -1):
            in_up, out_up = ch[i], ch[i - 1]
            if self.bilinear:
                self.up_ops.append(
                    nn.Sequential(
                        nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
                        nn.Conv2d(in_up, out_up, kernel_size=1, bias=bias),
                    )
                )
                self.up_convs.append(conv(out_up * 2, out_up))
            else:
                self.up_ops.append(
                    nn.ConvTranspose2d(in_up, out_up, kernel_size=2, stride=2, bias=bias)
                )
                self.up_convs.append(conv(out_up * 2, out_up))

        self.outc = nn.Conv2d(ch[0], n_classes, kernel_size=1)
        self.apply(_he_init)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skips: List[torch.Tensor] = []
        h = self.inc(x)
        skips.append(h)
        for down in self.down_convs:
            h = down(self.pool(h))
            skips.append(h)

        h = skips.pop()  # bottleneck
        for up, conv in zip(self.up_ops, self.up_convs):
            h = up(h)
            skip = skips.pop()
            if self.mode == "paper":
                skip = _center_crop(skip, h.shape[-2:])
            else:
                h = _pad_to(h, skip.shape[-2:])
            h = conv(torch.cat([skip, h], dim=1))
        return self.outc(h)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def _pad_to(tensor: torch.Tensor, size: Sequence[int]) -> torch.Tensor:
    """Pad tensor spatially so it matches ``size`` (H, W). Used in modern mode."""
    th, tw = size
    h, w = tensor.shape[-2:]
    if (h, w) == (th, tw):
        return tensor
    diff_y = th - h
    diff_x = tw - w
    if diff_y < 0 or diff_x < 0:
        raise RuntimeError(f"Cannot pad {(h, w)} up to smaller {(th, tw)}")
    return F.pad(
        tensor,
        [diff_x // 2, diff_x - diff_x // 2, diff_y // 2, diff_y - diff_y // 2],
    )


def _center_crop(tensor: torch.Tensor, size: Sequence[int]) -> torch.Tensor:
    th, tw = size
    h, w = tensor.shape[-2:]
    if h < th or w < tw:
        raise RuntimeError(
            f"Cannot crop skip of size {(h, w)} to {(th, tw)}. "
            "Paper-mode input spatial size must be even at every pool, typically 572."
        )
    y0 = (h - th) // 2
    x0 = (w - tw) // 2
    return tensor[..., y0 : y0 + th, x0 : x0 + tw]


def feature_shapes(spatial: int, mode: Mode) -> List[Tuple[int, int, int]]:
    """Return (channels, height, width) after each encoder DoubleConv, including bottleneck."""
    h = w = spatial
    shapes: List[Tuple[int, int, int]] = []
    shrink = 0 if mode == "modern" else 4  # two valid 3x3 convs
    for i, c in enumerate(FEATURE_CHANNELS):
        if i > 0:
            h //= 2
            w //= 2
        h -= shrink
        w -= shrink
        shapes.append((c, h, w))
    return shapes
