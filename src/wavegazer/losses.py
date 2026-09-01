"""Losses used by the original U-Net and by the modern training standard.

Paper (Ronneberger 2015)
    Pixel-wise softmax + weighted cross-entropy. The weight map puts extra
    mass on the thin background borders that separate touching cells.

Modern default (nnU-Net / V-Net lineage)
    Soft Dice + unweighted cross-entropy, summed. Dice handles class
    imbalance without a hand-crafted weight map.

These are the functions Wavegazer Net is scored against.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def softmax_cross_entropy(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Mean unweighted pixel-wise cross-entropy.

    logits: (N, C, H, W)
    target: (N, H, W) integer class ids
    """
    return F.cross_entropy(logits, target)


def weighted_ce_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    weight_map: torch.Tensor,
) -> torch.Tensor:
    """Paper equation (1): E = sum_x w(x) log(p_{ell(x)}(x)).

    Implemented as a mean so the scale is comparable across tile sizes.
    weight_map: (N, H, W) or (N, 1, H, W)
    """
    if weight_map.dim() == 4:
        weight_map = weight_map.squeeze(1)
    per_pixel = F.cross_entropy(logits, target, reduction="none")
    return (per_pixel * weight_map).mean()


def border_weight_map(
    target: torch.Tensor,
    *,
    w0: float = 10.0,
    sigma: float = 5.0,
    class_weight: torch.Tensor | None = None,
) -> torch.Tensor:
    """Paper equation (2) for a binary instance-style mask.

    w(x) = w_c(x) + w0 * exp(-(d1(x) + d2(x))^2 / (2 sigma^2))

    This implementation is a practical stand-in: it boosts pixels whose
    3x3 neighborhood contains both foreground and background, which is
    the morphological separation border the paper computes with distance
    transforms. A full d1/d2 distance-transform version can replace this
    once instance masks (not just semantic masks) are in the data loader.

    target: (N, H, W) with integer labels, 0 = background.
    """
    fg = (target > 0).float()
    kernel = torch.ones(1, 1, 3, 3, device=target.device, dtype=fg.dtype)
    neighborhood = F.conv2d(fg.unsqueeze(1), kernel, padding=1)
    # Pixels whose 3x3 window is mixed foreground/background: the thin border.
    border = ((neighborhood > 0) & (neighborhood < 9)).float().squeeze(1)
    wc = torch.ones_like(border) if class_weight is None else class_weight[target]
    # Full paper form uses d1+d2 distance-to-two-nearest-cells. On a 1-pixel
    # morphological border, d1+d2 is ~0 so the exponential is ~1.
    _ = sigma  # kept in the signature to match the paper's (w0, sigma) pair
    return wc + w0 * border


def dice_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    *,
    smooth: float = 1.0,
    from_logits: bool = True,
) -> torch.Tensor:
    """Soft Dice loss, 1 - 2|A∩B| / (|A| + |B|), averaged over classes.

    Background class is included. For nnU-Net-style foreground-only Dice
    pass logits[:, 1:] and a one-hot target without channel 0.
    """
    if from_logits:
        probs = torch.softmax(logits, dim=1)
    else:
        probs = logits
    n_classes = probs.shape[1]
    target_oh = F.one_hot(target.long(), num_classes=n_classes).permute(0, 3, 1, 2).float()
    dims = (0, 2, 3)
    intersection = (probs * target_oh).sum(dim=dims)
    denom = probs.sum(dim=dims) + target_oh.sum(dim=dims)
    dice = (2.0 * intersection + smooth) / (denom + smooth)
    return 1.0 - dice.mean()


def ce_dice_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    *,
    ce_weight: float = 1.0,
    dice_weight: float = 1.0,
) -> torch.Tensor:
    """nnU-Net default: L = L_Dice + L_CE."""
    return ce_weight * softmax_cross_entropy(logits, target) + dice_weight * dice_loss(logits, target)
