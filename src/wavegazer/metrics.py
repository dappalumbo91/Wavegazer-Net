"""Segmentation metrics used as the observable scoreboard.

- Dice / F1 on the foreground (ISBI cell tracking, most medical papers)
- Intersection over Union / Jaccard (paper Table 2 reports IOU)
- Pixel error (paper Table 1, EM challenge)
- Warping error and Rand error are dataset-specific ISBI EM metrics and
  are not computed here; published numbers are recorded in the outcomes doc.
"""

from __future__ import annotations

import torch


def _binary_counts(
    pred: torch.Tensor,
    target: torch.Tensor,
    class_index: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    p = pred == class_index
    t = target == class_index
    tp = (p & t).sum().float()
    fp = (p & ~t).sum().float()
    fn = (~p & t).sum().float()
    return tp, fp, fn


def dice_coefficient(
    logits_or_pred: torch.Tensor,
    target: torch.Tensor,
    *,
    class_index: int = 1,
    from_logits: bool = True,
    eps: float = 1e-6,
) -> torch.Tensor:
    """2 TP / (2 TP + FP + FN) for one class. Perfect overlap = 1."""
    pred = logits_or_pred.argmax(1) if from_logits else logits_or_pred
    tp, fp, fn = _binary_counts(pred, target, class_index)
    return (2 * tp + eps) / (2 * tp + fp + fn + eps)


def intersection_over_union(
    logits_or_pred: torch.Tensor,
    target: torch.Tensor,
    *,
    class_index: int = 1,
    from_logits: bool = True,
    eps: float = 1e-6,
) -> torch.Tensor:
    """TP / (TP + FP + FN). Paper Table 2 uses this (called IOU)."""
    pred = logits_or_pred.argmax(1) if from_logits else logits_or_pred
    tp, fp, fn = _binary_counts(pred, target, class_index)
    return (tp + eps) / (tp + fp + fn + eps)


def pixel_error(
    logits_or_pred: torch.Tensor,
    target: torch.Tensor,
    *,
    from_logits: bool = True,
) -> torch.Tensor:
    """Fraction of pixels whose argmax class disagrees with the label."""
    pred = logits_or_pred.argmax(1) if from_logits else logits_or_pred
    return (pred != target).float().mean()
