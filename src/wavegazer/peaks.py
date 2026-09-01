"""Peak / NMS / 7 µm match — the detect scoreboard, not dense Dice.

Biohub / PIXEL_FIRST: bipartite node match at 7 µm. This module is that
gate only (no linking, no adj_edge_jaccard yet).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from .fsot_seeds import SEEDS


@dataclass
class PeakSet:
    """xy in pixel coordinates (x=col, y=row), scores high-is-better."""

    xy: torch.Tensor  # (P, 2) float, columns (x, y)
    score: torch.Tensor  # (P,)


def local_maxima(
    field: torch.Tensor,
    *,
    window: int,
    min_score: torch.Tensor | float,
) -> PeakSet:
    """field: (1,1,H,W) or (H,W). window odd. min_score broadcastable."""
    if field.dim() == 2:
        field = field[None, None]
    elif field.dim() == 3:
        field = field[None]
    if window % 2 == 0:
        window += 1
    pad = window // 2
    pooled = F.max_pool2d(field, kernel_size=window, stride=1, padding=pad)
    is_max = field >= pooled - 1e-12
    if not torch.is_tensor(min_score):
        min_score = field.new_tensor(float(min_score))
    while min_score.dim() < field.dim():
        min_score = min_score.unsqueeze(-1)
    keep = is_max & (field > min_score)
    # suppress exact-tie plateaus: keep one pixel per connected run via unique rows
    ys, xs = torch.where(keep[0, 0])
    scores = field[0, 0, ys, xs]
    xy = torch.stack([xs.float(), ys.float()], dim=1)
    return PeakSet(xy=xy, score=scores)


def nms(peaks: PeakSet, radius_px: float) -> PeakSet:
    """Greedy score-order suppression. radius in pixels."""
    if peaks.xy.numel() == 0:
        return peaks
    order = torch.argsort(peaks.score, descending=True)
    xy = peaks.xy[order]
    sc = peaks.score[order]
    keep: list[int] = []
    r2 = radius_px * radius_px
    for i in range(xy.size(0)):
        if any(((xy[i] - xy[j]) ** 2).sum() <= r2 for j in keep):
            continue
        keep.append(i)
    idx = torch.tensor(keep, device=xy.device, dtype=torch.long)
    return PeakSet(xy=xy[idx], score=sc[idx])


def match_xy(
    pred_xy: torch.Tensor,
    gt_xy: torch.Tensor,
    max_dist_px: float,
) -> dict[str, float]:
    """Greedy bipartite match. pred/gt (N,2) as (x,y) pixels."""
    n_gt = int(gt_xy.size(0))
    n_pr = int(pred_xy.size(0))
    if n_gt == 0:
        return {
            "n_gt": 0,
            "n_pred": n_pr,
            "tp": 0,
            "recall": 1.0 if n_pr == 0 else 0.0,
            "precision": 1.0 if n_pr == 0 else 0.0,
            "f1": 1.0 if n_pr == 0 else 0.0,
        }
    if n_pr == 0:
        return {
            "n_gt": n_gt,
            "n_pred": 0,
            "tp": 0,
            "recall": 0.0,
            "precision": 0.0,
            "f1": 0.0,
        }
    # pairwise distances
    d = torch.cdist(pred_xy.float(), gt_xy.float(), p=2)
    used_p = torch.zeros(n_pr, dtype=torch.bool, device=d.device)
    used_g = torch.zeros(n_gt, dtype=torch.bool, device=d.device)
    tp = 0
    # smallest-distance first among pairs under threshold
    flat = d.reshape(-1)
    order = torch.argsort(flat)
    w_gt = n_gt
    for idx in order.tolist():
        if flat[idx].item() > max_dist_px:
            break
        p, g = divmod(int(idx), w_gt)
        if used_p[p] or used_g[g]:
            continue
        used_p[p] = True
        used_g[g] = True
        tp += 1
        if tp == min(n_pr, n_gt):
            break
    rec = tp / n_gt
    prec = tp / n_pr
    f1 = 0.0 if rec + prec == 0 else 2 * rec * prec / (rec + prec)
    return {
        "n_gt": n_gt,
        "n_pred": n_pr,
        "tp": tp,
        "recall": rec,
        "precision": prec,
        "f1": f1,
    }


def sparse_gate(field: torch.Tensor, *, k: float | None = None) -> torch.Tensor:
    """mean + k·std. Default k=φ (dense-sparse masks). Detect uses √φ."""
    if k is None:
        k = SEEDS.phi
    mu = field.mean(dim=(-2, -1), keepdim=True)
    sd = field.std(dim=(-2, -1), keepdim=True).clamp_min(1e-6)
    return mu + k * sd


def detect_gate(field: torch.Tensor) -> torch.Tensor:
    """Looser seed tail for detect: mean + √φ · std. Still zero free params."""
    return sparse_gate(field, k=SEEDS.phi ** 0.5)
