"""WavegazerNet — FSOT visual equivalent of the canonical U-Net.

Same in/out contract as BaselineUNet modern mode:
    (N, C, H, W) → (N, K, H, W) with H, W preserved.

Gaze is L0 image S (Optics fold on luma). Codon kernels stay as a frozen
texture stem but do not enter logits until a named split shows they help.
We do **not** multiply features by 1+|S|P_NEW when S is a per-pixel field
of thousands — that residual law is for a domain scalar of order one.

This is not the parked PyTorch 'FSOT neuron v1' that still used Linear +
softmax. That path is documented as free-param opposite of the seed spine.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .blob import DEFAULT_YX_UM, MATCH_UM, multi_scale_blob_map, sigma_px
from .fsot_routes import VISUAL_FOREGROUND_SIGN, VISUAL_LADDER
from .fsot_seeds import SEEDS
from .operators import CodonMixer, collapse_logits, field_from_features
from .peaks import PeakSet, detect_gate, local_maxima, nms


def _luma(x: torch.Tensor) -> torch.Tensor:
    if x.size(1) == 1:
        return x
    # Rec. 601 luma, broadcast over extra channels if C>3.
    w = x.new_tensor([0.299, 0.587, 0.114]).view(1, 3, 1, 1)
    c = min(3, x.size(1))
    return (x[:, :c] * w[:, :c]).sum(dim=1, keepdim=True)


class WavegazerNet(nn.Module):
    def __init__(self, in_channels: int = 1, n_classes: int = 2, *, sparse: bool = False) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.n_classes = n_classes
        self.sparse = sparse
        self.ladder = VISUAL_LADDER
        # Frozen codon filters: texture branch only. Not the gaze backbone.
        self.stem = CodonMixer(in_channels)

    def gaze_field(self, x: torch.Tensor) -> torch.Tensor:
        """Optics-fold S on luma. Dense Dice / U-Net contract lives here."""
        self.stem(x)
        return field_from_features(_luma(x), self.ladder[0])

    def blob_field(
        self,
        x: torch.Tensor,
        *,
        um_per_px: float = DEFAULT_YX_UM,
        match_um: float = MATCH_UM,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """DoG blob map and Optics S on that map. Peaks live on the DoG."""
        luma = _luma(x)
        blobs = multi_scale_blob_map(luma, sigma_px(um_per_px, match_um))
        s_map = field_from_features(blobs, self.ladder[0])
        return blobs, s_map

    def detect(
        self,
        x: torch.Tensor,
        *,
        um_per_px: float = DEFAULT_YX_UM,
        match_um: float = MATCH_UM,
        max_peaks: int = 256,
    ) -> PeakSet:
        """Centroids for the 7 µm scoreboard. Zero trainable weights.

        Peak *locations* come from φ-DoG (same as the intensity control) so
        S cannot drop a DoG local max. S only re-ranks for NMS.
        """
        if x.size(0) != 1:
            raise ValueError("detect() is per-image; batch later")
        blobs, s_map = self.blob_field(x, um_per_px=um_per_px, match_um=match_um)
        sig = sigma_px(um_per_px, match_um)
        window = max(int(2 * sig) | 1, 3)
        peaks = local_maxima(blobs, window=window, min_score=detect_gate(blobs))
        if peaks.xy.size(0) == 0:
            return peaks
        xs = peaks.xy[:, 0].long().clamp(0, blobs.size(-1) - 1)
        ys = peaks.xy[:, 1].long().clamp(0, blobs.size(-2) - 1)
        b_at = blobs[0, 0, ys, xs]
        s_at = s_map[0, 0, ys, xs]
        s_lo = s_at.min()
        s_hi = s_at.max().clamp_min(s_lo + 1e-6)
        s_n = (s_at - s_lo) / (s_hi - s_lo)
        peaks = PeakSet(xy=peaks.xy, score=b_at + SEEDS.p_new * s_n)
        if peaks.xy.size(0) > max_peaks:
            top = torch.argsort(peaks.score, descending=True)[:max_peaks]
            peaks = PeakSet(xy=peaks.xy[top], score=peaks.score[top])
        return nms(peaks, radius_px=sig)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.size(1) != self.in_channels:
            raise ValueError(f"expected {self.in_channels} channels, got {x.size(1)}")
        S_out = self.gaze_field(x)
        if self.sparse:
            # Biohub-style: cells are rare. Median would call half the tile fg.
            # Gate at mean + φ·std — seed-derived tail, not a fitted threshold.
            mu = S_out.mean(dim=(-2, -1), keepdim=True)
            sd = S_out.std(dim=(-2, -1), keepdim=True).clamp_min(1e-6)
            loc = mu + SEEDS.phi * sd
        else:
            flat = S_out.flatten(-2)
            loc = flat.median(dim=-1).values.view(*S_out.shape[:2], 1, 1)
        S_rel = VISUAL_FOREGROUND_SIGN * (S_out - loc)
        logits = collapse_logits(S_rel, self.n_classes)
        if logits.shape[-2:] != x.shape[-2:]:
            logits = F.interpolate(logits, size=x.shape[-2:], mode="bilinear", align_corners=False)
        return logits

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def count_frozen(self) -> int:
        return sum(b.numel() for b in self.buffers())
