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

from .blob import DEFAULT_YX_UM, MATCH_UM, multi_scale_blob_map, sigma_px, sigma_um
from .fsot_routes import VISUAL_FOREGROUND_SIGN, VISUAL_LADDER
from .fsot_seeds import CODON_CHANNELS, SEEDS
from .operators import CodonMixer, collapse_logits, field_from_features
from .peaks import PeakSet, detect_gate, local_maxima, nms, nms_xyz_um


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

    def detect_volume(
        self,
        vol: torch.Tensor,
        *,
        yx_um: float = DEFAULT_YX_UM,
        z_um: float = 1.625,
        match_um: float = MATCH_UM,
        nms_um: float | None = None,
        max_peaks_per_plane: int = 256,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """3D centroids. vol is (Z,Y,X) or (1,Z,Y,X).

        Default NMS is σ (same as 2D detect): two peaks inside the 7 µm match
        ball can both survive. Pass nms_um=match_um for one centroid per cell
        (better density, can merge a dim GT into a brighter neighbor).
        """
        if vol.dim() == 4:
            vol = vol[0]
        if vol.dim() != 3:
            raise ValueError(f"detect_volume expects (Z,Y,X), got {tuple(vol.shape)}")
        z_n = int(vol.size(0))
        sig = sigma_px(yx_um, match_um)
        window = max(int(2 * sig) | 1, 3)
        xs, ys, zs, sc = [], [], [], []
        for z in range(z_n):
            img = vol[z][None, None]
            blobs, s_map = self.blob_field(img, um_per_px=yx_um, match_um=match_um)
            peaks = local_maxima(blobs, window=window, min_score=detect_gate(blobs))
            if peaks.xy.size(0) == 0:
                continue
            xi = peaks.xy[:, 0].long().clamp(0, blobs.size(-1) - 1)
            yi = peaks.xy[:, 1].long().clamp(0, blobs.size(-2) - 1)
            b_at = blobs[0, 0, yi, xi]
            s_at = s_map[0, 0, yi, xi]
            s_n = (s_at - s_at.min()) / (s_at.max().clamp_min(s_at.min() + 1e-6) - s_at.min() + 1e-8)
            score = b_at + SEEDS.p_new * s_n
            if score.numel() > max_peaks_per_plane:
                top = torch.argsort(score, descending=True)[:max_peaks_per_plane]
                peaks = PeakSet(xy=peaks.xy[top], score=score[top])
                score = peaks.score
            xs.append(peaks.xy[:, 0])
            ys.append(peaks.xy[:, 1])
            zs.append(torch.full((peaks.xy.size(0),), float(z), device=vol.device))
            sc.append(score)
        if not xs:
            return vol.new_zeros(0, 3), vol.new_zeros(0)
        xyz = torch.stack([torch.cat(xs), torch.cat(ys), torch.cat(zs)], dim=1)
        score = torch.cat(sc)
        radius = sigma_um(match_um) if nms_um is None else nms_um
        return nms_xyz_um(xyz, score, radius, yx_um=yx_um, z_um=z_um)

    def codon_codes(self, vol: torch.Tensor, xyz: torch.Tensor) -> torch.Tensor:
        """(P, 64) raw codon 3×3 responses at peaks. Genetics local “what”.

        Spatial stem only — trit mix is the *compare* (see ``codon_ident``),
        not a second mix on the vector.
        """
        if vol.dim() == 4:
            vol = vol[0]
        spatial = self.stem.spatial
        if xyz.numel() == 0:
            return spatial.new_zeros(0, CODON_CHANNELS)
        vol = vol.to(dtype=spatial.dtype)
        z_n, h, w = int(vol.size(0)), int(vol.size(1)), int(vol.size(2))
        zi = xyz[:, 2].long().clamp(0, z_n - 1)
        codes = spatial.new_zeros(xyz.size(0), CODON_CHANNELS)
        for z in zi.unique().tolist():
            sel = zi == z
            img = vol[int(z)][None, None]
            feat = F.conv2d(img, spatial, padding=1)
            # Cell-scale pool: window 2φ+1 (odd). 3×3 codon is local; this is the blob.
            k = int(2 * SEEDS.phi + 1) | 1
            feat = F.avg_pool2d(feat, kernel_size=k, stride=1, padding=k // 2)[0]
            xs = xyz[sel, 0].long().clamp(0, w - 1)
            ys = xyz[sel, 1].long().clamp(0, h - 1)
            codes[sel] = feat[:, ys, xs].transpose(0, 1).contiguous()
        return codes

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
