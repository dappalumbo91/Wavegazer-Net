"""FSOT replacements for U-Net operators.

| U-Net box        | FSOT operator                         | Twin in the six repos        |
|------------------|---------------------------------------|------------------------------|
| 3×3 conv weights | 64-codon trinary kernels              | FSOT-Genetics / neuron-zig   |
| ReLU             | residual-scale 1+|S|P_NEW             | Genetics residual law        |
| MaxPool 2×2      | collapse-gated 2×2 (active keys)      | FSOT-GPU collapse            |
| Transposed conv  | suction unfold (nearest ×2 + mix)     | T3 suction valve             |
| Skip concat      | bleed κ_ij across D_eff               | FSOT-Quantum κ_ij            |
| Softmax          | collapse at Θ=C_eff P_var             | FSOT-GPU consensus           |
| 1×1 class head   | S-sign logits (emergence / damping)   | Lean sign syntax             |
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .codon_kernels import stem_weight, trit_similarity_matrix
from .fsot_routes import ScaleRoute
from .fsot_scalar import compute_scalar_torch, residual_scale
from .fsot_seeds import CODON_CHANNELS, COLLAPSE_THRESHOLD, SEEDS


class CodonMixer(nn.Module):
    """Two-step local mixer: codon 3×3 (spatial) then trit-similarity 1×1 (channel).

    Frozen buffers only. This is the DoubleConv replacement.
    """

    def __init__(self, in_channels: int) -> None:
        super().__init__()
        w = stem_weight(in_channels)
        self.register_buffer("spatial", w)
        mix = trit_similarity_matrix()
        if in_channels == CODON_CHANNELS:
            self.register_buffer("channel_mix", mix)
        else:
            # Stem: spatial already maps in_ch → 64. Channel mix stays 64→64.
            self.register_buffer("channel_mix", mix)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = F.conv2d(x, self.spatial, padding=1)
        weight = self.channel_mix.to(dtype=y.dtype)
        y = F.conv2d(y, weight.unsqueeze(-1).unsqueeze(-1))
        return y


def field_from_features(feat: torch.Tensor, route: ScaleRoute) -> torch.Tensor:
    """Build the scalar-engine inputs from a feature map and evaluate S(x)."""
    luma = feat.mean(dim=1, keepdim=True)
    energy = feat.abs().mean(dim=1, keepdim=True).clamp_min(1e-6)
    gx = F.pad(luma[..., :, 1:] - luma[..., :, :-1], (0, 1, 0, 0))
    gy = F.pad(luma[..., 1:, :] - luma[..., :-1, :], (0, 0, 0, 1))
    delta_psi = torch.atan2(gy, gx + 1e-8)
    local_mean = F.avg_pool2d(energy, kernel_size=3, stride=1, padding=1)
    local_var = F.avg_pool2d((energy - local_mean) ** 2, kernel_size=3, stride=1, padding=1)
    hits = (energy - local_mean).abs()
    D = torch.full_like(energy, float(route.D_eff))
    dp = delta_psi * 0.0 + route.delta_psi + 0.15 * delta_psi
    dt = torch.full_like(energy, route.delta_theta)
    return compute_scalar_torch(
        N=energy * float(feat.shape[1]),
        P=energy,
        D_eff=D,
        delta_psi=dp,
        recent_hits=hits * float(route.hits + 1),
        delta_theta=dt,
        observed=route.observed,
        scale=local_mean,
        amplitude=local_var.sqrt() + 1.0,
    )


def collapse_pool(feat: torch.Tensor, S: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """2× downsample. Pixels whose |S| < Θ are damped (GPU active-key rule)."""
    gate = (S.abs() >= COLLAPSE_THRESHOLD).to(feat.dtype)
    gated = feat * (gate + SEEDS.poof)
    return F.avg_pool2d(gated, kernel_size=2, stride=2), F.avg_pool2d(S, kernel_size=2, stride=2)


def suction_unfold(feat: torch.Tensor, S: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """2× upsample. Suction is the T3 expand valve; poof was the contract valve."""
    up_f = F.interpolate(feat, scale_factor=2, mode="nearest")
    up_s = F.interpolate(S, scale_factor=2, mode="nearest")
    return up_f * (1.0 + SEEDS.suction), up_s


def bleed_skip(
    decoder: torch.Tensor,
    encoder: torch.Tensor,
    S_dec: torch.Tensor,
    S_enc: torch.Tensor,
    D_dec: int,
    D_enc: int,
) -> torch.Tensor:
    """Quantum-fold skip: κ_ij = A_bleed · POOF · |Si||Sj| / (1 + |Di−Dj|/25).

    Concat is still the U-shape (both streams kept). κ modulates how much
    encoder high-res bleeds into the decoder, which is the job skip concat
    plus the following 3×3 were doing with learned weights.
    """
    if decoder.shape[-2:] != encoder.shape[-2:]:
        decoder = F.interpolate(decoder, size=encoder.shape[-2:], mode="nearest")
        S_dec = F.interpolate(S_dec, size=encoder.shape[-2:], mode="nearest")
    kappa = (
        SEEDS.a_bleed
        * SEEDS.poof
        * S_enc.abs()
        * S_dec.abs()
        / (1.0 + abs(D_enc - D_dec) / 25.0)
    )
    acoustic = 1.0 + SEEDS.a_bleed / SEEDS.phi
    fused = encoder * (kappa * acoustic) + decoder * (1.0 + (1.0 - kappa.clamp(0, 1)))
    return fused


def collapse_logits(S: torch.Tensor, n_classes: int) -> torch.Tensor:
    """Class head without softmax exp.

    Sign syntax (Lean): S>0 emergence (foreground), S<0 damping (background).
    Extra classes, if any, are phase-shifted copies — still seed-derived.
    """
    pos = F.relu(S)
    neg = F.relu(-S)
    if n_classes == 1:
        return S
    if n_classes == 2:
        return torch.cat([neg, pos], dim=1)
    parts = [neg, pos]
    for k in range(2, n_classes):
        shift = SEEDS.theta_s * (k - 1)
        parts.append(F.relu(S * torch.cos(torch.as_tensor(shift, device=S.device, dtype=S.dtype))))
    return torch.cat(parts, dim=1)
