"""Seed-derived blob gaze. This is the detect head the field actually scores.

U-Net Biohub detectors emit peaks, then centroids. A fitted LoG/DoG is the
classical non-learned version of that head. Here the two Gaussians are
σ and φσ, and σ is the competition match length in pixels:

    σ_px = MATCH_UM / (µm_per_px · π)

Bright-on-dark centers are G(σ) − G(φσ). No fitted kernel widths.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from .fsot_seeds import SEEDS

MATCH_UM = 7.0
DEFAULT_YX_UM = 0.40625  # Biohub dump Y/X scale on 44b6_* volumes


def sigma_px(um_per_px: float = DEFAULT_YX_UM, match_um: float = MATCH_UM) -> float:
    return match_um / (um_per_px * SEEDS.pi)


def _gauss_kernel_1d(sigma: float, device, dtype) -> torch.Tensor:
    radius = max(int(math.ceil(3.0 * sigma)), 1)
    x = torch.arange(-radius, radius + 1, device=device, dtype=dtype)
    k = torch.exp(-0.5 * (x / sigma) ** 2)
    return (k / k.sum()).view(1, 1, -1)


def gaussian_blur(x: torch.Tensor, sigma: float) -> torch.Tensor:
    """Separable Gaussian, NCHW, replicate pad. σ in pixels."""
    if sigma < 0.15:
        return x
    ky = _gauss_kernel_1d(sigma, x.device, x.dtype)
    kx = ky.view(1, 1, 1, -1)
    ky = ky.view(1, 1, -1, 1)
    py, px = ky.size(-2) // 2, kx.size(-1) // 2
    y = F.pad(x, (px, px, py, py), mode="replicate")
    y = F.conv2d(y, ky)
    y = F.conv2d(y, kx)
    return y


def phi_dog(luma: torch.Tensor, sigma: float) -> torch.Tensor:
    """G(σ) − G(φσ). Positive at bright blob centers of width ~σ."""
    small = gaussian_blur(luma, sigma)
    large = gaussian_blur(luma, sigma * SEEDS.phi)
    return small - large


def blob_map(luma: torch.Tensor, sigma: float) -> torch.Tensor:
    """Non-negative blob field, 0–1 per image, for S or for peak finding."""
    dog = F.relu(phi_dog(luma, sigma))
    lo = dog.amin(dim=(-2, -1), keepdim=True)
    hi = dog.amax(dim=(-2, -1), keepdim=True).clamp_min(lo + 1e-6)
    return (dog - lo) / (hi - lo)


def multi_scale_blob_map(luma: torch.Tensor, sigma: float) -> torch.Tensor:
    """Max over {σ/φ, σ, φσ}. Catches cells a bit smaller or larger than σ."""
    scales = (sigma / SEEDS.phi, sigma, sigma * SEEDS.phi)
    acc = None
    for s in scales:
        b = blob_map(luma, max(float(s), 0.5))
        acc = b if acc is None else torch.maximum(acc, b)
    return acc
