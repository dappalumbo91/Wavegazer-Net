"""Preregistered D_eff ladder for the visual U.

Each U-Net resolution is a dimensional interface, not a learned width.
If a visual residual is bad, change this table — do not add a weight.

Why these domains (from FSOT-2.1-Lean domain table):

- Optics (D=10)            finest pixels: light / edges
- Condensed_Matter (D=14)  local material / texture
- Fluid_Dynamics (D=15)    tissue deformation, the skip-bleed native fold
- Seismology (D=18)        bulk chaotic context (unobserved)
- Cosmology (D=25)         compactification ceiling = bottleneck
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ScaleRoute:
    name: str
    D_eff: int
    hits: int
    delta_psi: float
    delta_theta: float
    observed: bool


# Encoder order: fine → coarse. Decoder walks this list backwards.
VISUAL_LADDER: tuple[ScaleRoute, ...] = (
    ScaleRoute("Optics", 10, 0, 0.6, 1.0, True),
    ScaleRoute("Condensed_Matter", 14, 0, 0.5, 1.0, True),
    ScaleRoute("Fluid_Dynamics", 15, 1, 0.9, 1.0, False),
    ScaleRoute("Seismology", 18, 2, 1.2, 1.0, False),
    ScaleRoute("Cosmology", 25, 0, 1.0, 1.0, False),
)

# Local class polarity at the collapse head. +1 means S above the tile
# median is foreground. Image-level S correlates with bright-on-dark
# structure (corr ~0.95 on a test square). This is a preregistered route
# bit, like observed, not a fitted weight.
VISUAL_FOREGROUND_SIGN = 1
