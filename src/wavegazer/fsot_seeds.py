"""FSOT Layer-0/1/2 constants. Zero free parameters.

Numeric twins of:
- FSOT-2.1-Lean `vendor/fsot_compute.py` (authority pin D1D38A)
- FSOT-GPU `fsot_lib/seeds.py`

Do not add fitted values here. If a visual residual is bad, change the
preregistered D_eff route in `fsot_routes.py`, not these numbers.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class Seeds:
    pi: float = math.pi
    e: float = math.e
    phi: float = (1.0 + math.sqrt(5.0)) / 2.0
    gamma: float = 0.5772156649015329
    g_catalan: float = 0.9159655941772190

    alpha: float = 8.082937414140405e-4
    psi_con: float = 0.6321205588285577
    eta_eff: float = 0.46694220692425986
    beta: float = 2.620866911333223e-17
    chaos: float = -0.3310241826104818
    theta_s: float = 0.29089654054517305
    poof: float = 0.1534822148944508

    c_eff: float = 0.9577022026205613
    p_var: float = 0.9579871226722757
    b_in: float = 0.7879407922764435
    a_in: float = 1.6668538450045732
    a_bleed: float = 1.046973630587551
    suction: float = 0.14703398542810284
    p_new: float = 0.30030227667037146
    c_factor: float = 0.287600151819184
    k: float = 0.42022166416069665

    @property
    def collapse_threshold(self) -> float:
        """Θ = C_eff · P_var. GPU/Ada collapse gate. Not a softmax temperature."""
        return self.c_eff * self.p_var


SEEDS = Seeds()
COLLAPSE_THRESHOLD = SEEDS.collapse_threshold
AUTHORITY_PIN = "D1D38A"
CODON_CHANNELS = 64
COMPACTIFICATION_CEILING = 25.0
