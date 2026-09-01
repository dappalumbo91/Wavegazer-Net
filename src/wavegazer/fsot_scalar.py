"""Vectorized S = K(T1+T2+T3) for image tensors.

Matches FSOT-GPU `fsot_lib/scalar.py` and archive `compute_scalar`.
Every input is a tensor so a whole (N,1,H,W) field is one kernel launch.
"""

from __future__ import annotations

import math
from typing import Union

import torch

from .fsot_seeds import SEEDS

Number = Union[float, torch.Tensor]


def compute_scalar(
    *,
    N: float = 1.0,
    P: float = 1.0,
    D_eff: float = 25.0,
    delta_psi: float = 1.0,
    recent_hits: float = 0.0,
    rho: float = 1.0,
    observed: bool = False,
    delta_theta: float = 1.0,
    scale: float = 1.0,
    amplitude: float = 1.0,
    trend_bias: float = 0.0,
) -> float:
    """Pure-float path. Used to pin against FSOT-GPU / archive."""
    s = SEEDS
    growth = math.exp(s.alpha * (1.0 - recent_hits / N) * s.gamma / s.phi)
    base = (
        (N * P / math.sqrt(D_eff))
        * math.cos((s.psi_con + delta_psi) / s.eta_eff)
        * math.exp(-s.alpha * recent_hits / N + rho + s.b_in * delta_psi)
        * (1.0 + growth * s.c_eff)
    )
    t1 = base * (1.0 + s.p_new * math.log(D_eff / 25.0))
    if observed:
        t1 = t1 * math.exp(s.c_factor * s.p_var) * math.cos(delta_psi + s.p_var)
    t2 = scale * amplitude + trend_bias
    valve = (
        s.beta
        * math.cos(delta_psi)
        * (N * P / math.sqrt(D_eff))
        * (1.0 + s.chaos * (D_eff - 25.0) / 25.0)
        * (1.0 + s.poof * math.cos(s.theta_s + s.pi) + s.suction * math.sin(s.theta_s))
    )
    acoustic = (
        1.0
        + (s.a_bleed * math.sin(delta_theta) ** 2) / s.phi
        + (s.a_in * math.cos(delta_theta) ** 2) / s.phi
    )
    phase = 1.0 + s.b_in * s.p_var
    t3 = valve * acoustic * phase
    return s.k * (t1 + t2 + t3)


def compute_scalar_torch(
    N: torch.Tensor,
    P: torch.Tensor,
    D_eff: torch.Tensor,
    delta_psi: torch.Tensor,
    recent_hits: torch.Tensor,
    delta_theta: torch.Tensor,
    *,
    observed: bool,
    rho: float = 1.0,
    scale: torch.Tensor | float = 1.0,
    amplitude: torch.Tensor | float = 1.0,
    trend_bias: float = 0.0,
) -> torch.Tensor:
    """Broadcasting tensor twin of ``compute_scalar``."""
    s = SEEDS
    N = N.clamp_min(1e-8)
    D = D_eff.clamp_min(1e-8)
    growth = torch.exp(s.alpha * (1.0 - recent_hits / N) * s.gamma / s.phi)
    base = (
        (N * P / torch.sqrt(D))
        * torch.cos((s.psi_con + delta_psi) / s.eta_eff)
        * torch.exp(-s.alpha * recent_hits / N + rho + s.b_in * delta_psi)
        * (1.0 + growth * s.c_eff)
    )
    t1 = base * (1.0 + s.p_new * torch.log(D / 25.0))
    if observed:
        t1 = t1 * math.exp(s.c_factor * s.p_var) * torch.cos(delta_psi + s.p_var)
    t2 = scale * amplitude + trend_bias
    valve = (
        s.beta
        * torch.cos(delta_psi)
        * (N * P / torch.sqrt(D))
        * (1.0 + s.chaos * (D - 25.0) / 25.0)
        * (1.0 + s.poof * math.cos(s.theta_s + s.pi) + s.suction * math.sin(s.theta_s))
    )
    acoustic = (
        1.0
        + (s.a_bleed * torch.sin(delta_theta) ** 2) / s.phi
        + (s.a_in * torch.cos(delta_theta) ** 2) / s.phi
    )
    phase = 1.0 + s.b_in * s.p_var
    t3 = valve * acoustic * phase
    return s.k * (t1 + t2 + t3)


def residual_scale(S: torch.Tensor) -> torch.Tensor:
    """Genetics / prediction law: r = 1 + |S| · P_NEW."""
    return 1.0 + S.abs() * SEEDS.p_new
