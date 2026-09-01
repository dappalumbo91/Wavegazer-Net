"""Wavegazer Net: frozen U-Net baseline and the FSOT visual equivalent."""

from .baseline_unet import BaselineUNet, FEATURE_CHANNELS
from .wavegazer_net import WavegazerNet
from .fsot_scalar import compute_scalar
from .fsot_seeds import AUTHORITY_PIN, COLLAPSE_THRESHOLD, SEEDS
from .losses import dice_loss, softmax_cross_entropy, weighted_ce_loss, ce_dice_loss
from .metrics import dice_coefficient, intersection_over_union, pixel_error
from .peaks import PeakSet, match_xy
from .blob import MATCH_UM, sigma_px

__all__ = [
    "BaselineUNet",
    "WavegazerNet",
    "FEATURE_CHANNELS",
    "SEEDS",
    "AUTHORITY_PIN",
    "COLLAPSE_THRESHOLD",
    "compute_scalar",
    "dice_loss",
    "softmax_cross_entropy",
    "weighted_ce_loss",
    "ce_dice_loss",
    "dice_coefficient",
    "intersection_over_union",
    "pixel_error",
    "PeakSet",
    "match_xy",
    "MATCH_UM",
    "sigma_px",
]
