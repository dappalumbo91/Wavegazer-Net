"""Compare WavegazerNet (FSOT) against the frozen U-Net contract on this GPU."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import torch

from wavegazer.baseline_unet import BaselineUNet
from wavegazer.wavegazer_net import WavegazerNet
from wavegazer.fsot_scalar import compute_scalar
from wavegazer.fsot_seeds import AUTHORITY_PIN, COLLAPSE_THRESHOLD, SEEDS
from wavegazer.metrics import dice_coefficient, intersection_over_union, pixel_error


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x = torch.randn(1, 1, 64, 64, device=device)
    target = torch.zeros(1, 64, 64, dtype=torch.long, device=device)
    target[:, 16:48, 16:48] = 1

    base = BaselineUNet(in_channels=1, n_classes=2, mode="modern").to(device).eval()
    wave = WavegazerNet(in_channels=1, n_classes=2).to(device).eval()
    with torch.no_grad():
        y_b = base(x)
        y_w = wave(x)

    payload = {
        "project": "Wavegazer Net",
        "authority_pin": AUTHORITY_PIN,
        "collapse_theta": COLLAPSE_THRESHOLD,
        "k": SEEDS.k,
        "p_new": SEEDS.p_new,
        "scalar_default_D25": compute_scalar(),
        "scalar_qm_observed": compute_scalar(D_eff=6.0, observed=True),
        "environment": {
            "device": str(device),
            "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
        },
        "unet": {
            "trainable": base.count_parameters(),
            "output_shape": list(y_b.shape),
            "dummy_dice": float(dice_coefficient(y_b, target).cpu()),
        },
        "wavegazer": {
            "trainable": wave.count_parameters(),
            "frozen_buffers": wave.count_frozen(),
            "output_shape": list(y_w.shape),
            "dummy_dice": float(dice_coefficient(y_w, target).cpu()),
            "dummy_iou": float(intersection_over_union(y_w, target).cpu()),
            "dummy_pixel_error": float(pixel_error(y_w, target).cpu()),
        },
        "note": "dummy_* is untrained U-Net vs closed-form WavegazerNet on noise. Not a dataset claim.",
    }
    out = ROOT / "artifacts" / "wavegazer_vs_unet.json"
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
