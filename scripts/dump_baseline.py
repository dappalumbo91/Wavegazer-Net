"""Dump the frozen U-Net baseline: shapes, param counts, device, dummy metrics.

Run from the project root after the venv is installed:

    .\\.venv\\Scripts\\python.exe scripts\\dump_baseline.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import torch

from wavegazer.baseline_unet import (
    PAPER_INPUT_SIZE,
    PAPER_OUTPUT_SIZE,
    BaselineUNet,
    feature_shapes,
)
from wavegazer.losses import ce_dice_loss
from wavegazer.metrics import dice_coefficient, intersection_over_union, pixel_error


def _device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _record_model(mode: str, spatial: int, device: torch.device) -> dict:
    net = BaselineUNet(in_channels=1, n_classes=2, mode=mode).to(device)
    net.eval()
    x = torch.randn(1, 1, spatial, spatial, device=device)
    with torch.no_grad():
        y = net(x)
        dummy_target = torch.zeros(1, y.shape[-2], y.shape[-1], dtype=torch.long, device=device)
        dummy_target[:, y.shape[-2] // 4 : 3 * y.shape[-2] // 4, y.shape[-1] // 4 : 3 * y.shape[-1] // 4] = 1
        loss = ce_dice_loss(y, dummy_target)
        metrics = {
            "dummy_dice": float(dice_coefficient(y, dummy_target).cpu()),
            "dummy_iou": float(intersection_over_union(y, dummy_target).cpu()),
            "dummy_pixel_error": float(pixel_error(y, dummy_target).cpu()),
            "dummy_ce_dice_loss": float(loss.cpu()),
        }
    return {
        "mode": mode,
        "input_shape": [1, 1, spatial, spatial],
        "output_shape": list(y.shape),
        "parameters": net.count_parameters(),
        "encoder_feature_shapes": [
            {"channels": c, "height": h, "width": w} for c, h, w in feature_shapes(spatial, mode)
        ],
        **metrics,
    }


def main() -> None:
    device = _device()
    info = {
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "device": str(device),
    }
    if torch.cuda.is_available():
        info["gpu_name"] = torch.cuda.get_device_name(0)
        info["gpu_capability"] = ".".join(str(i) for i in torch.cuda.get_device_capability(0))
        x = torch.randn(256, 256, device="cuda")
        info["matmul_ok"] = bool(torch.isfinite(x @ x).all().item())

    modern = _record_model("modern", 256, device)
    paper = _record_model("paper", PAPER_INPUT_SIZE, device)
    assert paper["output_shape"][-1] == PAPER_OUTPUT_SIZE

    payload = {
        "environment": info,
        "modern": modern,
        "paper": paper,
        "published_outcomes_to_match_later": {
            "isbi_em_warping_error": 0.0003529,
            "isbi_em_rand_error": 0.0382,
            "isbi_em_pixel_error": 0.0611,
            "phc_u373_iou": 0.9203,
            "dic_hela_iou": 0.7756,
            "source": "Ronneberger, Fischer, Brox, MICCAI 2015, arXiv:1505.04597",
        },
    }
    out = ROOT / "artifacts" / "baseline_freeze.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
