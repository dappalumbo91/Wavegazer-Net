"""Serialize frozen Wavegazer buffers for Hugging Face / Kaggle."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import torch

from wavegazer.wavegazer_net import WavegazerNet


def main() -> None:
    net = WavegazerNet(in_channels=1, n_classes=2, sparse=True)
    payload = {
        "authority_pin": "D1D38A",
        "trainable": net.count_parameters(),
        "state": {k: v.cpu() for k, v in net.state_dict().items()},
    }
    out = ROOT / "artifacts" / "wavegazer_buffers.pt"
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, out)
    print(f"trainable={payload['trainable']} wrote {out} bytes={out.stat().st_size}")


if __name__ == "__main__":
    main()
