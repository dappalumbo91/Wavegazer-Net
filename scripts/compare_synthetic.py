"""Named synthetic cell split: train BaselineUNet, run WavegazerNet once.

This is the first scoreboard that is not noise. Blobs are generated from a
fixed seed so the split is reproducible. Wavegazer has nothing to train.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from wavegazer.baseline_unet import BaselineUNet
from wavegazer.losses import ce_dice_loss
from wavegazer.metrics import dice_coefficient, intersection_over_union, pixel_error
from wavegazer.wavegazer_net import WavegazerNet

SIZE = 64
N_TRAIN = 64
N_VAL = 16
EPOCHS = 12
BATCH = 8
SEED = 20260831


def _disks(n: int, size: int, rng: torch.Generator) -> tuple[torch.Tensor, torch.Tensor]:
    """Soft-blob images and hard disk masks. Microscopy-shaped, not ImageNet."""
    yy, xx = torch.meshgrid(
        torch.linspace(-1, 1, size),
        torch.linspace(-1, 1, size),
        indexing="ij",
    )
    images = torch.zeros(n, 1, size, size)
    masks = torch.zeros(n, size, size, dtype=torch.long)
    for i in range(n):
        n_cells = int(torch.randint(2, 6, (1,), generator=rng).item())
        canvas = torch.zeros(size, size)
        mask = torch.zeros(size, size)
        for _ in range(n_cells):
            cy = float(torch.empty(1).uniform_(-0.65, 0.65, generator=rng))
            cx = float(torch.empty(1).uniform_(-0.65, 0.65, generator=rng))
            r = float(torch.empty(1).uniform_(0.12, 0.28, generator=rng))
            amp = float(torch.empty(1).uniform_(0.6, 1.0, generator=rng))
            dist2 = (yy - cy) ** 2 + (xx - cx) ** 2
            blob = amp * torch.exp(-dist2 / (2 * (r * 0.7) ** 2))
            canvas = torch.maximum(canvas, blob)
            mask = torch.maximum(mask, (dist2 <= r * r).float())
        noise = 0.05 * torch.randn(size, size, generator=rng)
        images[i, 0] = (canvas + noise).clamp(0, 1)
        masks[i] = mask.long()
    return images, masks


def _eval(net: torch.nn.Module, images: torch.Tensor, masks: torch.Tensor, device: torch.device) -> dict:
    net.eval()
    dices, ious, pix = [], [], []
    with torch.no_grad():
        for i in range(0, images.size(0), BATCH):
            x = images[i : i + BATCH].to(device)
            y = masks[i : i + BATCH].to(device)
            logits = net(x)
            dices.append(float(dice_coefficient(logits, y).cpu()))
            ious.append(float(intersection_over_union(logits, y).cpu()))
            pix.append(float(pixel_error(logits, y).cpu()))
    return {
        "dice": sum(dices) / len(dices),
        "iou": sum(ious) / len(ious),
        "pixel_error": sum(pix) / len(pix),
    }


def _threshold_baseline(images: torch.Tensor, masks: torch.Tensor) -> dict:
    """Intensity > mean. A formula-free control so all-foreground is visible."""
    pred = (images[:, 0] > images[:, 0].mean(dim=(1, 2), keepdim=True)).long()
    logits = F.one_hot(pred, num_classes=2).permute(0, 3, 1, 2).float()
    return {
        "dice": float(dice_coefficient(logits, masks, from_logits=True)),
        "iou": float(intersection_over_union(logits, masks, from_logits=True)),
        "pixel_error": float(pixel_error(logits, masks, from_logits=True)),
    }


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rng = torch.Generator().manual_seed(SEED)
    train_x, train_y = _disks(N_TRAIN, SIZE, rng)
    val_x, val_y = _disks(N_VAL, SIZE, rng)

    unet = BaselineUNet(in_channels=1, n_classes=2, mode="modern").to(device)
    opt = torch.optim.Adam(unet.parameters(), lr=1e-3)
    loader = DataLoader(TensorDataset(train_x, train_y), batch_size=BATCH, shuffle=True)
    history = []
    unet.train()
    for epoch in range(EPOCHS):
        losses = []
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad(set_to_none=True)
            logits = unet(xb)
            loss = ce_dice_loss(logits, yb)
            loss.backward()
            opt.step()
            losses.append(float(loss.detach().cpu()))
        history.append({"epoch": epoch, "train_ce_dice": sum(losses) / len(losses)})

    wave = WavegazerNet(in_channels=1, n_classes=2).to(device)
    payload = {
        "split": {
            "name": "synthetic_cells_v0",
            "seed": SEED,
            "size": SIZE,
            "n_train": N_TRAIN,
            "n_val": N_VAL,
            "epochs_unet": EPOCHS,
        },
        "device": str(device),
        "unet_train_history": history,
        "val": {
            "intensity_threshold": _threshold_baseline(val_x, val_y),
            "unet_trained": _eval(unet, val_x, val_y, device),
            "wavegazer_closed_form": _eval(wave, val_x, val_y, device),
        },
        "note": (
            "Named split, not noise. WavegazerNet is not trained. "
            "If Wavegazer residual is bad, change fsot_routes.py, not conv weights."
        ),
    }
    out = ROOT / "artifacts" / "synthetic_cells_compare.json"
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
