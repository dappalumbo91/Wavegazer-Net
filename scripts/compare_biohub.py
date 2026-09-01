"""Score Wavegazer on local Biohub zarr+geff (the ~175 GB competition dump).

Dense masks are disks around GT nodes on a 2D max-Z projection at a busy time.
That is a localization proxy, not the official adj_edge_jaccard score.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from wavegazer.baseline_unet import BaselineUNet
from wavegazer.losses import ce_dice_loss
from wavegazer.metrics import dice_coefficient, intersection_over_union, pixel_error
from wavegazer.wavegazer_net import WavegazerNet

BIOHUB = Path(r"D:\Kaggle_Biohub_Data\train")
RADIUS_PX = 8  # ~3.25 µm at 0.40625 µm/px; competition match is 7 µm
N_VOLUMES = 24
SEED = 20260831
SIZE = 256
EPOCHS = 8
BATCH = 4


def _node_xy_t(geff_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    import zarr

    g = zarr.open(geff_path, mode="r")
    props = g["nodes"]["props"]
    x = np.asarray(props["x"]["values"])
    y = np.asarray(props["y"]["values"])
    t = np.asarray(props["t"]["values"])
    return x, y, t


def _frame_and_mask(zarr_path: Path, geff_path: Path) -> tuple[torch.Tensor, torch.Tensor] | None:
    import zarr

    x, y, t = _node_xy_t(geff_path)
    if len(t) == 0:
        return None
    # Busiest time index.
    values, counts = np.unique(t, return_counts=True)
    t0 = int(values[int(np.argmax(counts))])
    sel = t == t0
    xs, ys = x[sel], y[sel]
    arr = zarr.open(zarr_path, mode="r")["0"]
    # (T, Z, Y, X)
    if t0 < 0 or t0 >= arr.shape[0]:
        return None
    plane = np.asarray(arr[t0]).max(axis=0).astype(np.float32)  # Z-max
    lo, hi = np.quantile(plane, [0.01, 0.99])
    plane = np.clip((plane - lo) / max(hi - lo, 1e-6), 0, 1)
    h, w = plane.shape
    yy, xx = np.mgrid[0:h, 0:w]
    mask = np.zeros((h, w), dtype=np.float32)
    for cx, cy in zip(xs, ys):
        mask += ((yy - float(cy)) ** 2 + (xx - float(cx)) ** 2 <= RADIUS_PX**2).astype(np.float32)
    mask = np.clip(mask, 0, 1)
    if mask.sum() < 8:
        return None
    img = torch.from_numpy(plane.astype(np.float32))[None]
    m = torch.from_numpy(mask).long()
    if img.shape[-2:] != (SIZE, SIZE):
        img = F.interpolate(img[None], size=(SIZE, SIZE), mode="bilinear", align_corners=False)[0]
        m = F.interpolate(m[None, None].float(), size=(SIZE, SIZE), mode="nearest")[0, 0].long()
    return img, m


def _collect(n: int) -> tuple[torch.Tensor, torch.Tensor, list[str]]:
    zarrs = sorted(p for p in BIOHUB.iterdir() if p.name.endswith(".zarr"))
    images, masks, names = [], [], []
    for zp in zarrs:
        if len(images) >= n:
            break
        gp = zp.with_suffix(".geff")
        if not gp.is_dir() and not gp.exists():
            continue
        try:
            pair = _frame_and_mask(zp, gp)
        except Exception as exc:
            print(f"skip {zp.name}: {exc}")
            continue
        if pair is None:
            continue
        images.append(pair[0])
        masks.append(pair[1])
        names.append(zp.stem)
    if not images:
        raise RuntimeError(f"no Biohub frames loaded from {BIOHUB}")
    return torch.stack(images), torch.stack(masks), names


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
    pred = (images[:, 0] > images[:, 0].mean(dim=(1, 2), keepdim=True)).long()
    logits = F.one_hot(pred, num_classes=2).permute(0, 3, 1, 2).float()
    return {
        "dice": float(dice_coefficient(logits, masks, from_logits=True)),
        "iou": float(intersection_over_union(logits, masks, from_logits=True)),
        "pixel_error": float(pixel_error(logits, masks, from_logits=True)),
    }


def main() -> None:
    if not BIOHUB.is_dir():
        raise SystemExit(f"Biohub dump not found: {BIOHUB}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    images, masks, names = _collect(N_VOLUMES)
    g = torch.Generator().manual_seed(SEED)
    perm = torch.randperm(images.size(0), generator=g)
    n_val = max(4, images.size(0) // 4)
    val_idx, train_idx = perm[:n_val], perm[n_val:]
    train_x, train_y = images[train_idx], masks[train_idx]
    val_x, val_y = images[val_idx], masks[val_idx]

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
            loss = ce_dice_loss(unet(xb), yb)
            loss.backward()
            opt.step()
            losses.append(float(loss.detach().cpu()))
        history.append({"epoch": epoch, "train_ce_dice": sum(losses) / len(losses)})

    wave = WavegazerNet(in_channels=1, n_classes=2, sparse=True).to(device)
    payload = {
        "split": {
            "name": "biohub_zmax_disk_v0",
            "root": str(BIOHUB),
            "volumes": names,
            "n_train": int(train_x.size(0)),
            "n_val": int(val_x.size(0)),
            "size": SIZE,
            "radius_px": RADIUS_PX,
            "epochs_unet": EPOCHS,
            "seed": SEED,
        },
        "device": str(device),
        "unet_train_history": history,
        "val": {
            "intensity_threshold": _threshold_baseline(val_x, val_y),
            "unet_trained": _eval(unet, val_x, val_y, device),
            "wavegazer_closed_form": _eval(wave, val_x, val_y, device),
        },
        "note": (
            "Local Biohub dump. Disks around GT nodes on Z-max at the busiest t. "
            "Not official adj_edge_jaccard. Wavegazer is not trained."
        ),
    }
    out = ROOT / "artifacts" / "biohub_compare.json"
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
