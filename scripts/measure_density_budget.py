"""Recall after keeping top T_true/n_t peaks (density target, not a fitted k)."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import torch
import zarr

from wavegazer.blob import MATCH_UM
from wavegazer.peaks import match_xyz_um
from wavegazer.wavegazer_net import WavegazerNet

BIOHUB = Path(r"D:\Kaggle_Biohub_Data\train")
YX, ZU = 0.40625, 1.625
N = 16


def main() -> None:
    net = WavegazerNet(1, 2, sparse=True)
    zarrs = sorted(p for p in BIOHUB.iterdir() if p.name.endswith(".zarr"))[:N]
    print("name rec_full rec_budget n_pred budget n_gt t_true")
    recs_f, recs_b = [], []
    for zp in zarrs:
        gp = zp.with_suffix(".geff")
        g = zarr.open(gp, mode="r")
        props = g["nodes"]["props"]
        nd = {k: np.asarray(props[k]["values"]) for k in ("x", "y", "t", "z")}
        extra = dict(g.attrs.get("geff", {})).get("extra") or {}
        t_true = float(extra.get("estimated_number_of_nodes") or float("nan"))
        arr = zarr.open(zp, mode="r")["0"]
        vals, counts = np.unique(nd["t"], return_counts=True)
        t0 = int(vals[int(np.argmax(counts))])
        sel = nd["t"] == t0
        vol = np.asarray(arr[t0]).astype(np.float32)
        lo, hi = np.quantile(vol, [0.01, 0.99])
        vol = np.clip((vol - lo) / max(float(hi - lo), 1e-6), 0.0, 1.0)
        gt = torch.tensor(
            np.stack([nd["x"][sel], nd["y"][sel], nd["z"][sel]], 1),
            dtype=torch.float32,
        )
        with torch.no_grad():
            pred, sc = net.detect_volume(torch.from_numpy(vol), yx_um=YX, z_um=ZU)
        m = match_xyz_um(pred, gt, MATCH_UM, yx_um=YX, z_um=ZU)
        budget = max(1, int(round(t_true / arr.shape[0])))
        if pred.size(0) > budget:
            top = torch.argsort(sc, descending=True)[:budget]
            pred_b = pred[top]
        else:
            pred_b = pred
        mb = match_xyz_um(pred_b, gt, MATCH_UM, yx_um=YX, z_um=ZU)
        recs_f.append(m["recall"])
        recs_b.append(mb["recall"])
        print(
            zp.stem,
            f"{m['recall']:.3f}",
            f"{mb['recall']:.3f}",
            int(pred.size(0)),
            budget,
            int(gt.size(0)),
            f"{t_true:.0f}",
        )
    print("mean full", sum(recs_f) / len(recs_f), "mean budget", sum(recs_b) / len(recs_b))


if __name__ == "__main__":
    main()
