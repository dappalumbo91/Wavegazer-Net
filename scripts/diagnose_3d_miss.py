"""Why 44b6_2f31fc2f drops one of three GT nodes in 3D 7 µm match."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import torch
import zarr

from wavegazer.blob import MATCH_UM, sigma_px
from wavegazer.peaks import PeakSet, detect_gate, dist_um, local_maxima
from wavegazer.wavegazer_net import WavegazerNet

BIOHUB = Path(r"D:\Kaggle_Biohub_Data\train")
NAME = "44b6_2f31fc2f"
YX_UM = 0.40625
Z_UM = 1.625


def main() -> None:
    zp = BIOHUB / f"{NAME}.zarr"
    gp = BIOHUB / f"{NAME}.geff"
    g = zarr.open(gp, mode="r")
    props = g["nodes"]["props"]
    nd = {k: np.asarray(props[k]["values"]) for k in ("x", "y", "t", "z")}
    arr = zarr.open(zp, mode="r")["0"]
    print("volume", NAME, "zarr", arr.shape, "nodes", len(nd["t"]))
    values, counts = np.unique(nd["t"], return_counts=True)
    t0 = int(values[int(np.argmax(counts))])
    sel = nd["t"] == t0
    print("busiest t", t0, "n_gt", int(sel.sum()), "n_times", len(values), "max_t_count", int(counts.max()))
    print("all-t node count", len(nd["t"]), "mean nodes/t", len(nd["t"]) / max(len(values), 1))
    gt = torch.tensor(np.stack([nd["x"][sel], nd["y"][sel], nd["z"][sel]], axis=1), dtype=torch.float32)
    print("gt xyz", gt.tolist())

    vol = np.asarray(arr[t0]).astype(np.float32)
    lo, hi = np.quantile(vol, [0.01, 0.99])
    vol = np.clip((vol - lo) / max(float(hi - lo), 1e-6), 0.0, 1.0)

    net = WavegazerNet(1, 2, sparse=True)
    sig = sigma_px(YX_UM, MATCH_UM)
    window = max(int(2 * sig) | 1, 3)
    scale = torch.tensor([YX_UM, YX_UM, Z_UM])

    # Per-GT: intensity at voxel, DoG at plane, rank among plane peaks.
    for i, gxyz in enumerate(gt):
        x, y, z = [float(v) for v in gxyz]
        zi = int(round(z))
        yi = int(round(y))
        xi = int(round(x))
        zi = int(np.clip(zi, 0, vol.shape[0] - 1))
        yi = int(np.clip(yi, 0, vol.shape[1] - 1))
        xi = int(np.clip(xi, 0, vol.shape[2] - 1))
        img = torch.from_numpy(vol[zi])[None, None]
        blobs, s_map = net.blob_field(img, um_per_px=YX_UM, match_um=MATCH_UM)
        gate = float(detect_gate(blobs))
        blob_at = float(blobs[0, 0, yi, xi])
        s_at = float(s_map[0, 0, yi, xi])
        peaks = local_maxima(blobs, window=window, min_score=detect_gate(blobs))
        n_raw = int(peaks.xy.size(0))
        # nearest DoG peak in YX on this plane
        if n_raw:
            dpx = torch.hypot(peaks.xy[:, 0] - x, peaks.xy[:, 1] - y)
            nearest = int(torch.argmin(dpx))
            near_d = float(dpx[nearest])
            near_sc = float(peaks.score[nearest])
            order = torch.argsort(peaks.score, descending=True)
            rank = int((order == nearest).nonzero()[0]) + 1
        else:
            near_d, near_sc, rank = float("nan"), float("nan"), -1
        print(
            f"gt{i} xyz=({x:.1f},{y:.1f},{z:.1f}) I={vol[zi, yi, xi]:.4f} "
            f"blob={blob_at:.4f} gate={gate:.4f} S={s_at:.4f} "
            f"plane_peaks={n_raw} nearest_dog_px={near_d:.2f} "
            f"near_blob={near_sc:.4f} rank={rank}"
        )

    # Full 3D detect then nearest-pred distance per GT (the miss).
    import importlib.util

    spec = importlib.util.spec_from_file_location("cmp3d", ROOT / "scripts" / "compare_biohub_3d.py")
    cmp3d = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cmp3d)
    _peaks_volume = cmp3d._peaks_volume

    with torch.no_grad():
        pred, score = _peaks_volume(net, vol)
    d = dist_um(pred, gt, yx_um=YX_UM, z_um=Z_UM)
    for i in range(gt.size(0)):
        best = int(torch.argmin(d[:, i]))
        print(
            f"gt{i} nearest_pred_um={float(d[best, i]):.2f} "
            f"pred_xyz={pred[best].tolist()} score={float(score[best]):.4f} "
            f"n_pred={pred.size(0)}"
        )


if __name__ == "__main__":
    main()
