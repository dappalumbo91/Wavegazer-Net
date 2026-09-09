"""For each GT edge, is the true next closer in S than the nearest clutter?"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import torch
import zarr

from wavegazer.blob import MATCH_UM, sigma_px
from wavegazer.fsot_seeds import COLLAPSE_THRESHOLD, SEEDS
from wavegazer.peaks import detect_gate, dist_um, local_maxima
from wavegazer.track import max_link_um
from wavegazer.wavegazer_net import WavegazerNet

BIOHUB = Path(r"D:\Kaggle_Biohub_Data\train")
NAME = "44b6_0113de3b"
YX, ZU = 0.40625, 1.625


def _plane(arr, t, z):
    z = int(np.clip(z, 0, arr.shape[1] - 1))
    p = np.asarray(arr[t, z]).astype(np.float32)
    lo, hi = np.quantile(p, [0.01, 0.99])
    p = np.clip((p - lo) / max(float(hi - lo), 1e-6), 0.0, 1.0)
    return torch.from_numpy(p)[None, None]


def main() -> None:
    zp = BIOHUB / f"{NAME}.zarr"
    gp = BIOHUB / f"{NAME}.geff"
    g = zarr.open(gp, mode="r")
    props = g["nodes"]["props"]
    nd = {k: np.asarray(props[k]["values"]) for k in ("x", "y", "t", "z")}
    ids = np.asarray(g["nodes"]["ids"])
    edges = np.asarray(g["edges"]["ids"])
    id_to_row = {int(i): r for r, i in enumerate(ids.tolist())}
    arr = zarr.open(zp, mode="r")["0"]
    net = WavegazerNet(1, 2, sparse=True)
    sig = sigma_px(YX, MATCH_UM)
    window = max(int(2 * sig) | 1, 3)
    link_um = max_link_um(MATCH_UM)
    xyz = torch.tensor(np.stack([nd["x"], nd["y"], nd["z"]], 1), dtype=torch.float32)
    scale = torch.tensor([YX, YX, ZU])

    n_true_closer = n_ident_wins = n_clutter_closer = 0
    ds_true, ds_clut, dS_true, dS_clut = [], [], [], []
    for s_id, t_id in edges.tolist():
        s, t = id_to_row[int(s_id)], id_to_row[int(t_id)]
        ts, tt = int(nd["t"][s]), int(nd["t"][t])
        zs, zt = int(round(nd["z"][s])), int(round(nd["z"][t]))
        img_s = _plane(arr, ts, zs)
        img_t = _plane(arr, tt, zt)
        blobs_s, smap_s = net.blob_field(img_s)
        blobs_t, smap_t = net.blob_field(img_t)
        xs, ys = int(nd["x"][s]), int(nd["y"][s])
        xt, yt = int(nd["x"][t]), int(nd["y"][t])
        H, W = blobs_t.shape[-2:]
        xs, ys = int(np.clip(xs, 0, W - 1)), int(np.clip(ys, 0, H - 1))
        xt, yt = int(np.clip(xt, 0, W - 1)), int(np.clip(yt, 0, H - 1))
        s_src = float(smap_s[0, 0, ys, xs])
        s_tgt = float(smap_t[0, 0, yt, xt])
        pk = local_maxima(blobs_t, window=window, min_score=detect_gate(blobs_t))
        if pk.xy.size(0) == 0:
            continue
        # 3D dist using target-plane z for all plane peaks
        pred = torch.stack(
            [pk.xy[:, 0], pk.xy[:, 1], torch.full((pk.xy.size(0),), float(zt))],
            dim=1,
        )
        d_src = dist_um(pred, xyz[s][None], yx_um=YX, z_um=ZU)[:, 0]
        d_tgt = dist_um(pred, xyz[t][None], yx_um=YX, z_um=ZU)[:, 0]
        true_i = int(torch.argmin(d_tgt))
        # nearest peak to source that is not the true target peak
        order = torch.argsort(d_src)
        clut_i = None
        for i in order.tolist():
            if i != true_i:
                clut_i = i
                break
        if clut_i is None:
            continue
        dt = float(d_src[true_i])
        dc = float(d_src[clut_i])
        xi = int(pk.xy[clut_i, 0].clamp(0, W - 1))
        yi = int(pk.xy[clut_i, 1].clamp(0, H - 1))
        s_clut = float(smap_t[0, 0, yi, xi])
        ds_true.append(dt)
        ds_clut.append(dc)
        dS_true.append(abs(s_src - s_tgt))
        dS_clut.append(abs(s_src - s_clut))
        if dt <= dc:
            n_true_closer += 1
        else:
            n_clutter_closer += 1
        if abs(s_src - s_tgt) < abs(s_src - s_clut):
            n_ident_wins += 1
        print(
            f"t={ts}->{tt} dist_true={dt:.2f} dist_clut={dc:.2f} "
            f"dS_true={abs(s_src-s_tgt):.4f} dS_clut={abs(s_src-s_clut):.4f} "
            f"steal={dc<dt}"
        )

    n = len(ds_true)
    print(
        "n", n,
        "true_closer", n_true_closer,
        "clutter_closer", n_clutter_closer,
        "ident_wins", n_ident_wins,
        "median d_true", float(np.median(ds_true)),
        "median d_clut", float(np.median(ds_clut)),
        "median dS_true", float(np.median(dS_true)),
        "median dS_clut", float(np.median(dS_clut)),
        "collapse_th", COLLAPSE_THRESHOLD,
        "p_new", SEEDS.p_new,
    )


if __name__ == "__main__":
    main()
