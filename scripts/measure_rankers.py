"""Do labeled cells rank in the T_true budget under seed score formulas?"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import torch
import zarr

from wavegazer.blob import MATCH_UM, sigma_px, sigma_um
from wavegazer.fsot_seeds import SEEDS
from wavegazer.peaks import detect_gate, dist_um, local_maxima, nms_xyz_um
from wavegazer.wavegazer_net import WavegazerNet

BIOHUB = Path(r"D:\Kaggle_Biohub_Data\train")
YX, ZU = 0.40625, 1.625
N = 16


def peaks_with_parts(net, vol):
    sig = sigma_px(YX, MATCH_UM)
    window = max(int(2 * sig) | 1, 3)
    xs, ys, zs, blob, sraw = [], [], [], [], []
    for z in range(vol.shape[0]):
        img = torch.from_numpy(vol[z])[None, None]
        blobs, s_map = net.blob_field(img, um_per_px=YX, match_um=MATCH_UM)
        pk = local_maxima(blobs, window=window, min_score=detect_gate(blobs))
        if pk.xy.size(0) == 0:
            continue
        xi = pk.xy[:, 0].long().clamp(0, blobs.size(-1) - 1)
        yi = pk.xy[:, 1].long().clamp(0, blobs.size(-2) - 1)
        xs.append(pk.xy[:, 0])
        ys.append(pk.xy[:, 1])
        zs.append(torch.full((pk.xy.size(0),), float(z)))
        blob.append(blobs[0, 0, yi, xi])
        sraw.append(s_map[0, 0, yi, xi])
    xyz = torch.stack([torch.cat(xs), torch.cat(ys), torch.cat(zs)], dim=1)
    b = torch.cat(blob)
    s = torch.cat(sraw)
    s_n = (s - s.min()) / (s.max().clamp_min(s.min() + 1e-6) - s.min() + 1e-8)
    scores = {
        "blob": b,
        "s": s_n,
        "blob_pnew_s": b + SEEDS.p_new * s_n,
        "residual": b * (1.0 + SEEDS.p_new * s_n),
    }
    kept_xyz, kept_sc = nms_xyz_um(
        xyz, scores["blob_pnew_s"], sigma_um(MATCH_UM), yx_um=YX, z_um=ZU,
    )
    # Map kept rows back by exact (x,y,z) match after score-sort NMS.
    key = xyz[:, 0] * 1_000_003 + xyz[:, 1] * 1_009 + xyz[:, 2]
    kkey = kept_xyz[:, 0] * 1_000_003 + kept_xyz[:, 1] * 1_009 + kept_xyz[:, 2]
    pos = {float(k): i for i, k in enumerate(key.tolist())}
    idx = torch.tensor([pos[float(k)] for k in kkey.tolist()], dtype=torch.long)
    xyz = xyz[idx]
    return xyz, {k: v[idx] for k, v in scores.items()}


def main() -> None:
    net = WavegazerNet(1, 2, sparse=True)
    zarrs = sorted(p for p in BIOHUB.iterdir() if p.name.endswith(".zarr"))[:N]
    names = ["blob", "s", "blob_pnew_s", "residual"]
    rec = {k: [] for k in names}
    print("ranker mean_recall_at_budget")
    for zp in zarrs:
        g = zarr.open(zp.with_suffix(".geff"), mode="r")
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
        gt = torch.tensor(np.stack([nd["x"][sel], nd["y"][sel], nd["z"][sel]], 1), dtype=torch.float32)
        xyz, parts = peaks_with_parts(net, vol)
        budget = max(1, int(round(t_true / arr.shape[0])))
        d = dist_um(xyz, gt, yx_um=YX, z_um=ZU)
        hit = (d.min(dim=0).values <= MATCH_UM)
        for k, sc in parts.items():
            order = torch.argsort(sc, descending=True)
            keep = set(order[:budget].tolist())
            tp = 0
            used = set()
            for g_i in range(gt.size(0)):
                # nearest pred under 7 µm that is in keep
                col = d[:, g_i]
                for p in torch.argsort(col).tolist():
                    if float(col[p]) > MATCH_UM:
                        break
                    if p in keep and p not in used:
                        tp += 1
                        used.add(p)
                        break
            rec[k].append(tp / max(int(gt.size(0)), 1))
    for k in names:
        print(k, sum(rec[k]) / len(rec[k]))


if __name__ == "__main__":
    main()
