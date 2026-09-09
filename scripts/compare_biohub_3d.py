"""Competition-shaped detect: 3D anisotropic 7 µm match at one time.

CellMot / PIXEL_FIRST node rule is 7 µm in real space, not 17 px in YX only.
This scans every Z at the busiest t.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import torch

from wavegazer.blob import MATCH_UM, sigma_um
from wavegazer.peaks import match_xyz_um
from wavegazer.wavegazer_net import WavegazerNet

BIOHUB = Path(r"D:\Kaggle_Biohub_Data\train")
N_VOLUMES = 16
YX_UM = 0.40625
Z_UM = 1.625


def _nodes(geff_path: Path):
    import zarr

    g = zarr.open(geff_path, mode="r")
    props = g["nodes"]["props"]
    return {k: np.asarray(props[k]["values"]) for k in ("x", "y", "t", "z")}


def _norm(plane: np.ndarray) -> torch.Tensor:
    lo, hi = np.quantile(plane, [0.01, 0.99])
    plane = np.clip((plane - lo) / max(float(hi - lo), 1e-6), 0.0, 1.0)
    return torch.from_numpy(plane.astype(np.float32))[None, None]


def _peaks_volume(net: WavegazerNet, vol: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
    """vol (Z,Y,X) float32 0-1. Returns xyz (P,3), score (P,)."""
    return net.detect_volume(torch.from_numpy(vol), yx_um=YX_UM, z_um=Z_UM, match_um=MATCH_UM)


def _mean_metrics(rows: list[dict]) -> dict:
    if not rows:
        return {}
    keys = [k for k in rows[0] if isinstance(rows[0][k], (int, float))]
    return {k: sum(r[k] for r in rows) / len(rows) for k in keys}


def main() -> None:
    import zarr

    net = WavegazerNet(1, 2, sparse=True)
    zarrs = sorted(p for p in BIOHUB.iterdir() if p.name.endswith(".zarr"))
    rows, names = [], []
    for zp in zarrs:
        if len(names) >= N_VOLUMES:
            break
        gp = zp.with_suffix(".geff")
        try:
            nd = _nodes(gp)
            arr = zarr.open(zp, mode="r")["0"]
        except Exception as exc:
            print("skip", zp.name, exc)
            continue
        if len(nd["t"]) == 0:
            continue
        values, counts = np.unique(nd["t"], return_counts=True)
        t0 = int(values[int(np.argmax(counts))])
        sel = nd["t"] == t0
        if int(sel.sum()) < 1 or t0 >= arr.shape[0]:
            continue
        vol = np.asarray(arr[t0]).astype(np.float32)
        lo, hi = np.quantile(vol, [0.01, 0.99])
        vol = np.clip((vol - lo) / max(float(hi - lo), 1e-6), 0.0, 1.0)
        gt = torch.tensor(
            np.stack([nd["x"][sel], nd["y"][sel], nd["z"][sel]], axis=1),
            dtype=torch.float32,
        )
        with torch.no_grad():
            pred, _ = _peaks_volume(net, vol)
        m = match_xyz_um(pred, gt, MATCH_UM, yx_um=YX_UM, z_um=Z_UM)
        m["t"] = t0
        m["z_planes"] = int(vol.shape[0])
        rows.append(m)
        names.append(zp.stem)
        print(
            f"{zp.stem} t={t0} gt={m['n_gt']} pred={m['n_pred']} "
            f"rec={m['recall']:.3f} prec={m['precision']:.3f}"
        )

    payload = {
        "metric": "centroid_match_3d_7um_anisotropic",
        "match_um": MATCH_UM,
        "nms_um": sigma_um(MATCH_UM),
        "yx_um": YX_UM,
        "z_um": Z_UM,
        "n_volumes": len(names),
        "volumes": names,
        "wavegazer": {"per": rows, "mean": _mean_metrics(rows)},
        "note": (
            "Official node rule: 7 µm Euclidean with YX=0.40625, Z=1.625. "
            "One time (busiest t), all Z. Not adj_edge_jaccard."
        ),
        "competition_context": {
            "public_floor_full_score": 0.848,
            "public_top_full_score": 0.985,
            "full_metric": "adj_edge_jaccard + 0.1*division_jaccard",
            "cellmot_detect_recall_reported": 0.98,
        },
    }
    out = ROOT / "artifacts" / "biohub_peaks_3d_7um.json"
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({"mean": payload["wavegazer"]["mean"]}, indent=2))
    print("wrote", out)


if __name__ == "__main__":
    main()
