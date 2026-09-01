"""Biohub detect gate: centroid match at 7 µm (YX), vs intensity peaks.

This is the PIXEL_FIRST / competition *node* metric, not adj_edge_jaccard
(that needs linking). Floor to beat later: public CellMot ~0.848 on the
*full* score; this script only scores detect recall/precision @ 7 µm.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import torch

from wavegazer.blob import DEFAULT_YX_UM, MATCH_UM, multi_scale_blob_map, sigma_px
from wavegazer.peaks import PeakSet, detect_gate, local_maxima, match_xy, nms
from wavegazer.wavegazer_net import WavegazerNet

BIOHUB = Path(r"D:\Kaggle_Biohub_Data\train")
N_VOLUMES = 16
SEED = 20260831


def _nodes(geff_path: Path):
    import zarr

    g = zarr.open(geff_path, mode="r")
    props = g["nodes"]["props"]
    return (
        np.asarray(props["x"]["values"]),
        np.asarray(props["y"]["values"]),
        np.asarray(props["t"]["values"]),
        np.asarray(props["z"]["values"]),
    )


def _plane_and_gt(zarr_path: Path, geff_path: Path):
    import zarr

    x, y, t, z = _nodes(geff_path)
    if len(t) == 0:
        return None
    values, counts = np.unique(t, return_counts=True)
    t0 = int(values[int(np.argmax(counts))])
    sel = t == t0
    if int(sel.sum()) < 1:
        return None
    arr = zarr.open(zarr_path, mode="r")["0"]
    if t0 < 0 or t0 >= arr.shape[0]:
        return None
    z0 = int(np.clip(np.median(z[sel]), 0, arr.shape[1] - 1))
    planes = []
    for dz in (-1, 0, 1):
        zz = int(np.clip(z0 + dz, 0, arr.shape[1] - 1))
        plane = np.asarray(arr[t0, zz]).astype(np.float32)
        lo, hi = np.quantile(plane, [0.01, 0.99])
        plane = np.clip((plane - lo) / max(float(hi - lo), 1e-6), 0.0, 1.0)
        planes.append(torch.from_numpy(plane)[None, None])
    gt = torch.tensor(np.stack([x[sel], y[sel]], axis=1), dtype=torch.float32)
    return planes, gt, {"t": t0, "z": z0, "n_gt": int(gt.size(0))}


def _intensity_peaks(img: torch.Tensor, um_per_px: float) -> PeakSet:
    sig = sigma_px(um_per_px, MATCH_UM)
    field = multi_scale_blob_map(img, sig)
    window = max(int(2 * sig) | 1, 3)
    peaks = local_maxima(field, window=window, min_score=detect_gate(field))
    if peaks.xy.size(0) > 256:
        top = torch.argsort(peaks.score, descending=True)[:256]
        peaks = PeakSet(xy=peaks.xy[top], score=peaks.score[top])
    return nms(peaks, radius_px=sig)


def _mean_metrics(rows: list[dict]) -> dict:
    if not rows:
        return {}
    keys = [k for k in rows[0] if isinstance(rows[0][k], (int, float))]
    return {k: sum(r[k] for r in rows) / len(rows) for k in keys}


def main() -> None:
    if not BIOHUB.is_dir():
        raise SystemExit(f"missing {BIOHUB}")
    device = torch.device("cpu")  # detect is light; keep GPU free
    net = WavegazerNet(in_channels=1, n_classes=2, sparse=True).to(device)
    um = DEFAULT_YX_UM
    max_dist = MATCH_UM / um
    zarrs = sorted(p for p in BIOHUB.iterdir() if p.name.endswith(".zarr"))
    wave_rows, int_rows, names = [], [], []
    for zp in zarrs:
        if len(names) >= N_VOLUMES:
            break
        gp = zp.with_suffix(".geff")
        try:
            packed = _plane_and_gt(zp, gp)
        except Exception as exc:
            print(f"skip {zp.name}: {exc}")
            continue
        if packed is None:
            continue
        planes, gt, meta = packed
        sig = sigma_px(um, MATCH_UM)
        wave_xy, wave_sc = [], []
        int_xy, int_sc = [], []
        with torch.no_grad():
            for img in planes:
                img = img.to(device)
                pk = net.detect(img, um_per_px=um, match_um=MATCH_UM)
                wave_xy.append(pk.xy.cpu())
                wave_sc.append(pk.score.cpu())
                ip = _intensity_peaks(img, um)
                int_xy.append(ip.xy.cpu())
                int_sc.append(ip.score.cpu())
        wset = nms(
            PeakSet(xy=torch.cat(wave_xy, 0), score=torch.cat(wave_sc, 0)),
            radius_px=sig,
        )
        iset = nms(
            PeakSet(xy=torch.cat(int_xy, 0), score=torch.cat(int_sc, 0)),
            radius_px=sig,
        )
        pred, inten = wset.xy, iset.xy
        w = match_xy(pred, gt, max_dist)
        i = match_xy(inten, gt, max_dist)
        w.update(meta)
        i.update(meta)
        wave_rows.append(w)
        int_rows.append(i)
        names.append(zp.stem)
        print(
            f"{zp.stem} t={meta['t']} z={meta['z']} gt={w['n_gt']} "
            f"wave p={w['n_pred']} rec={w['recall']:.3f} prec={w['precision']:.3f} "
            f"int rec={i['recall']:.3f} prec={i['precision']:.3f}"
        )

    payload = {
        "metric": "centroid_match_yx_7um",
        "match_um": MATCH_UM,
        "yx_um_per_px": um,
        "max_dist_px": max_dist,
        "n_volumes": len(names),
        "volumes": names,
        "wavegazer": {"per": wave_rows, "mean": _mean_metrics(wave_rows)},
        "intensity_dog": {"per": int_rows, "mean": _mean_metrics(int_rows)},
        "note": (
            "YX-only 7 µm on the median-z plane at the busiest t. "
            "Not adj_edge_jaccard. Intensity baseline uses the same φ-DoG + NMS "
            "without the Optics S readout. Beat intensity_dog.f1, then CellMot detect."
        ),
    }
    out = ROOT / "artifacts" / "biohub_peaks_7um.json"
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({"wavegazer_mean": payload["wavegazer"]["mean"], "intensity_mean": payload["intensity_dog"]["mean"]}, indent=2))
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
