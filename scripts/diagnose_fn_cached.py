"""Inspect the 12 FN / 18 FP using cached detections vs GT edges."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import torch
import zarr

from wavegazer.blob import MATCH_UM
from wavegazer.track import _match_ids, link_nn, max_link_um, score_edges

BIOHUB = Path(r"D:\Kaggle_Biohub_Data\train")
NAME = "44b6_0113de3b"
YX, ZU = 0.40625, 1.625


def main() -> None:
    packed = torch.load(ROOT / "artifacts" / "peak_cache" / f"{NAME}.pt", map_location="cpu", weights_only=False)
    xyz_by_t = {int(k): v for k, v in packed["xyz"].items()}
    g = zarr.open(BIOHUB / f"{NAME}.geff", mode="r")
    props = g["nodes"]["props"]
    nd = {k: np.asarray(props[k]["values"]) for k in ("x", "y", "t", "z")}
    ids = np.asarray(g["nodes"]["ids"])
    edges = np.asarray(g["edges"]["ids"])
    id_to_row = {int(i): r for r, i in enumerate(ids.tolist())}
    gt_xyz = torch.tensor(np.stack([nd["x"], nd["y"], nd["z"]], 1), dtype=torch.float32)
    gt_t = torch.tensor(nd["t"], dtype=torch.float32)
    gt_edges = torch.tensor(
        [[id_to_row[int(s)], id_to_row[int(t)]] for s, t in edges.tolist() if int(s) in id_to_row],
        dtype=torch.long,
    )
    times = sorted(xyz_by_t)
    pred_xyz = torch.cat([xyz_by_t[t] for t in times], dim=0)
    pred_t = torch.cat([torch.full((xyz_by_t[t].size(0),), float(t)) for t in times])
    offset = {}
    n = 0
    for t in times:
        offset[t] = n
        n += xyz_by_t[t].size(0)
    link_um = max_link_um(MATCH_UM)
    pred_edges = []
    assign = {}  # global src -> global dst
    for t in range(max(times)):
        if t not in xyz_by_t or (t + 1) not in xyz_by_t:
            continue
        pairs = link_nn(xyz_by_t[t], xyz_by_t[t + 1], max_um=link_um, yx_um=YX, z_um=ZU)
        for i, j in pairs.tolist():
            gs, gd = offset[t] + i, offset[t + 1] + j
            assign[gs] = gd
            pred_edges.append([gs, gd])
    pe = torch.tensor(pred_edges, dtype=torch.long) if pred_edges else torch.zeros(0, 2, dtype=torch.long)
    p2g, g2p = _match_ids(pred_xyz, pred_t, gt_xyz, gt_t, MATCH_UM, YX, ZU)
    c = score_edges(pred_xyz, pred_t, pe, gt_xyz, gt_t, gt_edges, yx_um=YX, z_um=ZU)
    print("NN", c.tp, c.fp, c.fn, "J", c.edge_jaccard)
    sc_by_t = {int(k): v for k, v in packed["score"].items()}
    pred_sc = torch.cat([sc_by_t[t] for t in times])
    tp_d, fn_d = [], []
    scale = torch.tensor([YX, YX, ZU])
    for e_i, (gs, gt) in enumerate(gt_edges.tolist()):
        ps, pt = int(g2p[gs]), int(g2p[gt])
        if ps < 0 or pt < 0:
            print(f"edge{e_i} unmatched node ps={ps} pt={pt}")
            continue
        d_true = float(((pred_xyz[ps] - pred_xyz[pt]) * scale).pow(2).sum().sqrt())
        got = assign.get(ps)
        d_got = (
            float(((pred_xyz[ps] - pred_xyz[got]) * scale).pow(2).sum().sqrt())
            if got is not None
            else float("nan")
        )
        ok = got == pt
        if ok:
            tp_d.append(d_true)
            continue
        fn_d.append(d_true)
        dS_true = abs(float(pred_sc[ps] - pred_sc[pt]))
        dS_got = abs(float(pred_sc[ps] - pred_sc[got])) if got is not None else float("nan")
        print(
            f"FN/wrong edge{e_i} t={int(gt_t[gs])}->{int(gt_t[gt])} "
            f"d_true={d_true:.2f} d_got={d_got:.2f} dS_true={dS_true:.3f} dS_got={dS_got:.3f} "
            f"got={got} want={pt}"
        )
    print(
        "TP n", len(tp_d),
        "median/mean/min/max",
        float(np.median(tp_d)) if tp_d else None,
        float(np.mean(tp_d)) if tp_d else None,
        float(np.min(tp_d)) if tp_d else None,
        float(np.max(tp_d)) if tp_d else None,
    )
    print("FN d_true median", float(np.median(fn_d)) if fn_d else None)


if __name__ == "__main__":
    main()
