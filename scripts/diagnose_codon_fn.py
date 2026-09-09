"""Codon ident on the 12 FN pairs vs the stolen clutter, from cache."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import torch
import zarr

from wavegazer.blob import MATCH_UM
from wavegazer.track import _match_ids, codon_ident, link_nn, max_link_um, score_edges

BIOHUB = Path(r"D:\Kaggle_Biohub_Data\train")
NAME = "44b6_0113de3b"
YX, ZU = 0.40625, 1.625


def main() -> None:
    packed = torch.load(ROOT / "artifacts" / "peak_cache" / f"{NAME}.pt", map_location="cpu", weights_only=False)
    xyz_by_t = {int(k): v for k, v in packed["xyz"].items()}
    codon_by_t = {int(k): v for k, v in packed["codon"].items()}
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
    pred_c = torch.cat([codon_by_t[t] for t in times], dim=0)
    offset, n = {}, 0
    for t in times:
        offset[t] = n
        n += xyz_by_t[t].size(0)
    link_um = max_link_um(MATCH_UM)
    assign = {}
    for t in range(max(times)):
        if t not in xyz_by_t or t + 1 not in xyz_by_t:
            continue
        pairs = link_nn(xyz_by_t[t], xyz_by_t[t + 1], max_um=link_um, yx_um=YX, z_um=ZU)
        for i, j in pairs.tolist():
            assign[offset[t] + i] = offset[t + 1] + j
    p2g, g2p = _match_ids(pred_xyz, pred_t, gt_xyz, gt_t, MATCH_UM, YX, ZU)
    n_codon_wins = 0
    n_fn = 0
    for gs, gt in gt_edges.tolist():
        ps, pt = int(g2p[gs]), int(g2p[gt])
        if ps < 0 or pt < 0:
            continue
        got = assign.get(ps)
        if got == pt:
            continue
        n_fn += 1
        id_true = float(codon_ident(pred_c[ps][None], pred_c[pt][None]))
        id_got = float("nan")
        if got is not None:
            id_got = float(codon_ident(pred_c[ps][None], pred_c[got][None]))
        win = id_true > id_got if got is not None else False
        n_codon_wins += int(win)
        print(
            f"t={int(gt_t[gs])}->{int(gt_t[gt])} codon_true={id_true:.3f} "
            f"codon_got={id_got:.3f} win={win}"
        )
    print("FN", n_fn, "codon prefers true", n_codon_wins)


if __name__ == "__main__":
    main()
