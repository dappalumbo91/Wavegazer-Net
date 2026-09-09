"""Why greedy NN dropped 12 edges and invented 18 on 44b6_0113de3b.

Split: (A) GT-only linking — if this is already 1.0, extras cause FP.
(B) GT edge lengths in µm — if many > π·7, the radius is the FN.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import torch
import zarr

from wavegazer.blob import MATCH_UM
from wavegazer.track import link_nn, max_link_um, score_edges

BIOHUB = Path(r"D:\Kaggle_Biohub_Data\train")
NAME = "44b6_0113de3b"
YX, ZU = 0.40625, 1.625


def main() -> None:
    gp = BIOHUB / f"{NAME}.geff"
    g = zarr.open(gp, mode="r")
    props = g["nodes"]["props"]
    nd = {k: np.asarray(props[k]["values"]) for k in ("x", "y", "t", "z")}
    ids = np.asarray(g["nodes"]["ids"])
    edges = np.asarray(g["edges"]["ids"])
    extra = dict(g.attrs.get("geff", {})).get("extra") or {}
    t_true = float(extra.get("estimated_number_of_nodes") or float("nan"))
    id_to_row = {int(i): r for r, i in enumerate(ids.tolist())}
    gt_edges = []
    for s, t in edges.tolist():
        if int(s) in id_to_row and int(t) in id_to_row:
            gt_edges.append([id_to_row[int(s)], id_to_row[int(t)]])
    xyz = torch.tensor(np.stack([nd["x"], nd["y"], nd["z"]], 1), dtype=torch.float32)
    tt = torch.tensor(nd["t"], dtype=torch.float32)
    ge = torch.tensor(gt_edges, dtype=torch.long)
    scale = torch.tensor([YX, YX, ZU])
    link_um = max_link_um(MATCH_UM)
    print("nodes", len(ids), "edges", len(gt_edges), "link_um", link_um, "t_true", t_true)

    lengths = []
    dt = []
    for s, t in gt_edges:
        d = float(((xyz[s] - xyz[t]) * scale).pow(2).sum().sqrt())
        lengths.append(d)
        dt.append(int(tt[t] - tt[s]))
    lengths = np.array(lengths)
    print(
        "gt edge um: min/median/mean/max",
        float(lengths.min()),
        float(np.median(lengths)),
        float(lengths.mean()),
        float(lengths.max()),
        "n_over_link",
        int((lengths > link_um).sum()),
        "n_over_7",
        int((lengths > MATCH_UM).sum()),
        "dt unique",
        sorted(set(dt)),
    )

    # GT-only greedy NN at each consecutive labeled time.
    times = sorted(set(int(x) for x in tt.tolist()))
    pred_edges = []
    for a, b in zip(times, times[1:]):
        ia = torch.where(tt == a)[0]
        ib = torch.where(tt == b)[0]
        if ia.numel() == 0 or ib.numel() == 0:
            continue
        pairs = link_nn(xyz[ia], xyz[ib], max_um=link_um, yx_um=YX, z_um=ZU)
        if pairs.numel() == 0:
            continue
        pred_edges.append(torch.stack([ia[pairs[:, 0]], ib[pairs[:, 1]]], dim=1))
    pe = torch.cat(pred_edges, dim=0) if pred_edges else torch.zeros(0, 2, dtype=torch.long)
    c = score_edges(xyz, tt, pe, xyz, tt, ge, match_um=MATCH_UM, yx_um=YX, z_um=ZU)
    print(
        "GT-only NN",
        f"J={c.edge_jaccard:.3f}",
        f"tp/fp/fn={c.tp}/{c.fp}/{c.fn}",
        f"pred_edges={c.n_pred_edges}",
        f"node_rec={c.node_recall:.3f}",
        f"adj={c.adj_edge_jaccard(t_true):.3f}",
    )

    # Also: always link the true next if dt==1, report how many GT edges skip time.
    skip = sum(1 for d in dt if d != 1)
    print("gt edges with dt!=1", skip, "of", len(dt))


if __name__ == "__main__":
    main()
