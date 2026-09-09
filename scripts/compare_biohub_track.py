"""First linker score: greedy t→t+1 NN vs sparse GEFF edges.

Not CellMot transformer+ILP. Prints edge Jaccard and density-adjusted
Jaccard on a few full videos so we can see the gap to 0.848 / 0.985.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import torch
import zarr

from wavegazer.blob import MATCH_UM
from wavegazer.track import link_nn, max_link_um, score_edges
from wavegazer.wavegazer_net import WavegazerNet

BIOHUB = Path(r"D:\Kaggle_Biohub_Data\train")
N_VOLUMES = 1
YX_UM = 0.40625
Z_UM = 1.625


def _geff(geff_path: Path):
    g = zarr.open(geff_path, mode="r")
    props = g["nodes"]["props"]
    nd = {k: np.asarray(props[k]["values"]) for k in ("x", "y", "t", "z")}
    ids = np.asarray(g["nodes"]["ids"])
    edges = np.asarray(g["edges"]["ids"])
    extra = dict(g.attrs.get("geff", {})).get("extra") or {}
    t_true = float(extra.get("estimated_number_of_nodes") or float("nan"))
    return nd, ids, edges, t_true


def _norm_vol(vol: np.ndarray) -> np.ndarray:
    lo, hi = np.quantile(vol, [0.01, 0.99])
    return np.clip((vol - lo) / max(float(hi - lo), 1e-6), 0.0, 1.0).astype(np.float32)


def main() -> None:
    net = WavegazerNet(1, 2, sparse=True)
    net.eval()
    link_um = max_link_um(MATCH_UM)
    zarrs = sorted(p for p in BIOHUB.iterdir() if p.name.endswith(".zarr"))
    rows, names = [], []
    for zp in zarrs:
        if len(names) >= N_VOLUMES:
            break
        gp = zp.with_suffix(".geff")
        try:
            nd, ids, edges, t_true = _geff(gp)
            arr = zarr.open(zp, mode="r")["0"]
        except Exception as exc:
            print("skip", zp.name, exc)
            continue
        if len(nd["t"]) == 0:
            continue
        id_to_row = {int(i): r for r, i in enumerate(ids.tolist())}
        gt_edges = []
        for s, t in edges.tolist():
            if int(s) in id_to_row and int(t) in id_to_row:
                gt_edges.append([id_to_row[int(s)], id_to_row[int(t)]])
        gt_xyz = torch.tensor(np.stack([nd["x"], nd["y"], nd["z"]], axis=1), dtype=torch.float32)
        gt_t = torch.tensor(nd["t"], dtype=torch.float32)
        gt_edges_t = torch.tensor(gt_edges, dtype=torch.long) if gt_edges else torch.zeros(0, 2, dtype=torch.long)

        t_n = int(arr.shape[0])
        xyz_by_t: dict[int, torch.Tensor] = {}
        with torch.no_grad():
            for t in range(t_n):
                vol = _norm_vol(np.asarray(arr[t]))
                pred, _ = net.detect_volume(
                    torch.from_numpy(vol), yx_um=YX_UM, z_um=Z_UM, nms_um=MATCH_UM,
                )
                xyz_by_t[t] = pred
                if t % 20 == 0:
                    print(f"  {zp.stem} t={t}/{t_n} peaks={pred.size(0)}", flush=True)

        pred_xyz, pred_t, offset = [], [], {}
        n = 0
        for t in range(t_n):
            p = xyz_by_t[t]
            offset[t] = n
            if p.size(0):
                pred_xyz.append(p)
                pred_t.append(torch.full((p.size(0),), float(t)))
                n += int(p.size(0))
        if not pred_xyz:
            print(zp.stem, "no peaks")
            continue
        pred_xyz_t = torch.cat(pred_xyz, dim=0)
        pred_t_t = torch.cat(pred_t, dim=0)

        pred_edges = []
        for t in range(t_n - 1):
            a, b = xyz_by_t[t], xyz_by_t[t + 1]
            pairs = link_nn(a, b, max_um=link_um, yx_um=YX_UM, z_um=Z_UM)
            if pairs.numel() == 0:
                continue
            pred_edges.append(torch.stack([pairs[:, 0] + offset[t], pairs[:, 1] + offset[t + 1]], dim=1))
        pred_edges_t = torch.cat(pred_edges, dim=0) if pred_edges else torch.zeros(0, 2, dtype=torch.long)

        c = score_edges(
            pred_xyz_t, pred_t_t, pred_edges_t, gt_xyz, gt_t, gt_edges_t,
            match_um=MATCH_UM, yx_um=YX_UM, z_um=Z_UM,
        )
        adj = c.adj_edge_jaccard(t_true)
        row = {
            "n_gt_nodes": c.n_gt_nodes,
            "n_pred_nodes": c.n_pred_nodes,
            "n_gt_edges": c.n_gt_edges,
            "n_pred_edges": c.n_pred_edges,
            "node_tp": c.node_tp,
            "node_recall": c.node_recall,
            "edge_tp": c.tp,
            "edge_fp": c.fp,
            "edge_fn": c.fn,
            "edge_jaccard": c.edge_jaccard,
            "t_true": t_true,
            "adj_edge_jaccard": adj,
            "t_frames": t_n,
        }
        rows.append(row)
        names.append(zp.stem)
        print(
            f"{zp.stem} node_rec={c.node_recall:.3f} "
            f"J={c.edge_jaccard:.3f} J_adj={adj:.3f} "
            f"tp/fp/fn={c.tp}/{c.fp}/{c.fn} "
            f"T_pred={c.n_pred_nodes} T_true={t_true:.0f}"
        )

    def _mean(rs):
        if not rs:
            return {}
        keys = [k for k in rs[0] if isinstance(rs[0][k], (int, float))]
        return {k: sum(r[k] for r in rs) / len(rs) for k in keys}

    payload = {
        "metric": "greedy_nn_adj_edge_jaccard",
        "linker": "next_frame_nn",
        "max_link_um": link_um,
        "n_volumes": len(names),
        "volumes": names,
        "wavegazer": {"per": rows, "mean": _mean(rows)},
        "note": (
            "Greedy t→t+1 nearest neighbor, max_um=π·7. Not transformer+ILP. "
            "adj_edge_jaccard uses estimated_number_of_nodes. No division term."
        ),
        "competition_context": {
            "public_floor_full_score": 0.848,
            "public_top_full_score": 0.985,
            "full_metric": "adj_edge_jaccard + 0.1*division_jaccard",
        },
    }
    out = ROOT / "artifacts" / "biohub_track_nn.json"
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({"mean": payload["wavegazer"]["mean"]}, indent=2))
    print("wrote", out)


if __name__ == "__main__":
    main()
