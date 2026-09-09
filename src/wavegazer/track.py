"""Next-frame linking and sparse-GT edge Jaccard.

This is the first linker, not CellMot's transformer+ILP. Gate: recover
labeled GEFF edges after 7 µm node match. Official score still needs
divisions + density vs estimated_number_of_nodes.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .blob import MATCH_UM
from .fsot_seeds import SEEDS
from .peaks import dist_um


def max_link_um(match_um: float = MATCH_UM) -> float:
    """Search radius for t → t+1. Inverse of σ = match/π so cells may move ~π match."""
    return match_um * SEEDS.pi


def link_nn(
    xyz_t: torch.Tensor,
    xyz_tp: torch.Tensor,
    *,
    max_um: float,
    yx_um: float,
    z_um: float,
) -> torch.Tensor:
    """Greedy one-to-one nearest neighbor. Returns (E,2) indices into t and t+1."""
    n, m = int(xyz_t.size(0)), int(xyz_tp.size(0))
    if n == 0 or m == 0:
        return torch.zeros(0, 2, dtype=torch.long)
    d = dist_um(xyz_t, xyz_tp, yx_um=yx_um, z_um=z_um)
    flat = d.reshape(-1)
    order = torch.argsort(flat)
    used_t = torch.zeros(n, dtype=torch.bool, device=d.device)
    used_p = torch.zeros(m, dtype=torch.bool, device=d.device)
    pairs: list[tuple[int, int]] = []
    for idx in order.tolist():
        if float(flat[idx]) > max_um:
            break
        i, j = divmod(int(idx), m)
        if used_t[i] or used_p[j]:
            continue
        used_t[i] = True
        used_p[j] = True
        pairs.append((i, j))
        if len(pairs) == min(n, m):
            break
    if not pairs:
        return torch.zeros(0, 2, dtype=torch.long, device=xyz_t.device)
    return torch.tensor(pairs, dtype=torch.long, device=xyz_t.device)


@dataclass
class EdgeCounts:
    tp: int
    fp: int
    fn: int
    n_pred_nodes: int
    n_gt_nodes: int
    n_gt_edges: int
    n_pred_edges: int
    node_tp: int

    @property
    def edge_jaccard(self) -> float:
        den = self.tp + self.fp + self.fn
        return self.tp / den if den else float("nan")

    @property
    def node_recall(self) -> float:
        return self.node_tp / self.n_gt_nodes if self.n_gt_nodes else 1.0

    def adj_edge_jaccard(self, t_true: float, alpha: float = 0.1) -> float:
        j = self.edge_jaccard
        if j != j or not (t_true > 0):
            return float("nan")
        ratio = (self.n_pred_nodes - t_true) / t_true
        return max(0.0, j * (1.0 - alpha * ratio))


def _match_ids(
    pred_xyz: torch.Tensor,
    pred_t: torch.Tensor,
    gt_xyz: torch.Tensor,
    gt_t: torch.Tensor,
    max_um: float,
    yx_um: float,
    z_um: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-time bipartite 7 µm match. Returns pred→gt and gt→pred (-1 unmatched)."""
    n_p, n_g = int(pred_xyz.size(0)), int(gt_xyz.size(0))
    p2g = torch.full((n_p,), -1, dtype=torch.long)
    g2p = torch.full((n_g,), -1, dtype=torch.long)
    if n_p == 0 or n_g == 0:
        return p2g, g2p
    times = torch.unique(torch.cat([pred_t, gt_t])).tolist()
    for t in times:
        pi = torch.where(pred_t == t)[0]
        gi = torch.where(gt_t == t)[0]
        if pi.numel() == 0 or gi.numel() == 0:
            continue
        d = dist_um(pred_xyz[pi], gt_xyz[gi], yx_um=yx_um, z_um=z_um)
        flat = d.reshape(-1)
        order = torch.argsort(flat)
        used_p = torch.zeros(pi.numel(), dtype=torch.bool)
        used_g = torch.zeros(gi.numel(), dtype=torch.bool)
        n_g_t = int(gi.numel())
        for idx in order.tolist():
            if float(flat[idx]) > max_um:
                break
            a, b = divmod(int(idx), n_g_t)
            if used_p[a] or used_g[b]:
                continue
            used_p[a] = True
            used_g[b] = True
            p2g[int(pi[a])] = int(gi[b])
            g2p[int(gi[b])] = int(pi[a])
    return p2g, g2p


def score_edges(
    pred_xyz: torch.Tensor,
    pred_t: torch.Tensor,
    pred_edges: torch.Tensor,
    gt_xyz: torch.Tensor,
    gt_t: torch.Tensor,
    gt_edges: torch.Tensor,
    *,
    match_um: float = MATCH_UM,
    yx_um: float,
    z_um: float,
) -> EdgeCounts:
    """Sparse-GT edge TP/FP/FN after 7 µm node match (metrics.md rules)."""
    p2g, g2p = _match_ids(pred_xyz, pred_t, gt_xyz, gt_t, match_um, yx_um, z_um)
    node_tp = int((g2p >= 0).sum())
    n_g = int(gt_xyz.size(0))
    n_p = int(pred_xyz.size(0))

    gt_out: dict[int, set[int]] = {i: set() for i in range(n_g)}
    gt_in: dict[int, set[int]] = {i: set() for i in range(n_g)}
    gt_pair: set[tuple[int, int]] = set()
    for e in gt_edges.tolist() if gt_edges.numel() else []:
        s, t = int(e[0]), int(e[1])
        gt_out[s].add(t)
        gt_in[t].add(s)
        gt_pair.add((s, t))

    tp = 0
    fp = 0
    seen_tp: set[tuple[int, int]] = set()
    n_pred_edges = int(pred_edges.size(0))
    for e in pred_edges.tolist() if pred_edges.numel() else []:
        ps, pt = int(e[0]), int(e[1])
        gs = int(p2g[ps])
        gt = int(p2g[pt])
        if gs >= 0 and gt >= 0 and (gs, gt) in gt_pair:
            if (gs, gt) not in seen_tp:
                tp += 1
                seen_tp.add((gs, gt))
            continue
        source_conflict = gs >= 0 and len(gt_out[gs]) > 0
        target_conflict = gt >= 0 and len(gt_in[gt]) > 0
        if source_conflict or target_conflict:
            fp += 1
    fn = len(gt_pair) - tp
    return EdgeCounts(
        tp=tp,
        fp=fp,
        fn=fn,
        n_pred_nodes=n_p,
        n_gt_nodes=n_g,
        n_gt_edges=len(gt_pair),
        n_pred_edges=n_pred_edges,
        node_tp=node_tp,
    )
