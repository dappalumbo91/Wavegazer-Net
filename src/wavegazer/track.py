"""Next-frame linking and sparse-GT edge Jaccard.

This is the first linker, not CellMot's transformer+ILP. Gate: recover
labeled GEFF edges after 7 µm node match. Official score still needs
divisions + density vs estimated_number_of_nodes.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .blob import MATCH_UM, sigma_um
from .codon_kernels import trit_similarity_matrix
from .fsot_seeds import COLLAPSE_THRESHOLD, COMPACTIFICATION_CEILING, SEEDS

INV_PHI = 1.0 / SEEDS.phi
from .peaks import dist_um

_TRIT_SIM: torch.Tensor | None = None


def _trit_sim(device, dtype) -> torch.Tensor:
    global _TRIT_SIM
    if _TRIT_SIM is None or _TRIT_SIM.device != device or _TRIT_SIM.dtype != dtype:
        _TRIT_SIM = trit_similarity_matrix().to(device=device, dtype=dtype)
    return _TRIT_SIM


def patch_ncc(
    vol_t: torch.Tensor,
    xyz_t: torch.Tensor,
    vol_tp: torch.Tensor,
    xyz_tp: torch.Tensor,
    *,
    radius: int,
) -> torch.Tensor:
    """Normalized cross-correlation of luma patches. Cell-scale identity, no weights."""
    if vol_t.dim() == 4:
        vol_t = vol_t[0]
    if vol_tp.dim() == 4:
        vol_tp = vol_tp[0]

    def _pats(vol: torch.Tensor, xyz: torch.Tensor) -> torch.Tensor:
        if xyz.numel() == 0:
            return vol.new_zeros(0, (2 * radius + 1) ** 2)
        z_n, h, w = int(vol.size(0)), int(vol.size(1)), int(vol.size(2))
        r = radius
        plane_pad = {}
        out = vol.new_zeros(xyz.size(0), (2 * r + 1) ** 2)
        zi = xyz[:, 2].long().clamp(0, z_n - 1)
        for z in zi.unique().tolist():
            if z not in plane_pad:
                plane_pad[z] = torch.nn.functional.pad(vol[int(z)], (r, r, r, r))
            sel = zi == z
            pad = plane_pad[z]
            xs = xyz[sel, 0].long() + r
            ys = xyz[sel, 1].long() + r
            rows = []
            for x, y in zip(xs.tolist(), ys.tolist()):
                x = min(max(x, r), pad.size(1) - r - 1)
                y = min(max(y, r), pad.size(0) - r - 1)
                rows.append(pad[y - r : y + r + 1, x - r : x + r + 1].reshape(-1))
            out[sel] = torch.stack(rows, dim=0)
        return out

    def _ncc(xyz_a: torch.Tensor, xyz_b: torch.Tensor) -> torch.Tensor:
        a = _pats(vol_t, xyz_a)
        b = _pats(vol_tp, xyz_b)
        a = a - a.mean(dim=-1, keepdim=True)
        b = b - b.mean(dim=-1, keepdim=True)
        a = a / a.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        b = b / b.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        return ((a @ b.transpose(0, 1)).clamp(-1.0, 1.0) + 1.0) * 0.5

    acc = _ncc(xyz_t, xyz_tp)
    for dz in (-1.0, 1.0):
        sa = xyz_t.clone()
        sb = xyz_tp.clone()
        sa[:, 2] = (sa[:, 2] + dz).clamp(0, vol_t.size(0) - 1)
        sb[:, 2] = (sb[:, 2] + dz).clamp(0, vol_tp.size(0) - 1)
        acc = acc + _ncc(sa, sb)
    return acc / 3.0


def codon_ident(codes_t: torch.Tensor, codes_tp: torch.Tensor) -> torch.Tensor:
    """Genetics compare in codon space.

    L2-normalizing wiped identity: 3×3 codon responses at blob centers are
    nearly parallel (same kernel pattern, different brightness). Relative L1
    keeps magnitude, then trit-bilinear scores the *direction* of the
    residual ``u/|u|``. Product is the pair identity, mapped through collapse Θ.
    """
    if codes_t.numel() == 0 or codes_tp.numel() == 0:
        return codes_t.new_zeros(codes_t.size(0), codes_tp.size(0))
    l1 = (codes_t[:, None, :] - codes_tp[None, :, :]).abs().sum(dim=-1)
    nrm = codes_t.abs().sum(dim=-1)[:, None] + codes_tp.abs().sum(dim=-1)[None, :]
    rel = l1 / nrm.clamp_min(1e-6)
    mag = 1.0 / (1.0 + rel / COLLAPSE_THRESHOLD)
    dim = int(codes_t.size(-1))
    n_blk = max(1, dim // 64)
    ang = mag.new_zeros(mag.shape)
    sim = None
    for b in range(n_blk):
        u = torch.nn.functional.normalize(codes_t[:, b * 64 : (b + 1) * 64], dim=-1, eps=1e-6)
        v = torch.nn.functional.normalize(codes_tp[:, b * 64 : (b + 1) * 64], dim=-1, eps=1e-6)
        if sim is None:
            sim = _trit_sim(u.device, u.dtype)
        ang = ang + ((u @ sim @ v.transpose(0, 1)).clamp(-1.0, 1.0) + 1.0) * 0.5
    ang = ang / float(n_blk)
    return (mag * ang).clamp(min=0.0) ** SEEDS.phi


def max_link_um(match_um: float = MATCH_UM) -> float:
    """NN control radius. Inverse of σ = match/π."""
    return match_um * SEEDS.pi


def max_link_um_fsot(match_um: float = MATCH_UM) -> float:
    """Bleed search ball: one golden match. π·match was letting 15 µm leftovers steal."""
    return match_um * SEEDS.phi


def expected_step_um(match_um: float = MATCH_UM) -> float:
    """Fluid first-step prior: one golden DoG length. GT median on 0113de3b is ~2.9 µm."""
    return sigma_um(match_um) * SEEDS.phi


def min_track_len() -> int:
    """φ² ≈ 2.6 → 3. Drop isolates/pairs. φ⁴=7 is for after tracks are already long."""
    return max(2, int(round(SEEDS.phi ** 2)))


def _greedy_pairs(score: torch.Tensor) -> list[tuple[int, int]]:
    n, m = score.shape
    flat = score.reshape(-1)
    order = torch.argsort(flat, descending=True)
    used_t = torch.zeros(n, dtype=torch.bool, device=score.device)
    used_p = torch.zeros(m, dtype=torch.bool, device=score.device)
    pairs: list[tuple[int, int]] = []
    for idx in order.tolist():
        if float(flat[idx]) < 0.0:
            break
        i, j = divmod(int(idx), m)
        if used_t[i] or used_p[j]:
            continue
        used_t[i] = True
        used_p[j] = True
        pairs.append((i, j))
        if len(pairs) == min(n, m):
            break
    return pairs


def snap_xyz(
    vol: torch.Tensor,
    xyz: torch.Tensor,
    *,
    xy_r: int = 1,
    z_r: int = 1,
) -> torch.Tensor:
    """Intensity center-of-mass in a window. Argmax jumped to specks; COM does not."""
    if xyz.numel() == 0:
        return xyz
    if vol.dim() == 4:
        vol = vol[0]
    z_n, h, w = int(vol.size(0)), int(vol.size(1)), int(vol.size(2))
    out = xyz.clone()
    for i in range(xyz.size(0)):
        xi = int(round(float(xyz[i, 0])))
        yi = int(round(float(xyz[i, 1])))
        zi = int(round(float(xyz[i, 2])))
        z0, z1 = max(0, zi - z_r), min(z_n, zi + z_r + 1)
        y0, y1 = max(0, yi - xy_r), min(h, yi + xy_r + 1)
        x0, x1 = max(0, xi - xy_r), min(w, xi + xy_r + 1)
        patch = vol[z0:z1, y0:y1, x0:x1].clamp_min(0)
        mass = float(patch.sum())
        if mass <= 1e-8:
            continue
        zz = torch.arange(z0, z1, device=vol.device, dtype=patch.dtype)
        yy = torch.arange(y0, y1, device=vol.device, dtype=patch.dtype)
        xx = torch.arange(x0, x1, device=vol.device, dtype=patch.dtype)
        wz = (patch * zz[:, None, None]).sum() / mass
        wy = (patch * yy[None, :, None]).sum() / mass
        wx = (patch * xx[None, None, :]).sum() / mass
        out[i, 0] = float(wx)
        out[i, 1] = float(wy)
        out[i, 2] = float(wz)
    return out


def filter_short_tracks(
    n_nodes: int,
    edges: torch.Tensor,
    *,
    min_len: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Drop weakly-connected components shorter than φ⁴. Returns keep-mask, filtered edges."""
    if min_len is None:
        min_len = min_track_len()
    keep = torch.zeros(n_nodes, dtype=torch.bool)
    if n_nodes == 0:
        return keep, edges
    parent = list(range(n_nodes))

    def find(a: int) -> int:
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    if edges.numel():
        for s, t in edges.tolist():
            union(int(s), int(t))
    size = [0] * n_nodes
    for i in range(n_nodes):
        size[find(i)] += 1
    for i in range(n_nodes):
        if size[find(i)] >= min_len:
            keep[i] = True
    if edges.numel() == 0:
        return keep, edges
    src, dst = edges[:, 0], edges[:, 1]
    ekeep = keep[src] & keep[dst]
    return keep, edges[ekeep]


def select_nodes(
    xyz: torch.Tensor,
    t: torch.Tensor,
    edges: torch.Tensor,
    keep: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Rebuild a graph on the kept nodes (short-track output)."""
    idx = torch.where(keep)[0]
    remap = torch.full((keep.size(0),), -1, dtype=torch.long, device=xyz.device)
    remap[idx] = torch.arange(idx.size(0), device=xyz.device)
    if edges.numel():
        mapped = remap[edges]
        ok = (mapped[:, 0] >= 0) & (mapped[:, 1] >= 0)
        edges = mapped[ok]
    else:
        edges = edges
    return xyz[keep], t[keep], edges


def kappa_link(
    dist_um: torch.Tensor,
    *,
    s_i: torch.Tensor | None = None,
    s_j: torch.Tensor | None = None,
    codon_t: torch.Tensor | None = None,
    codon_tp: torch.Tensor | None = None,
    patch: torch.Tensor | None = None,
    peak_at_um: torch.Tensor | float,
    match_um: float = MATCH_UM,
) -> torch.Tensor:
    """Quantum bleed κ on a pair, with Fluid spatial prior.

    ident_S = 1 / (1 + |Si−Sj|/Θ)     collapse: same key if S agrees
    ident_C = codon trit agreement      Genetics local what
    spat    = 1 / (1 + |d − peak|/σ)
    ΔD      = 25 · d / (φ·match)
    κ       = A_bleed · POOF · ident_S · ident_C · spat / (1 + ΔD/25)
    """
    sig = sigma_um(match_um)
    ident = dist_um.new_ones(dist_um.shape)
    if s_i is not None and s_j is not None:
        ident = ident * (1.0 / (1.0 + (s_i - s_j).abs() / COLLAPSE_THRESHOLD))
    if codon_t is not None and codon_tp is not None:
        ident = ident * codon_ident(codon_t, codon_tp)
    if patch is not None:
        ident = ident * patch
    spat = 1.0 / (1.0 + (dist_um - peak_at_um).abs() / sig)
    delta_d = COMPACTIFICATION_CEILING * dist_um / max_link_um_fsot(match_um)
    return SEEDS.a_bleed * SEEDS.poof * ident * spat / (1.0 + delta_d / COMPACTIFICATION_CEILING)


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


def link_fsot(
    xyz_t: torch.Tensor,
    xyz_tp: torch.Tensor,
    *,
    max_um: float,
    yx_um: float,
    z_um: float,
    s_t: torch.Tensor | None = None,
    s_tp: torch.Tensor | None = None,
    codon_t: torch.Tensor | None = None,
    codon_tp: torch.Tensor | None = None,
    patch: torch.Tensor | None = None,
    vel_t: torch.Tensor | None = None,
    match_um: float = MATCH_UM,
) -> tuple[torch.Tensor, torch.Tensor]:
    """One-to-one t→t+1 by bleed κ, not nearest Euclidean.

    Spatial term is a soft NN (peaks at 0). Identity is collapse
    ``1/(1+|Si−Sj|/Θ)``. Product lets a same-S partner beat a slightly
    closer speck. Inertia (Fluid) is optional: it followed wrong tracks
    on 0113de3b and is off by default.

    Returns (E,2) pairs and (M,3) velocity at t+1 (pixels, 0 if unmatched).
    """
    n, m = int(xyz_t.size(0)), int(xyz_tp.size(0))
    zero_pairs = torch.zeros(0, 2, dtype=torch.long, device=xyz_t.device)
    vel_tp = xyz_tp.new_zeros(m, 3)
    if n == 0 or m == 0:
        return zero_pairs, vel_tp

    d_cur = dist_um(xyz_t, xyz_tp, yx_um=yx_um, z_um=z_um)
    d_use = d_cur
    peak_at: torch.Tensor | float = d_cur.new_zeros(())
    if vel_t is not None:
        pred = xyz_t + vel_t
        d_pred = dist_um(pred, xyz_tp, yx_um=yx_um, z_um=z_um)
        has = (vel_t.pow(2).sum(dim=-1, keepdim=True) > 0).to(d_cur.dtype)
        d_use = has * d_pred + (1.0 - has) * d_cur
        peak_at = (1.0 - has) * 0.0
    s_i = s_t[:, None] if s_t is not None else None
    s_j = s_tp[None, :] if s_tp is not None else None
    kappa = kappa_link(
        d_use, s_i=s_i, s_j=s_j, codon_t=codon_t, codon_tp=codon_tp,
        patch=patch, peak_at_um=peak_at, match_um=match_um,
    )
    kappa = kappa.masked_fill(d_cur > max_um, -1.0)
    pairs: list[tuple[int, int]] = []
    if patch is not None:
        # NN first. Patch may override only when the nearest neighbor
        # looks unlike (NCC < 1/φ) — speck steal on easy videos — so
        # similar cells inside 7 µm keep NN (0c582fdc).
        spat = (max_um - d_cur).clone()
        spat = spat.masked_fill(d_cur > max_um, -1.0)
        nn_pairs = _greedy_pairs(spat)
        used_t = torch.zeros(n, dtype=torch.bool, device=d_cur.device)
        used_p = torch.zeros(m, dtype=torch.bool, device=d_cur.device)
        weak: list[int] = []
        gate = COLLAPSE_THRESHOLD
        for i, j in nn_pairs:
            ncc = float(patch[i, j]) if d_cur[i, j] <= max_um else 0.0
            if ncc >= gate:
                used_t[i] = True
                used_p[j] = True
                pairs.append((i, j))
            else:
                weak.append(i)
        if weak:
            lock = patch.clone()
            lock = lock.masked_fill(d_cur > match_um, -1.0)
            lock = lock.masked_fill(used_p[None, :], -1.0)
            weak_mask = torch.ones(n, dtype=torch.bool, device=d_cur.device)
            weak_mask[torch.tensor(weak, device=d_cur.device)] = False
            lock = lock.masked_fill(weak_mask[:, None], -1.0)
            for i, j in _greedy_pairs(lock):
                used_t[i] = True
                used_p[j] = True
                pairs.append((i, j))
        rest = kappa.masked_fill(used_t[:, None], -1.0).masked_fill(used_p[None, :], -1.0)
        pairs.extend(_greedy_pairs(rest))
    else:
        pairs = _greedy_pairs(kappa)
        pair_t = torch.tensor(pairs, dtype=torch.long, device=xyz_t.device) if pairs else zero_pairs
        if pair_t.numel():
            pair_t = consensus_swaps(kappa, pair_t)
        vel_tp = xyz_tp.new_zeros(m, 3)
        if pair_t.numel():
            vel_tp[pair_t[:, 1]] = xyz_tp[pair_t[:, 1]] - xyz_t[pair_t[:, 0]]
        return pair_t, vel_tp
    if not pairs:
        return zero_pairs, vel_tp
    pair_t = torch.tensor(pairs, dtype=torch.long, device=xyz_t.device)
    vel_tp = xyz_tp.new_zeros(m, 3)
    vel_tp[pair_t[:, 1]] = xyz_tp[pair_t[:, 1]] - xyz_t[pair_t[:, 0]]
    return pair_t, vel_tp


def consensus_swaps(kappa: torch.Tensor, pairs: torch.Tensor, *, rounds: int = 8) -> torch.Tensor:
    """Raise total κ by swapping two edges. Discrete consensus, no softmax."""
    e = int(pairs.size(0))
    if e < 2:
        return pairs
    src = pairs[:, 0].tolist()
    dst = pairs[:, 1].tolist()
    klist = kappa.detach().cpu().tolist()
    for _ in range(rounds):
        moved = False
        for a in range(e):
            ia, ja = src[a], dst[a]
            row_a = klist[ia]
            for b in range(a + 1, e):
                ib, jb = src[b], dst[b]
                cur = row_a[ja] + klist[ib][jb]
                alt_ajb, alt_ibja = row_a[jb], klist[ib][ja]
                if alt_ajb + alt_ibja > cur + 1e-9 and alt_ajb >= 0.0 and alt_ibja >= 0.0:
                    dst[a], dst[b] = jb, ja
                    moved = True
                    break
            if moved:
                break
        if not moved:
            break
    return torch.tensor(list(zip(src, dst)), dtype=torch.long, device=pairs.device)


def link_times(
    xyz_by_t: dict[int, torch.Tensor],
    *,
    max_um: float,
    yx_um: float,
    z_um: float,
    score_by_t: dict[int, torch.Tensor] | None = None,
    match_um: float = MATCH_UM,
) -> torch.Tensor:
    """Chain FSOT links over integer times. Returns global (E,2) into concatenated nodes."""
    times = sorted(xyz_by_t)
    offset: dict[int, int] = {}
    n = 0
    for t in times:
        offset[t] = n
        n += int(xyz_by_t[t].size(0))
    vel = None
    edges: list[torch.Tensor] = []
    for t, tp in zip(times, times[1:]):
        a, b = xyz_by_t[t], xyz_by_t[tp]
        s_t = score_by_t[t] if score_by_t is not None else None
        s_tp = score_by_t[tp] if score_by_t is not None else None
        # Times may skip; only carry velocity across dt==1.
        use_vel = vel if (tp == t + 1) else None
        pairs, vel_next = link_fsot(
            a, b, max_um=max_um, yx_um=yx_um, z_um=z_um,
            s_t=s_t, s_tp=s_tp, vel_t=use_vel, match_um=match_um,
        )
        vel = vel_next if tp == t + 1 else None
        if pairs.numel() == 0:
            continue
        edges.append(torch.stack([pairs[:, 0] + offset[t], pairs[:, 1] + offset[tp]], dim=1))
    if not edges:
        return torch.zeros(0, 2, dtype=torch.long)
    return torch.cat(edges, dim=0)


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
