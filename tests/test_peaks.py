import torch

from wavegazer.blob import blob_map, multi_scale_blob_map, phi_dog, sigma_px
from wavegazer.peaks import match_xy
from wavegazer.wavegazer_net import WavegazerNet


def _blob_image(h: int = 64, w: int = 64, cy: int = 32, cx: int = 32, r: float = 5.0):
    yy, xx = torch.meshgrid(torch.arange(h).float(), torch.arange(w).float(), indexing="ij")
    g = torch.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * r * r))
    return g[None, None]


def test_phi_dog_peaks_at_blob_center():
    img = _blob_image()
    dog = phi_dog(img, sigma=3.0)
    _, _, ys, xs = torch.where(dog == dog.max())
    assert abs(int(ys[0]) - 32) <= 2
    assert abs(int(xs[0]) - 32) <= 2


def test_detect_hits_synthetic_centroid():
    net = WavegazerNet(in_channels=1, n_classes=2, sparse=True)
    img = _blob_image()
    peaks = net.detect(img, um_per_px=0.40625, match_um=7.0)
    assert peaks.xy.size(0) >= 1
    xy = peaks.xy[peaks.score.argmax()]
    dist = float(torch.hypot(xy[0] - 32, xy[1] - 32))
    assert dist <= sigma_px(0.40625, 7.0)


def test_match_xyz_um_anisotropic():
    from wavegazer.peaks import match_xyz_um

    # 7 µm in Z at 1.625 µm/px is ~4.3 px; same 7 µm in YX is ~17 px.
    gt = torch.tensor([[10.0, 10.0, 5.0]])
    pred_hit = torch.tensor([[10.0, 10.0, 8.0]])  # 3 z * 1.625 = 4.875 µm
    pred_miss = torch.tensor([[10.0, 10.0, 15.0]])  # 10 z * 1.625 = 16.25 µm
    h = match_xyz_um(pred_hit, gt, 7.0, yx_um=0.40625, z_um=1.625)
    m = match_xyz_um(pred_miss, gt, 7.0, yx_um=0.40625, z_um=1.625)
    assert h["tp"] == 1
    assert m["tp"] == 0


def test_nms_sigma_keeps_peaks_inside_match_ball():
    """7 µm NMS merges a true cell with a brighter 5 µm neighbor; σ does not."""
    from wavegazer.blob import MATCH_UM, sigma_um
    from wavegazer.peaks import nms_xyz_um

    yx, z = 0.40625, 1.625
    dx = 5.0 / yx  # 5 µm in X
    xyz = torch.tensor([[0.0, 0.0, 0.0], [dx, 0.0, 0.0]])
    score = torch.tensor([1.0, 0.8])
    kept, _ = nms_xyz_um(xyz, score, sigma_um(MATCH_UM), yx_um=yx, z_um=z)
    merged, _ = nms_xyz_um(xyz, score, MATCH_UM, yx_um=yx, z_um=z)
    assert kept.size(0) == 2
    assert merged.size(0) == 1


def test_codon_codes_differ_on_two_blobs():
    from wavegazer.wavegazer_net import WavegazerNet

    net = WavegazerNet(1, 2, sparse=True)
    img = torch.zeros(4, 32, 32)
    img[1, 8, 8] = 1.0
    img[1, 24, 24] = 0.3
    xyz = torch.tensor([[8.0, 8.0, 1.0], [24.0, 24.0, 1.0]])
    codes = net.codon_codes(img, xyz)
    assert codes.size(0) == 2
    assert codes.size(1) == 64 * 3
    assert not torch.allclose(codes[0], codes[1])


def test_codon_ident_prefers_same_vector():
    from wavegazer.track import codon_ident

    torch.manual_seed(0)
    a = torch.randn(2, 64)
    same = codon_ident(a, a)
    scaled = codon_ident(a, a * 3.0)
    opp = codon_ident(a, -a)
    assert float(same[0, 0]) > float(scaled[0, 0])
    assert float(same[0, 0]) > float(opp[0, 0])


def test_patch_ncc_prefers_same_blob():
    from wavegazer.track import patch_ncc

    vol = torch.zeros(3, 32, 32)
    vol[1, 8:12, 8:12] = 1.0
    vol[2, 9:13, 9:13] = 1.0
    vol[2, 20:22, 20:22] = 0.4
    a = torch.tensor([[10.0, 10.0, 1.0]])
    b = torch.tensor([[11.0, 11.0, 2.0], [21.0, 21.0, 2.0]])
    ncc = patch_ncc(vol, a, vol, b, radius=5)
    assert float(ncc[0, 0]) > float(ncc[0, 1])


def test_link_fsot_overrides_weak_nn():
    """Keep NN when the nearest patch matches; override when it looks unlike."""
    from wavegazer.blob import MATCH_UM
    from wavegazer.fsot_seeds import SEEDS
    from wavegazer.track import link_fsot, link_nn

    yx, z = 0.40625, 1.625
    src = torch.tensor([[0.0, 0.0, 0.0]])
    true_xy = 5.0 / yx
    clut_xy = 2.0 / yx
    dst = torch.tensor([[true_xy, 0.0, 0.0], [clut_xy, 0.0, 0.0]])
    nn = link_nn(src, dst, max_um=MATCH_UM * SEEDS.pi, yx_um=yx, z_um=z)
    assert int(nn[0, 1]) == 1
    unlike = torch.tensor([[0.95, 0.2]])
    fs, _ = link_fsot(
        src, dst, max_um=MATCH_UM * SEEDS.pi, yx_um=yx, z_um=z, patch=unlike,
    )
    assert int(fs[0, 1]) == 0
    alike = torch.tensor([[0.5, 0.95]])
    fs2, _ = link_fsot(
        src, dst, max_um=MATCH_UM * SEEDS.pi, yx_um=yx, z_um=z, patch=alike,
    )
    assert int(fs2[0, 1]) == 1


def test_link_fsot_codon_beats_near_clutter():
    """Near speck with the opposite codon code loses to a farther same-code cell."""
    from wavegazer.blob import MATCH_UM
    from wavegazer.fsot_seeds import SEEDS
    from wavegazer.track import link_fsot, link_nn

    yx, z = 0.40625, 1.625
    src = torch.tensor([[0.0, 0.0, 0.0]])
    true_xy = 5.0 / yx
    clut_xy = 3.0 / yx
    dst = torch.tensor([[true_xy, 0.0, 0.0], [clut_xy, 0.0, 0.0]])
    code = torch.zeros(1, 64)
    code[0, 0] = 1.0
    code_tp = torch.zeros(2, 64)
    code_tp[0, 0] = 1.0
    code_tp[1, 0] = -1.0  # TTT-like vs AAA: PRIMARY agreement −1
    nn = link_nn(src, dst, max_um=MATCH_UM * SEEDS.pi, yx_um=yx, z_um=z)
    fs, _ = link_fsot(
        src, dst, max_um=MATCH_UM * SEEDS.pi, yx_um=yx, z_um=z,
        codon_t=code, codon_tp=code_tp,
    )
    assert int(nn[0, 1]) == 1
    assert int(fs[0, 1]) == 0


def test_filter_short_tracks_drops_specks():
    from wavegazer.track import filter_short_tracks, min_track_len, select_nodes

    # One long chain of 8 and a 2-node speck.
    edges = torch.tensor(
        [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [8, 9]],
    )
    xyz = torch.zeros(10, 3)
    t = torch.arange(10).float()
    keep, e2 = filter_short_tracks(10, edges, min_len=7)
    assert int(keep[:8].sum()) == 8
    assert int(keep[8:].sum()) == 0
    xyz2, t2, e3 = select_nodes(xyz, t, e2, keep)
    assert xyz2.size(0) == 8
    assert e3.size(0) == 7


def test_snap_xyz_moves_to_bright_pixel():
    from wavegazer.track import snap_xyz

    vol = torch.zeros(3, 8, 8)
    vol[1, 4, 5] = 1.0
    xyz = torch.tensor([[4.0, 4.0, 1.0]])
    out = snap_xyz(vol, xyz, xy_r=1, z_r=1)
    assert abs(float(out[0, 0]) - 5.0) < 0.2
    assert abs(float(out[0, 1]) - 4.0) < 0.2


def test_link_fsot_identity_beats_near_clutter():
    """Nearest Euclidean picks a 3 µm speck; bleed κ keeps the same-S cell at 5 µm."""
    from wavegazer.blob import MATCH_UM
    from wavegazer.fsot_seeds import SEEDS
    from wavegazer.track import link_fsot, link_nn

    yx, z = 0.40625, 1.625
    src = torch.tensor([[0.0, 0.0, 0.0]])
    true_xy = 5.0 / yx
    clut_xy = 3.0 / yx
    dst = torch.tensor([[true_xy, 0.0, 0.0], [clut_xy, 0.0, 0.0]])
    s_t = torch.tensor([1.0])
    s_tp = torch.tensor([1.0, 0.0])
    nn = link_nn(src, dst, max_um=MATCH_UM * SEEDS.pi, yx_um=yx, z_um=z)
    fs, _ = link_fsot(
        src, dst, max_um=MATCH_UM * SEEDS.pi, yx_um=yx, z_um=z,
        s_t=s_t, s_tp=s_tp,
    )
    assert int(nn[0, 1]) == 1
    assert int(fs[0, 1]) == 0


def test_link_nn_and_sparse_edge_jaccard():
    from wavegazer.track import link_nn, score_edges

    yx, z = 0.40625, 1.625
    # Two GT nodes, one edge. Pred has the same two plus a distractor.
    pred_xyz = torch.tensor(
        [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [80.0, 80.0, 10.0]],
    )
    pred_t = torch.tensor([0.0, 1.0, 0.0])
    gt_xyz = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    gt_t = torch.tensor([0.0, 1.0])
    gt_edges = torch.tensor([[0, 1]])
    xyz_t = pred_xyz[pred_t == 0]
    xyz_tp = pred_xyz[pred_t == 1]
    edges = link_nn(xyz_t, xyz_tp, max_um=7.0, yx_um=yx, z_um=z)
    # Map plane-local indices back to pred rows: t=0 rows 0 and 2, t=1 row 1.
    t_idx = torch.where(pred_t == 0)[0]
    p_idx = torch.where(pred_t == 1)[0]
    pred_edges = torch.stack([t_idx[edges[:, 0]], p_idx[edges[:, 1]]], dim=1)
    c = score_edges(
        pred_xyz, pred_t, pred_edges, gt_xyz, gt_t, gt_edges, yx_um=yx, z_um=z,
    )
    assert c.tp == 1
    assert c.fn == 0
    assert c.node_tp == 2
    assert c.edge_jaccard == 1.0


def test_match_xy_perfect():
    gt = torch.tensor([[10.0, 10.0], [40.0, 40.0]])
    pred = torch.tensor([[11.0, 9.0], [39.0, 41.0], [80.0, 80.0]])
    m = match_xy(pred, gt, max_dist_px=3.0)
    assert m["tp"] == 2
    assert m["recall"] == 1.0
    assert abs(m["precision"] - 2 / 3) < 1e-6


def test_blob_map_in_unit_interval():
    img = _blob_image()
    b = blob_map(img, sigma=3.0)
    assert float(b.min()) >= 0.0
    assert float(b.max()) <= 1.0 + 1e-5
    m = multi_scale_blob_map(img, sigma=3.0)
    assert float(m.max()) <= 1.0 + 1e-5
