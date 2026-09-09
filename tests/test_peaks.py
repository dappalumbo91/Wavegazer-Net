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
