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
