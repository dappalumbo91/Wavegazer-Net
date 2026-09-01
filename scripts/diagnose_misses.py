"""Inspect GT pixels on volumes where Wavegazer recall lags φ-DoG."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from wavegazer.blob import DEFAULT_YX_UM, MATCH_UM, multi_scale_blob_map, sigma_px
from wavegazer.peaks import detect_gate, local_maxima, match_xy, nms
from wavegazer.wavegazer_net import WavegazerNet

spec = importlib.util.spec_from_file_location("cmp", ROOT / "scripts" / "compare_biohub_peaks.py")
cmp = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cmp)

MISSES = ["44b6_18ced818", "44b6_12dfb391", "44b6_abf82518", "44b6_144b256d"]


def main() -> None:
    net = WavegazerNet(1, 2, sparse=True)
    um = DEFAULT_YX_UM
    maxd = MATCH_UM / um
    sig = sigma_px(um, MATCH_UM)
    window = max(int(2 * sig) | 1, 3)
    root = Path(r"D:\Kaggle_Biohub_Data\train")
    for name in MISSES:
        zp = root / f"{name}.zarr"
        packed = cmp._plane_and_gt(zp, zp.with_suffix(".geff"))
        if packed is None:
            print(name, "no pack")
            continue
        planes, gt, meta = packed
        print("\n==", name, meta)
        for i, img in enumerate(planes):
            blobs, s_map = net.blob_field(img, um_per_px=um)
            dog_p = nms(local_maxima(blobs, window=window, min_score=detect_gate(blobs)), sig)
            s_p = net.detect(img, um_per_px=um)
            d_dog = match_xy(dog_p.xy, gt, maxd)
            d_s = match_xy(s_p.xy, gt, maxd)
            print(
                f"  plane{i} dog_rec={d_dog['recall']:.2f} s_rec={d_s['recall']:.2f} "
                f"dog_n={d_dog['n_pred']} s_n={d_s['n_pred']}"
            )


if __name__ == "__main__":
    main()
