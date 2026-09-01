# Wavegazer Net

A visual segmentation network **equivalent to U-Net**, built from
Fluid Spacetime Omni-Theory (FSOT) instead of fitted convolution weights.

The working title was “Flow Net”. That name already belongs to FlowNet
(Dosovitskiy et al., ICCV 2015, optical flow). This project is **Wavegazer Net**.
See `docs/NAME.md`.

The control is a frozen canonical U-Net. The candidate is `WavegazerNet`: same
U-graph and tensor contract, every interior box replaced by a named FSOT
operator (codon mix, \(D_{\mathrm{eff}}\) fold, bleed skip, collapse head).
Zero trainable parameters. Pin **D1D38A**.

See `docs/00_BASELINE_CONTRACT.md` and `docs/06_FSOT_COMPONENT_MAP.md`.

## What is in here

| Path | Role |
|------|------|
| `docs/NAME.md` | Why the project is not called FlowNet |
| `docs/00_BASELINE_CONTRACT.md` | Frozen in/out shapes, metrics, comparison rules |
| `docs/01_ARCHITECTURE.md` | How a U-Net is built and why the U exists |
| `docs/02_MATHEMATICS.md` | Conv, pool, skip, softmax, weighted CE, Dice, He init |
| `docs/03_OBSERVED_OUTCOMES.md` | Published EM / cell-tracking / nnU-Net / Carvana numbers |
| `docs/04_STANDARD_BUILD.md` | 2015 paper recipe and the nnU-Net-era recipe |
| `docs/05_SCHEMATIC.md` | Architecture and pipeline drawings |
| `docs/schematics/` | SVG schematics (exact labels) |
| `docs/papers/` | U-Net, FCN, 3D U-Net, nnU-Net PDFs |
| `vendor/Pytorch-UNet/` | milesial 2D PyTorch U-Net (**GPL-3.0**, read only) |
| `vendor/dynamic-network-architectures/` | MIC-DKFZ PlainConvUNet used by nnU-Net (Apache-2.0) |
| `src/wavegazer/` | BaselineUNet (control) + WavegazerNet (FSOT) + losses/metrics |
| `docs/06_FSOT_COMPONENT_MAP.md` | Box-by-box replacement table |
| `docs/07_COMPETITOR_SCOREBOARD.md` | Field metrics and the order of floors to beat |
| `vendor/fsot/` | The six FSOT GitHub repos (Lean hub is the math authority) |
| `scripts/dump_baseline.py` | Writes `artifacts/baseline_freeze.json` |
| `scripts/dump_wavegazer.py` | Writes `artifacts/wavegazer_vs_unet.json` |
| `scripts/compare_synthetic.py` | Named synthetic cell split: train U-Net, run Wavegazer |

## Machine

This tree was stood up on:

- Windows, Python 3.11
- NVIDIA GeForce RTX 5070 (12 GB, sm_120)
- PyTorch **2.9.1+cu128** (Blackwell needs CUDA 12.8+ wheels)

## Setup

```powershell
cd "C:\Users\damia\Desktop\Wavegazer net"
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install torch==2.9.1+cu128 torchvision --index-url https://download.pytorch.org/whl/cu128
.\.venv\Scripts\python.exe -m pip install numpy pytest
```

## Check the baseline

```powershell
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe scripts\dump_baseline.py
.\.venv\Scripts\python.exe scripts\dump_wavegazer.py
.\.venv\Scripts\python.exe scripts\render_schematic.py
.\.venv\Scripts\python.exe scripts\compare_synthetic.py
.\.venv\Scripts\python.exe scripts\compare_biohub.py
.\.venv\Scripts\python.exe scripts\compare_biohub_peaks.py
```

`dump_baseline.py` records parameter counts, feature-map shapes, a dummy
(untrained) forward, and GPU identity. Dummy Dice is **not** a performance
claim. `compare_synthetic.py` is the first named split.

## Data on this machine

The Kaggle Biohub dump is on **D:**, not the mystery USB:

`D:\Kaggle_Biohub_Data\train` — 199 zarr+geff pairs, **~175 GB**.

`scripts/compare_biohub.py` scores a 2D Z-max + disk-around-centroid proxy
(not the official track metric). Use `WavegazerNet(..., sparse=True)` there.

## Next

If a residual on a named split is bad, do not add conv weights — change
the \(D_{\mathrm{eff}}\) ladder in `src/wavegazer/fsot_routes.py`.
Detect gate is live (`compare_biohub_peaks.py`): Wavegazer F1 0.164 vs
φ-DoG 0.139 @ 7 µm on 16 volumes. Next: raise recall without dumping
precision, then a trained U-Net peak head on the same frames.
