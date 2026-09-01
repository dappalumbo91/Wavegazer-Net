# Competitor scoreboard

Wavegazer is supposed to compete with other visual models, not only pass
self-tests. This file names the metrics, the floors, and the order of
gates. Do not quote a later gate as won if an earlier one is red.

Authority pin **D1D38A**. Zero trainable weights on the seed spine.

## Why not one number

There are two different hired jobs:

| Job | What the field scores | Our script |
|-----|----------------------|------------|
| **Dense segmentation** (U-Net paper) | Dice / IoU / pixel error | `compare_synthetic.py`, square unit test |
| **Cell detect** (Biohub / PIXEL_FIRST) | Bipartite centroids **≤ 7 µm**, then linking | `compare_biohub_peaks.py` |
| **Full Biohub track** | `adj_edge_jaccard + 0.1×div` | not yet (needs edges/ILP) |

Archive residual ≤ 0.5% on scientific panels does **not** transfer to these
(see `DISCONNECT_REVIEW.md` on the Desktop). We re-measure.

## Floors to beat (in order)

1. **Intensity / φ-DoG peaks** on the same Biohub frames, same NMS, same 7 µm.
   If Wavegazer loses this, the Optics S readout is not helping detect.
2. **Centroid recall @ 7 µm** high enough that linking can work. PIXEL_FIRST
   target: GT node recall @ 7 µm; precision not insane vs estimated node count.
3. **Trained U-Net detect** on the same split (CellMot-style peak head).
4. **Public CellMot full score ~0.848** (official baseline floor).
5. **Climb toward ~0.985** only after (2)–(4).

Dense Dice vs a *trained* U-Net on synthetic disks is a control, not the
Biohub prize.

## Detect protocol (locked)

- Data: `D:\Kaggle_Biohub_Data\train` (~175 GB, 199 volumes)
- Frame: busiest `t`, plane at median GT `z` (not Z-max)
- Match: YX Euclidean, **7 µm / 0.40625 µm·px⁻¹ ≈ 17.23 px**
- Wavegazer head: φ-DoG (`σ`, `φσ`) → Optics S → sparse gate `mean+φ·std` → NMS radius `σ`
- Intensity control: same DoG + NMS, **no** S
- Report: mean recall, precision, F1, n_gt, n_pred over N volumes

3D anisotropic match and adj_edge_jaccard come after this gate is green.

## Live detect numbers (2026-08-31)

16 Biohub volumes, 3-plane union (median z ± 1), multi-scale φ-DoG, YX 7 µm,
artifact `biohub_peaks_7um.json`:

| Method | Recall | Precision | F1 | n_pred / n_gt |
|--------|--------|-----------|----|----------------|
| φ-DoG intensity (control) | **0.803** | 0.078 | 0.132 | 41.6 / 3.5 |
| Wavegazer (DoG → Optics S → NMS) | 0.729 | 0.076 | 0.129 | 39.1 / 3.5 |

Recall climbed vs the previous single-plane run (0.67 → 0.73). The same
protocol lifted the intensity control further (0.67 → 0.80). S is not yet
winning gate 1. This is **not** a CellMot 0.848 result.

## Current artifacts

| File | Gate |
|------|------|
| `artifacts/synthetic_cells_compare.json` | dense Dice |
| `artifacts/biohub_compare.json` | dense disks on Z-max (proxy, class-imbalanced) |
| `artifacts/biohub_peaks_7um.json` | **this** detect gate |
