# Competitor scoreboard

Wavegazer is supposed to compete with other visual models, not only pass
self-tests. This file names the metrics, the floors, and the order of
gates. Do not quote a later gate as won if an earlier one is red.

Authority pin **D1D38A**. Zero trainable weights on the seed spine.

## Versus the competition (plain)

The Biohub/CellMot prize is **not** Dice and **not** 7 µm F1 on a 2D slice.

```
pixels → peaks → centroids (7 µm, 3D anisotropic)
      → edges + ILP → adj_edge_jaccard + 0.1×division
```

| Rung | Metric | Wavegazer now | Competition |
|------|--------|---------------|-------------|
| Detect, 2D YX @ 7 µm | recall | **0.92** (tied φ-DoG) | CellMot U-Net ~**0.98** on thr0.99 |
| Detect, 3D 7 µm | recall | **1.00** (16 vol, σ NMS) | same 7 µm rule, all Z |
| Density vs `T_true` | nodes / frame | **~968** vs **~258** | U-Net thr0.99 sits near `T_true` |
| Top-`T_true` prune | 3D recall | **0.66** (score rank) | trained peak head keeps GT in budget |
| Linking | edge Jaccard | patch-NCC lock **0.772** best / **0.600** mean (4 videos) | transformer edges + ILP |
| Official-shaped | adj_edge_jaccard | **0.767** best / **0.587** mean (4 videos) | floor **0.848**, top **~0.985** |
| Local hard5 (CellMot U-Net) | same official metric | — | baseline 0.71, FT+short-track **0.74** |

**Where we sit:** detect box is green. 3D anisotropic 7 µm recall is **1.0** on
16 train volumes after NMS at σ (`MATCH_UM/π`), not at the match radius.
That matches CellMot’s reported detect recall (~0.98). It is still **not**
a 0.848-class number.

**Where we have to go:**

1. **Keep 7 µm cell NMS for linking.** σ NMS is the detect-recall setting
   (1.0, ~968/frame, density would cap adj at ~0.72). Full-video 7 µm NMS
   sat at 27534 vs `T_true` 25755 on `44b6_0113de3b` — density is fine.
   Seed rankers and Z-focus cannot prune σ-NMS extras without losing GT.
2. **Linking (still the gap to 0.848).** Luma-patch NCC lock inside 7 µm
   is the first identity that moves the official number: easy video
   **0.767** (was 0.555 NN). 4-video mean **0.587** vs NN **0.529**.
   Not 0.848. Patch lock **loses to NN** on `0c582fdc`. Remaining 6 FN /
   7 FP on the easy video: J=44/57=0.772; killing those 7 FP with no TP
   loss would be 0.88. We cannot tell the 7 FP from TP without a stronger
   appearance metric. Divisions not scored.
3. **Then** public LB vs 0.848 / 0.985.

Do not quote 1.0 detect recall as “near 0.848”. Different units.

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

**48 Biohub volumes**, median-z ± 1 **plus Z-max**, multi-scale φ-DoG, YX 7 µm,
artifact `biohub_peaks_7um.json`.

Peak *locations* come from φ-DoG (S only re-ranks), so S-specific misses
closed. Z-max recovered off-plane cells (`144b256d` 0 → 0.5).

| Method | Recall | Precision | F1 | n_pred / n_gt |
|--------|--------|-----------|----|----------------|
| φ-DoG intensity | **0.921** | 0.042 | 0.078 | 95.8 / 4.0 |
| Wavegazer (DoG peaks, S rank) | **0.921** | 0.042 | 0.078 | 95.9 / 4.0 |

Gate 1: **matched at 0.92 recall**. Extra peaks (precision 0.04) are expected
on sparse GT. Not a CellMot 0.848 result.

## Live 3D detect (2026-09-09)

**16 Biohub volumes**, all 64 Z at the busiest t, anisotropic 7 µm
(YX 0.40625, Z 1.625). Artifact `biohub_peaks_3d_7um.json`.

NMS at the **match** radius (7 µm) merged a true cell with a brighter
neighbor 9 µm away (`44b6_2f31fc2f`, recall 0.979). NMS at **σ**
(`MATCH_UM/π ≈ 2.23 µm`), same as 2D `detect()`, recovered it.

| Method | Recall | n_pred / frame | vs ~258 `T_true/T` |
|--------|--------|----------------|---------------------|
| 7 µm NMS (cell packing) | 0.979 | 390 | 1.5×, adj factor ~0.95 |
| σ NMS (same as 2D detect) | **1.000** | 968 | 3.8×, adj factor ~0.72 |
| + Z-focus then 7 µm NMS | 0.911 | 331 | closer density, **recall loss** |
| top `T_true/T` by blob or S | 0.62–0.66 | ~258 | S is not a CellMot ranker |

Detect gate: **green** at σ NMS. Z-focus and seed rankers do not recover
`T_true` without dropping labeled cells. Full-video linking used **7 µm
NMS** so density stays near `T_true`.

## Live linking (2026-09-09)

Luma-patch NCC lock inside 7 µm, then bleed κ. 7 µm cell NMS.
Artifact `biohub_track_nn.json`. No division term.

| Video | node rec | NN `J_adj` | FSOT `J_adj` | TP/FP/FN |
|-------|----------|------------:|-------------:|----------|
| `44b6_0113de3b` | 1.00 | 0.555 | **0.767** | 44/7/6 |
| `44b6_0b24845f` | 0.90 | 0.453 | **0.481** | 32/15/17 |
| `44b6_0c582fdc` | 0.97 | **0.591** | 0.542 | 53/25/17 |
| `44b6_0db75fae` | 1.00 | 0.519 | **0.557** | 116/52/35 |
| mean | 0.97 | 0.529 | **0.587** | |

Best video **0.767**. Floor **0.848** is **not beaten**. Patch NCC is the
first identity that actually moves assignments (easy video 0.555→0.767).
It **hurts** `0c582fdc` vs NN. Hard videos still miss nodes (0.90) or
over-detect (1.3× `T_true`). Snap-to-bright and φ⁴ short-track both
dropped labeled cells.

## Current artifacts

| File | Gate |
|------|------|
| `artifacts/synthetic_cells_compare.json` | dense Dice |
| `artifacts/biohub_compare.json` | dense disks on Z-max (proxy, class-imbalanced) |
| `artifacts/biohub_peaks_7um.json` | 2D YX detect |
| `artifacts/biohub_peaks_3d_7um.json` | **3D 7 µm detect** |
| `artifacts/biohub_track_nn.json` | FSOT bleed vs greedy NN linking |
