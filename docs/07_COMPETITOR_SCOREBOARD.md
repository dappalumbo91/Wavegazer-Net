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
| Linking | edge Jaccard | FSOT bleed **0.619** / **0.500** (2 videos) vs NN 0.559 / 0.471 | transformer edges + ILP |
| Official-shaped | adj_edge_jaccard | FSOT **0.615** / **0.481** (mean **0.548**) | floor **0.848**, top **~0.985** |
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
2. **Linking (still the gap to 0.848).** Bleed κ
   (`A_bleed·POOF·ident/(1+ΔD/25)`, ident = collapse `1/(1+|ΔS|/Θ)`,
   search `φ·7` µm) beats greedy NN: **0.615 vs 0.555** and **0.481 vs
   0.453** adj on two videos. GT-only NN is already **1.0** — extras steal
   the 11–17 FN. Identity cannot outrank a 2 µm speck vs a 7 µm true
   partner. Golden-step + inertia followed wrong tracks (J dropped to
   0.37). Next: codon-patch identity or an ILP on these κ costs, not a
   tighter DoG. Divisions not scored (0.1×).
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

Full videos, 7 µm cell NMS. Artifact `biohub_track_nn.json`.
No transformer+ILP. No division term.

| Video | linker | node rec | TP/FP/FN | J | T_pred / T_true | J_adj |
|-------|--------|----------|----------|---|-----------------|-------|
| `44b6_0113de3b` | greedy NN `π·7` | 1.00 | 38/18/12 | 0.559 | 27534/25755 | 0.555 |
| `44b6_0113de3b` | **FSOT bleed `φ·7`** | 1.00 | 39/13/11 | **0.619** | 27534/25755 | **0.615** |
| `44b6_0b24845f` | greedy NN | 0.90 | 32/19/17 | 0.471 | 45136/32795 | 0.453 |
| `44b6_0b24845f` | **FSOT bleed** | 0.90 | 32/15/17 | **0.500** | 45136/32795 | **0.481** |

Mean FSOT `adj_edge_jaccard` **0.548** vs NN **0.504**. Floor **0.848**.
The lift is real and on-spine (Quantum κ + collapse ident). Remaining FN
are nearer extras; remaining density miss on the second video is 1.38×
`T_true`.

## Current artifacts

| File | Gate |
|------|------|
| `artifacts/synthetic_cells_compare.json` | dense Dice |
| `artifacts/biohub_compare.json` | dense disks on Z-max (proxy, class-imbalanced) |
| `artifacts/biohub_peaks_7um.json` | 2D YX detect |
| `artifacts/biohub_peaks_3d_7um.json` | **3D 7 µm detect** |
| `artifacts/biohub_track_nn.json` | FSOT bleed vs greedy NN linking |
