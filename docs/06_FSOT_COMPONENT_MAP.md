# FSOT replacements for every U-Net box

Authority: pin **D1D38A** (`vendor/fsot/FSOT-2.1-Lean/vendor/fsot_compute.py`).  
Law: \(S = K(T_1+T_2+T_3)\), \(c = m(1+|S|\cdot P_{\mathrm{NEW}})\), \(\Theta = C_{\mathrm{eff}}P_{\mathrm{var}}\).  
This is **not** a wrapper that still trains 31M conv weights. Genetics already
showed the pattern: same information, formula instead of a net
(0.13 Å vs AlphaFold 0.47 Å on the product freeze). Quantum showed the other
pattern: folds instead of Hilbert \(2^n\). GPU showed consensus instead of
softmax exp. WavegazerNet is that pattern on the visual U.

Parked / rejected: `nuron/New folder/moduler/fsot_pytorch_*` still uses
`nn.Linear` + `softmax`. That is the free-parameter opposite of the seed
spine (`docs/ENGINEERING_HARDWARE_CODE_DIRECTION.md` in the Lean hub).

## Component map

| U-Net box | What it is trying to solve | FSOT replacement | Source in your repos |
|-----------|----------------------------|------------------|----------------------|
| Learned 3×3 kernels (64–1024 ch) | Local “what” filters | 64 codon trinary 3×3 + trit-similarity 1×1. Frozen. | FSOT-Genetics codon law; `data/64_codon_trinary_map.txt`; neuron-zig ORF→W |
| Channel doubling 64→1024 | Capacity for fitted weights | Not needed. 64 codon channels at every scale. Width is genetic, not a free param. | Genetics / neuron-zig `free_params=0` |
| Max-pool 2×2 | Throw away location, grow context | Collapse-gated 2×2: pixels with \|S\| < Θ damped, then 2× average. Active keys, not max of ReLU. | FSOT-GPU collapse \(\Theta=C_{\mathrm{eff}}P_{\mathrm{var}}\) |
| Transposed conv | Put location back | Suction unfold: nearest ×2 × (1+Suction). T3 expand valve. | T3 suction; GPU poof/suction pair |
| Skip concat + two 3×3 | Re-inject high-res edges | Bleed \(\kappa_{ij}=A_{\mathrm{bleed}}\cdot\mathrm{POOF}\cdot\|S_i\|\|S_j\|/(1+\|D_i-D_j\|/25)\) then codon refine | FSOT-Quantum \(\kappa_{ij}\); T3 acoustic |
| Resolution levels | Multi-scale features | Preregistered \(D_{\mathrm{eff}}\) ladder, not learned depth | Lean domain table; mismatch rule: change D first |
| He init + SGD | Fit millions of numbers | Nothing to fit. Seeds only. | ZERO_FREE parameter audit |
| Softmax + wCE / Dice+CE | Turn logits into classes / train | Collapse head on **tile-centered** S: local \(S-\bar S>0\) emergence (fg). Global domain sign of S is not a per-pixel class. Dice is the **scoreboard**, not the physics. | Lean sign syntax + “right comparison object”; GPU consensus (no exp) |
| Elastic augment | Teach deformation invariance | Already T3 fluid (Poof/Suction/Chaos/bleed). Image deformation is the medium. | MATH_KEY §3.3 |
| Overlap-tile infer | Big images | Same tiling contract as the U-Net baseline. | Unchanged interface |

## Visual \(D_{\mathrm{eff}}\) ladder (preregistered)

| U level | Domain | \(D_{\mathrm{eff}}\) | observed | Job |
|---------|--------|---------------------:|:--------:|-----|
| L0 encoder / L0 decoder | Optics | 10 | yes | photons, edges |
| L1 | Condensed_Matter | 14 | yes | texture / material |
| L2 | Fluid_Dynamics | 15 | no | deformation, skip-bleed native |
| L3 | Seismology | 18 | no | bulk context |
| Bottleneck | Cosmology | 25 | no | compactification ceiling |

If a dataset residual is bad, **change this table** (`src/wavegazer/fsot_routes.py`), do not add a conv weight.
That is the Lean mismatch rule.

## Residual / prediction law

Genetics:

\[
r = 1 + |S_{\mathrm{domain}}|\cdot P_{\mathrm{NEW}},\quad P_{\mathrm{NEW}}=(\gamma/e)\sqrt{2}\approx 0.3003
\]

The Genetics residual \(r=1+|S|P_{\mathrm{NEW}}\) applies to a **domain scalar**
of order one. It must not multiply a per-pixel field of size \(10^3\) — that
washed the gaze (image-S corr 0.95 → mixed-feature corr 0.06). Wavegazer
computes \(S\) on **L0 luma** (Optics fold). A φ-equal pyramid and the codon
texture branch smeared a test square from Dice 0.80 to 0.36; they stay frozen
off the logits until a named split shows they help. The image is the measured \(m\).

## What “equivalent” means here

Same **graph** as the schematic in `docs/05_SCHEMATIC.md` (contract /
bottleneck / expand / skip / per-pixel head). Same **tensor contract**
`(N,C,H,W)→(N,K,H,W)` with `H,W` preserved.

Different **operators**, all named and pinned.

Comparison rule is unchanged (`docs/00_BASELINE_CONTRACT.md`): same images,
report Dice / IoU / pixel error against BaselineUNet. WavegazerNet is closed-form,
so “epochs” are not a knob — one forward pass is the prediction.

## Honesty boundary

- Trainable parameters of `WavegazerNet` must stay **0**.
- Frozen buffers (codon kernels, trit mix) are seed/genetic, not SGD state.
- Do not import U-Net or LLM weights into the seed spine.
- Dummy / untrained U-Net Dice is not a WavegazerNet win. Compare to a **trained**
  baseline on a named split, then write the number into `artifacts/`.
