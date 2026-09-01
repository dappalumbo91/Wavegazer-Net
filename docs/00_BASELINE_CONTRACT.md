# Baseline contract

This file freezes what “the U-Net baseline” means in this repo so
WavegazerNet can be compared without moving the goalposts.

Nothing in this contract is the FSOT formula. It is the **control**.

## In / out contract (modern mode, the one we will train)

| Item | Frozen value |
|------|----------------|
| Input tensor | `(N, C, H, W)` with `H = W = 256` for the first comparison runs |
| Output tensor | `(N, K, H, W)` logits, **same spatial size** as input |
| Default `C` | 1 (grayscale microscopy / EM). RGB is `C = 3`. |
| Default `K` | 2 (background / foreground). Multi-class is allowed; report per-class Dice. |
| Topology | 4 downsamplings, 4 upsamplings, skip concat at every level |
| Channels | 64 → 128 → 256 → 512 → 1024 → 512 → 256 → 128 → 64 → `K` |
| Down operator | 2×2 max-pool, stride 2 |
| Up operator | 2×2 transposed convolution, stride 2 |
| Block | two 3×3 convolutions, each followed by ReLU (modern: BatchNorm before ReLU) |
| Skip | copy encoder feature map, concatenate on channel axis, no addition |
| Head | 1×1 convolution to `K` logits. Softmax is in the loss, not the net. |

Paper mode (`mode="paper"`) is kept as a second freeze: valid convolutions,
input 572×572 → output 388×388, no batch-norm. Use it to check paper-faithfulness,
not as the training default.

Measured on this machine (`artifacts/baseline_freeze.json`):

| Mode | Parameters | In | Out |
|------|------------|----|-----|
| modern | 31,035,586 | 256×256 | 256×256 |
| paper | 31,030,658 | 572×572 | 388×388 |

## Scoreboard (what “better” means)

Report all four. Do not quote a win from one number.

| Metric | Definition | Baseline published number |
|--------|------------|---------------------------|
| Dice (foreground) | `2 TP / (2 TP + FP + FN)` | not in the 2015 paper; modern default |
| IoU / Jaccard | `TP / (TP + FP + FN)` | PhC-U373 **0.9203**, DIC-HeLa **0.7756** |
| Pixel error | fraction of wrong argmax pixels | ISBI EM **0.0611** |
| Warping error | ISBI EM topology metric | **0.0003529** (needs the EM challenge scorer) |

A new architecture is an equivalent (or better) U-Net only if, on the **same
split of the same dataset, same augmentations, same epoch budget**, it matches
or beats the modern-mode BaselineUNet on Dice and IoU.

## What is allowed to change later

WavegazerNet names its replacements in `docs/06_FSOT_COMPONENT_MAP.md`.
If you change the candidate, name the replacement explicitly:

1. Which operator changes (conv, pool, upsample, skip, nonlinearity, loss).
2. Which tensor ranks/shapes stay identical so the rest of the net still runs.
3. Which scoreboard numbers will be re-measured.

Until then, do not “improve” the baseline. Freeze it.

## Machine freeze (this workstation)

Recorded when `scripts/dump_baseline.py` is run. Expected:

- GPU: NVIDIA GeForce RTX 5070, 12 GB, compute sm_120
- PyTorch: 2.9.1+cu128 (or newer cu128/cu130 wheel with sm_120 kernels)
- Python: 3.11
