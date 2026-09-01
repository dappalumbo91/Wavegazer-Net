# Observed outcomes of U-Net (the control numbers)

These are published, citable results of the **standard architecture**, not
numbers we have reproduced yet. Reproduction of a subset is a later step.
Until then, treat this table as the observable base point.

## 1. Original paper (Ronneberger et al., MICCAI 2015)

Source: `docs/papers/1505.04597_unet.pdf`, Tables 1–2, §4–5.

### ISBI EM neuronal membranes (Drosophila VNC, 30 training images 512²)

Ranking as of 6 March 2015, sorted by warping error:

| Method | Warping error | Rand error | Pixel error |
|--------|---------------|------------|-------------|
| U-Net (7-rotation TTA, no extra post-process) | **0.0003529** | 0.0382 | 0.0611 |
| Ciresan et al. sliding-window CNN | 0.000420 | 0.0504 | — |
| Human (where reported in the challenge) | ~0.000005 | ~0.0021 | ~0.0010 |

U-Net beat the previous CNN on warping error without dataset-specific
post-processing. Some later entries beat it on Rand error by stacking heavy
post-process on Ciresan’s probability map (the paper notes 78 submissions
from one group).

### ISBI cell tracking 2015 — segmentation IOU

| Method | PhC-U373 | DIC-HeLa |
|--------|----------|----------|
| U-Net | **0.9203** | **0.7756** |
| Second best 2015 | 0.83 | 0.46 |
| KTH-SE (2014) | 0.7953 | 0.4607 |
| IMCB-SG (2014) | 0.2669 | 0.2935 |

Training set sizes: 35 partially annotated frames (PhC-U373), 20 (DIC-HeLa).
That is the “few images” result people still cite.

### Speed / compute (2015 hardware)

- Train: ~10 hours on an NVIDIA Titan (6 GB)
- Infer: a 512×512 image in **less than one second** on a then-recent GPU

## 2. What the community observed after that

These are the reasons U-Net became the default visual segmentation net,
not extra architecture tricks.

| Observation | Evidence |
|-------------|----------|
| The U generalizes past microscopy | Same topology, new data: CT/MRI organs, satellite, industrial defects, driving (Carvana masks). |
| Architecture tweaks often do not beat a tuned vanilla U | nnU-Net (Isensee et al.): residual / dense / attention add-ons did not help once preprocessing, loss, and augmentation were fixed. `docs/papers/1809.10486_nnunet.pdf` |
| Loss + data pipeline move the number more than a new block | Switching CE → Dice+CE, adding elastic/gamma augment, and overlap-window inference are the usual jumps. |
| Skip concat is the load-bearing idea | Ablations that drop skips lose thin structures. Additive (ResNet-style) skips are a different inductive bias and need matched channel counts. |
| Same-padding is an engineering win, not a scientific one | Output aligned to input simplifies training code. Border accuracy is recovered by overlap-tile / sliding window, which the paper already had. |

## 3. nnU-Net on the Medical Segmentation Decathlon (the modern standard)

Source: Isensee et al., workshop paper 2018 (`1809.10486`) and Nature Methods
2021. Mean **Dice** on phase-1 cross-validation (Table 2 of the workshop paper).
Bold = used for their test submission. Test-set Dice in the last row.

| Task (label) | 2D U-Net | 3D U-Net | Cascade | Test set |
|--------------|----------|----------|---------|----------|
| BrainTumour 1 / 2 / 3 | 78.60 / 58.65 / 77.42 | 80.71 / 62.22 / 79.07 | — | 67.71 / 47.73 / 68.16 |
| Heart | 91.36 | 92.45 | 92.40 | **92.77** |
| Liver 1 / 2 | 94.37 / 53.94 | 94.11 / 61.74 | 95.38 / 58.49 | **95.24 / 73.71** |
| Hippocampus 1 / 2 | 88.52 / 86.70 | 89.87 / 88.20 | — | **90.37 / 88.95** |
| Prostate 1 / 2 | 61.98 / 84.31 | 60.77 / 83.73 | — | **75.81 / 89.59** |
| Lung | 52.68 | 55.87 | 66.85 | **69.20** |
| Pancreas 1 / 2 | 74.70 / 35.41 | 77.69 / 42.69 | 79.30 / 52.12 | **79.53 / 52.27** |

At submission they held the **highest mean Dice** on the public leaderboard for
all classes of all seven phase-1 tasks except BrainTumour class 1.

Read this as: a plain U-Net, configured automatically, is the thing to beat
on medical 3D. A new formula that only wins on a toy 2D set has not yet
shown U-Net-equivalence.

## 4. A widely copied 2D open-source run (milesial/Pytorch-UNet)

Source: `vendor/Pytorch-UNet/README.md`.

- Task: Kaggle Carvana image masking (cars, high-res RGB)
- Train: ~5k images from scratch
- Held-out Dice: **0.988423** on >100k test images
- `torch.hub` pretrained: `unet_carvana`

This is **not** biomedical and **not** the paper split. It is useful as a
sanity check that the same 64–1024 U, with padding and BatchNorm, trains
cleanly in PyTorch.

## 5. Failure modes the literature keeps seeing

Record these so a later equivalent is not declared better after hiding them.

1. **Thin structures vanish** if skips are weak or downsampling is too aggressive
   (membranes, small vessels, cell borders).
2. **Touching instances merge** unless the loss (weight map, or a separate
   watershed / instance head) punishes the gap.
3. **Anisotropic 3D** (thick slices): naive 3D U-Net can lose to a 2D U-Net
   on the high-resolution plane (nnU-Net discussion of Prostate).
4. **Domain shift**: intensity, resolution, and scanner change drop Dice
   far more than swapping ReLU for GELU.
5. **Overfitting the architecture to one dataset**: the Decathlon exists
   specifically because of this.

## 6. What we will measure here, on this machine

After `scripts/dump_baseline.py`:

- parameter count of BaselineUNet (modern and paper)
- encoder feature-map shapes
- a dummy forward Dice/IoU (random weights — **not** a performance claim)
- GPU identity (RTX 5070 / sm_120)

The first **trained** number on a real dataset gets appended to
`artifacts/baseline_freeze.json` under a new key, with the dataset name, split
seed, epoch count, and augmentations. No trained number exists yet.
