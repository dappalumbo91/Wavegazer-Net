# How a U-Net is built and why it is shaped like a U

Primary source: Ronneberger, Fischer, Brox, *U-Net: Convolutional Networks for
Biomedical Image Segmentation*, MICCAI 2015, [arXiv:1505.04597](https://arxiv.org/abs/1505.04597).
PDF: `docs/papers/1505.04597_unet.pdf`.

Open-source systems this document is read against:

- `vendor/Pytorch-UNet` — milesial, the common 2D PyTorch copy.
- `vendor/dynamic-network-architectures` — MIC-DKFZ `PlainConvUNet`, the
  architecture nnU-Net actually instantiates.

## The problem it solves

A classification CNN maps an image to **one** label. Biomedical work needs a
label **per pixel** (a segmentation map), often from tens of annotated images,
not a million.

The 2012 answer (Ciresan et al.) ran a small net on every overlapping patch.
That localizes, but it is slow (redundant patches) and it forces a bad trade:
big patches give context and destroy localization via pooling; small patches
keep edges and cannot see context.

U-Net’s answer: one fully convolutional net that does both, in one forward pass.

## Two paths, one U

```
input image
    │
    ▼
 contracting path  ──────────────── skip ─────────────►  expansive path
 (encoder)                                               (decoder)
    │                                                          │
    │  space ↓  channels ↑                                     │  space ↑  channels ↓
    ▼                                                          ▼
              bottleneck (lowest resolution, widest features)
```

**Contracting path.** Ordinary CNN. Each step:

1. two unpadded 3×3 convolutions, each followed by ReLU
2. 2×2 max-pool, stride 2 (downsample)
3. double the number of feature channels

This path is “what is here?” — membranes, nuclei, texture — at growing context.

**Expansive path.** Symmetric reverse. Each step:

1. upsample by 2×2 transposed convolution (“up-convolution”), halve channels
2. concatenate the matching encoder feature map (the skip), after cropping
   it so the spatial sizes match (valid convs eat a border)
3. two 3×3 convolutions + ReLU

This path is “where is the boundary?” — it puts the coarse semantics back onto
a high-resolution grid.

**Head.** A 1×1 convolution maps the last 64-channel vector at each pixel to
`K` class logits.

**No fully connected layers.** Every operation is spatial. Arbitrary-size images
are tiled (overlap-tile, paper Figure 2). Missing border context is mirrored.

Total: **23 convolutional layers** (9 double-conv blocks = 18, plus 4
transposed convs, plus the 1×1 head).

## Skip connections, precisely

A skip is **not** a ResNet add. It is:

```
z_ℓ = concat( crop(encoder_ℓ), upsample(decoder_{ℓ+1}) )   along channels
```

If the encoder map is `C_e × H × W` and the upsampled decoder map is
`C_d × H × W`, the concat is `(C_e + C_d) × H × W`. The following two
convolutions learn how to mix “sharp local edges” with “coarse class identity”.

That is the architectural claim: pooling throws away location; the skip puts
location back **without** forcing the bottleneck to remember every edge.

## Channel / size schedule (paper figure 1)

Valid 3×3 conv reduces each spatial axis by 2, so two convs cost 4 pixels.

| Stage | After the two convs | Channels |
|-------|---------------------|----------|
| input | 572 × 572 | C |
| L0 encoder | 568 × 568 | 64 |
| L1 encoder | 280 × 280 | 128 |
| L2 encoder | 136 × 136 | 256 |
| L3 encoder | 64 × 64 | 512 |
| bottleneck | 28 × 28 | 1024 |
| L3 decoder | 52 × 52 | 512 |
| L2 decoder | 100 × 100 | 256 |
| L1 decoder | 196 × 196 | 128 |
| L0 decoder | 388 × 388 | 64 |
| 1×1 head | 388 × 388 | K |

Tiling constraint from the paper: every 2×2 max-pool must see an even height
and even width.

## What open source actually ships

Almost nobody trains the valid-conv paper net anymore.

| Design choice | 2015 paper | milesial Pytorch-UNet | nnU-Net PlainConvUNet |
|---------------|------------|------------------------|------------------------|
| Padding | valid (0) | same (1) | same `(k-1)//2` |
| Input = output size | no (572→388) | yes | yes |
| Normalization | none | BatchNorm | InstanceNorm |
| Nonlinearity | ReLU | ReLU | Leaky ReLU (slope 1e-2) |
| Upsample | 2×2 transposed conv | transposed conv (default) or bilinear | matching transposed conv |
| Skip | crop + concat | pad upsampled + concat | concat (sizes already match) |
| Deep supervision | no | no | optional extra 1×1 heads at decoder stages |
| Dimensionality | 2D | 2D | 2D or 3D from the same code |

Our `src/wavegazer/baseline_unet.py` implements **paper** and **modern**.
Modern is the training default. Paper is the faithfulness check.

## Lineage (so “U-Net equivalent” is well-defined)

1. **FCN** (Long, Shelhamer, Darrell 2015, `docs/papers/1411.4038_fcn.pdf`) —
   replace fully connected layers with convs, upsample a coarse score map,
   add skips that are **fused into the score**, not into a deep decoder.
2. **U-Net** (2015) — make the decoder as deep as the encoder and concat
   features, not scores. This is the U.
3. **3D U-Net** (Çiçek et al. 2016, `docs/papers/1606.06650_3d_unet.pdf`) —
   the same U with 3×3×3 convs and 2×2×2 pools, trained from sparse slices.
4. **V-Net** (Milletari et al. 2016) — 3D, residual blocks inside the U,
   Dice loss.
5. **nnU-Net** (Isensee et al. 2018/2021, `docs/papers/1809.10486_nnunet.pdf`)
   — keep the vanilla U, automate everything around it. Empirically, that
   beats most architectural “improvements” when the baseline is fully tuned.

A Wavegazer Net equivalent should say whether it is replacing (2), extending (3),
or replacing the operators inside the same U topology.

## Schematic

- Architecture: `docs/schematics/unet_canonical.svg`
- Train/infer pipeline: `docs/schematics/unet_pipeline.svg`
- Notes: `docs/05_SCHEMATIC.md`
