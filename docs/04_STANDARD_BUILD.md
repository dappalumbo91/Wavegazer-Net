# Standard way to build a U-Net (2015 recipe and 2021 practice)

Two recipes. Use the modern one to train. Use the paper one to check that
the topology is actually a U-Net.

## A. Paper recipe (Caffe, 2015)

Copied from arXiv:1505.04597 §§2–3.

1. **Input tile** large enough that every pool sees even H and W. Canonical:
   572×572. Prefer a big tile over a big batch (batch = 1).
2. **Encoder:** four times `{ two 3×3 valid conv + ReLU ; 2×2 max-pool }`,
   channels 64, 128, 256, 512, then bottleneck 1024.
3. **Decoder:** four times `{ 2×2 up-conv ; crop-concat skip ; two 3×3 ReLU }`.
4. **Head:** 1×1 conv to K classes.
5. **Init:** He normal, std \(\sqrt{2/N}\).
6. **Loss:** pixel softmax + weighted CE, \(w_0=10\), \(\sigma \approx 5\).
7. **Opt:** SGD, momentum 0.99.
8. **Augment:** elastic 3×3 grid, \(\sigma=10\) px, plus rotation/shift/gray.
9. **Dropout** at the bottleneck.
10. **Infer:** overlap-tile; mirror-pad the missing context. Optional 7-fold
    rotation TTA (used for the EM number).

That is the entire scientific method of the paper. There is no scheduler
magic, no BatchNorm, no Dice.

## B. Modern open-source recipe (what you should copy)

This is the intersection of milesial/Pytorch-UNet and nnU-Net / PlainConvUNet.
It is how U-Nets are built in 2026 if you are not inventing a new one.

### Architecture

- Same U, **same padding**, so `H_out = H_in`.
- Channels still 64–1024 (or 32–512 on small GPUs).
- Normalization: BatchNorm (2D natural-image) or InstanceNorm (3D medical,
  small batches).
- Nonlinearity: ReLU or leaky ReLU 0.01.
- Upsample: transposed conv (learned) unless checkerboard artifacts appear,
  then bilinear + 1×1.
- Optional: deep supervision (a 1×1 head at each decoder stage, losses
  weighted `1, 1/2, 1/4, …`).
- 2D vs 3D is a kernel-dimension switch, not a new idea
  (`nn.Conv2d` → `nn.Conv3d` in PlainConvUNet).

### Data

1. Crop to nonzero (MRI brain, etc.).
2. Resample to a consistent voxel spacing (median of the dataset).
3. Normalize: CT by dataset percentiles then z-score; MRI z-score per volume.
4. Train on patches that fit GPU memory, not necessarily whole slices.
5. Oversample patches that contain foreground (~33% of a batch in nnU-Net).

### Train

- Loss: `Dice + CE` (semantic). Keep the paper weight map only if you must
  split touching instances and you have instance IDs.
- Optimizer: SGD momentum 0.99, LR 0.01, poly decay, 1000 epochs of 250
  batches **or** Adam 3e-4 with a short cosine — pick one and freeze it.
- AMP (autocast) on CUDA. Gradient clip 12 (nnU-Net).
- Augment on the fly: flip, rotate, scale, elastic, gamma, blur, noise.
- 5-fold CV. Pick the fold-mean Dice, not a lucky seed.

### Infer

- Sliding window, overlap 50%.
- Weight the center of each patch higher than the border (Gaussian).
- Test-time flip (and optionally rotation).
- Ensemble the 5 folds if you need the last 0.5 Dice.
- Post-process only if the training labels justify it (e.g. “there is always
  one connected liver” → drop extra components).

### Hardware on this box

RTX 5070, 12 GB, sm_120. Practical starting points:

| Setup | Patch | Batch | Notes |
|-------|-------|-------|-------|
| 2D modern, 1–3 ch, 2 classes | 256² or 512² | 8–16 | 512² is comfortable in 12 GB with AMP |
| 2D paper 572 | 572² | 1–2 | valid convs waste border; use for checks |
| 3D (later) | 64³–128³ | 2 | will need mixed precision |

PyTorch must be a **cu128 or newer** wheel. Older cu121/cu124 wheels do not
contain sm_120 kernels; `cuda.is_available()` can still be True and then
matmul dies.

## C. How the two vendor trees map onto this

**milesial/Pytorch-UNet** (`vendor/Pytorch-UNet`)

- `unet/unet_parts.py`: `DoubleConv`, `Down`, `Up`, `OutConv` — the whole U.
- `Up.forward`: upsample, pad to match skip, `torch.cat` on dim 1, DoubleConv.
- `train.py` + `utils/dice_score.py`: a complete 2D loop.
- GPL-3.0. Read it, do not paste it into `src/`.

**MIC-DKFZ PlainConvUNet** (`vendor/dynamic-network-architectures`)

- `architectures/unet.py`: encoder returns skip list, decoder consumes it.
- `building_blocks/simple_conv_blocks.py`: conv → (dropout) → norm → nonlin,
  with same-padding.
- `building_blocks/unet_decoder.py`: transposed conv, concat (`2 * skip_ch`),
  stacked convs, optional `seg_layers` for deep supervision.
- Apache-2.0. This is the architecture nnU-Net trains.

Our `BaselineUNet` is a third, small implementation of the same U so the
project is not legally or structurally tied to either vendor. Shapes and
layer counts are tested against the paper figure.

## D. Minimum loop we will use for comparison runs

When a formula-net exists, both nets run this loop on the same data:

```
for each epoch:
    sample a batch of (image, mask)
    augment identically
    logits = net(image)
    loss = Dice(softmax(logits), mask) + CE(logits, mask)
    SGD / Adam step
evaluate:
    argmax logits
    report Dice, IoU, pixel error  (and EM warping/Rand if on that set)
```

Any extra trick (TTA, ensemble, CRF) must be applied to **both** nets or to
neither. Otherwise the comparison is invalid.
