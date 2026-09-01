---
license: apache-2.0
library_name: pytorch
pipeline_tag: image-segmentation
tags:
  - fsot
  - unet
  - cell-detection
  - zero-shot
  - biomedical
---

# Wavegazer Net

Seed-derived visual U (FSOT) for segmentation and cell **centroid detect**.
**Zero trainable weights** on the spine. Pin `D1D38A`.

This is **not** FlowNet (optical flow, ICCV 2015).

## Use

```python
import torch
from wavegazer import WavegazerNet

net = WavegazerNet(in_channels=1, n_classes=2, sparse=True).eval()
x = torch.rand(1, 1, 256, 256)
logits = net(x)                 # dense (N, K, H, W)
peaks = net.detect(x)           # centroids, 7 µm Biohub protocol
```

Frozen codon kernels load with the module; optional `wavegazer_buffers.pt`
is the same buffers serialized.

## Benchmarks (honest)

- Dense square unit test: Dice > 0.7
- Synthetic disks vs trained U-Net: U-Net wins (fitted); Wavegazer is closed-form
- Biohub detect @ 7 µm (16 volumes, YX): see `artifacts/biohub_peaks_7um.json`

Not a CellMot 0.848 claim. Detect gate first, linking later.

Code: GitHub `dappalumbo91/Wavegazer-Net`
