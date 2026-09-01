# Mathematics of the current U-Net

Symbols match the 2015 paper unless noted. Implementation:
`src/wavegazer/baseline_unet.py`, `src/wavegazer/losses.py`, `src/wavegazer/metrics.py`.

## 1. Discrete convolution

A 2D convolution at output location `(i, j)`, output channel `c`, is

\[
(W * x)_{c,i,j}
= b_c + \sum_{c'}\sum_{m=0}^{k-1}\sum_{n=0}^{k-1}
W_{c,c',m,n}\, x_{c',\, i+m,\, j+n}
\]

U-Net uses \(k = 3\). **Valid** (paper) means the kernel stays inside \(x\),
so each conv shrinks height and width by \(k-1 = 2\). **Same** (modern) pads
one pixel so the spatial size does not change.

After each conv, ReLU:

\[
\mathrm{ReLU}(z) = \max(0, z)
\]

nnU-Net replaces this with leaky ReLU, \(\max(0.01 z, z)\).

A **double-conv** block is two of these in sequence. That is the only learned
spatial mixer at a given resolution.

## 2. Downsampling (contract)

\[
y_{c,i,j} = \max_{p,q \in \{0,1\}} x_{c,\, 2i+p,\, 2j+q}
\]

Max-pool 2×2, stride 2. No learned parameters. Each application:

- halves \(H\) and \(W\)
- (in this architecture) is followed by a double-conv that doubles channels

Receptive field grows roughly as \(1 + \sum_\ell (k-1)\cdot \mathrm{stride\_product}_\ell\).
Four pools give a stride product of 16 at the bottleneck, which is why a 256²
input has a 16² bottleneck.

## 3. Upsampling (expand)

Learned transposed convolution, kernel 2, stride 2. Inserts zeros and applies
a learned 2×2 kernel, doubling spatial size and (here) halving channels.

Bilinear upsample + 1×1 conv is the parameter-light alternative (milesial
`bilinear=True`). The paper uses transposed conv.

## 4. Skip concatenation

At decoder level \(\ell\),

\[
z^{(\ell)}
= \mathrm{concat}\big(\mathrm{crop}(e^{(\ell)}),\, u^{(\ell)}\big)
\in \mathbb{R}^{C_e + C_d \times H \times W}
\]

`crop` is a center crop. It exists only because valid convolutions made
\(e^{(\ell)}\) larger than \(u^{(\ell)}\). Same-padding networks skip the crop.

This is **not** a residual add. Addition would require \(C_e = C_d\) and would
mix features in place. Concatenation keeps both streams intact so the next
3×3 kernels can choose.

## 5. Softmax (per pixel)

Let \(a_k(x)\) be the logit of class \(k\) at pixel \(x \in \Omega \subset \mathbb{Z}^2\).
Paper:

\[
p_k(x) = \frac{\exp(a_k(x))}{\sum_{k'=1}^{K} \exp(a_{k'}(x))}
\]

The network itself returns \(a\), not \(p\). Softmax lives in the loss.

## 6. Paper loss: weighted cross-entropy

Equation (1):

\[
E = \sum_{x \in \Omega} w(x)\, \log \big(p_{\ell(x)}(x)\big)
\]

In code this is a **mean** of `-log p_true`, multiplied by \(w(x)\), so tile
size does not explode the scale. \(\ell(x)\) is the ground-truth class.

Equation (2), the weight map:

\[
w(x) = w_c(x) + w_0 \exp\!\left( -\frac{(d_1(x) + d_2(x))^2}{2\sigma^2} \right)
\]

- \(w_c\): class-frequency balance (rare class gets higher weight)
- \(d_1(x)\): distance to the nearest instance border
- \(d_2(x)\): distance to the second-nearest instance border
- paper constants: \(w_0 = 10\), \(\sigma \approx 5\) pixels

The exponential is large only in the thin gap between two touching cells.
That is how one semantic class (“cell”) is forced to split into instances.

`src/wavegazer/losses.py` implements the full weighted CE. The morphological
border map in that file is a stand-in for \(d_1+d_2\) until instance masks
exist in a dataloader. Swap it for a true distance transform on instance
IDs when those labels are available.

## 7. Modern loss: Dice + cross-entropy

nnU-Net (and most current U-Nets) use

\[
\mathcal{L} = \mathcal{L}_{\mathrm{Dice}} + \mathcal{L}_{\mathrm{CE}}
\]

Soft Dice for class \(k\), with softmax probabilities \(u\) and one-hot \(v\):

\[
\mathcal{L}_{\mathrm{dc}}
= -\frac{2}{|K|} \sum_{k \in K}
\frac{\sum_i u_i^k v_i^k}{\sum_i u_i^k + \sum_i v_i^k}
\]

Equivalently people minimize \(1 - \mathrm{Dice}\). Dice is a **set overlap**
on the whole map, so a class that occupies 1% of the pixels is not drowned
by the background. That is why the hand-crafted \(w(x)\) is no longer required
for semantic (not instance) segmentation.

V-Net (Milletari 2016) introduced Dice as a training loss. nnU-Net kept it
and added CE so gradients still exist when Dice saturates.

## 8. Weight initialization (He)

Paper, following He et al. 2015: draw conv weights from a Gaussian with

\[
\mathrm{std} = \sqrt{\frac{2}{N}}, \qquad N = k^2 \cdot C_{\mathrm{in}}
\]

Example in the paper: 3×3, 64 previous channels → \(N = 9 \cdot 64 = 576\),
\(\mathrm{std} = \sqrt{2/576}\). This is `kaiming_normal_` with `fan_in` and
`nonlinearity='relu'`. Our baseline applies it to every conv and transposed conv.

## 9. Optimizer (paper vs now)

| | Paper | nnU-Net v2 default |
|--|-------|---------------------|
| Method | SGD, momentum 0.99 | SGD, momentum 0.99, Nesterov |
| Batch | 1 tile (maximize tile, not batch) | small (2–4 for 3D, larger for 2D) |
| LR | Caffe default / tuned | 0.01, poly decay over 1000 epochs |
| Epoch | until ~10 h on a 6 GB Titan | 250 iterations × 1000 epochs |

Adam at \(3 \times 10^{-4}\) appeared in the 2018 nnU-Net workshop paper and
was later dropped for SGD in the production trainer. Either is a valid
baseline; **do not mix them when comparing architectures**.

## 10. Data augmentation that actually mattered

The paper’s claim that “tens of images suffice” is not an architectural
miracle. It is elastic deformation:

- random displacement vectors on a coarse 3×3 grid
- sampled from \(\mathcal{N}(0, 10^2)\) pixels
- bicubic interpolation to every pixel
- plus rotation, shift, gray-value jitter
- dropout at the bottleneck as implicit augmentation

Tissue deforms. Teaching the net that invariance is cheaper than annotating
more slides.

## 11. Metrics (the numbers in the outcomes doc)

Dice / F1 on class \(c\):

\[
\mathrm{Dice}_c = \frac{2|P_c \cap T_c|}{|P_c| + |T_c|} = \frac{2\,\mathrm{TP}}{2\,\mathrm{TP}+\mathrm{FP}+\mathrm{FN}}
\]

IoU / Jaccard (paper Table 2):

\[
\mathrm{IoU}_c = \frac{|P_c \cap T_c|}{|P_c \cup T_c|} = \frac{\mathrm{TP}}{\mathrm{TP}+\mathrm{FP}+\mathrm{FN}}
\]

Pixel error (paper Table 1): fraction of pixels with wrong argmax.

Warping error and Rand error are **topology** metrics from the ISBI EM
challenge. They need the challenge’s evaluation code; we record the published
values but do not reimplement them in `src/wavegazer/metrics.py`.

Relation: \(\mathrm{Dice} = 2\,\mathrm{IoU}/(1+\mathrm{IoU})\). If you only
have one, you can convert, but still report both because literature mixes them.
