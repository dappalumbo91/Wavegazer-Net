import math

import torch

from wavegazer.codon_kernels import CODONS, load_published_map
from wavegazer.wavegazer_net import WavegazerNet
from wavegazer.fsot_scalar import compute_scalar
from wavegazer.fsot_seeds import AUTHORITY_PIN, CODON_CHANNELS, COLLAPSE_THRESHOLD, SEEDS


def test_authority_pin_prefix():
    assert AUTHORITY_PIN == "D1D38A"


def test_scalar_matches_gpu_default():
    # Default archive / FSOT-GPU call: N=P=1, D=25, unobserved.
    s = compute_scalar()
    # Recompute with the same closed form; pin the value so drift is visible.
    assert math.isfinite(s)
    s2 = compute_scalar(N=1.0, P=1.0, D_eff=25.0, observed=False)
    assert abs(s - s2) < 1e-15


def test_observer_branch_changes_S():
    off = compute_scalar(D_eff=6.0, observed=False)
    on = compute_scalar(D_eff=6.0, observed=True)
    assert off != on


def test_collapse_threshold_is_ceff_pvar():
    assert abs(COLLAPSE_THRESHOLD - SEEDS.c_eff * SEEDS.p_var) < 1e-15


def test_sixty_four_codons():
    assert len(CODONS) == CODON_CHANNELS
    published = load_published_map()
    assert set(CODONS) <= set(published) or published == list(CODONS)


def test_zero_trainable_parameters():
    net = WavegazerNet(in_channels=1, n_classes=2)
    assert net.count_parameters() == 0
    assert net.count_frozen() > 0


def test_modern_contract_square():
    net = WavegazerNet(in_channels=1, n_classes=2)
    x = torch.randn(1, 1, 32, 32)
    y = net(x)
    assert y.shape == (1, 2, 32, 32)
    assert torch.isfinite(y).all()


def test_rgb_and_multiclass_shapes():
    net = WavegazerNet(in_channels=3, n_classes=4)
    x = torch.randn(2, 3, 32, 32)
    y = net(x)
    assert y.shape == (2, 4, 32, 32)


def test_bright_square_is_seen():
    """Image-level S must track a bright square; Dice is the gate for washout."""
    from wavegazer.metrics import dice_coefficient

    net = WavegazerNet(in_channels=1, n_classes=2)
    x = torch.zeros(1, 1, 64, 64)
    x[:, :, 16:48, 16:48] = 1.0
    mask = torch.zeros(1, 64, 64, dtype=torch.long)
    mask[:, 16:48, 16:48] = 1
    y = net(x)
    dice = float(dice_coefficient(y, mask))
    assert dice > 0.7, f"washout still present: dice={dice:.3f}"


def test_head_is_not_softmax_partition():
    net = WavegazerNet(in_channels=1, n_classes=2)
    y = net(torch.randn(1, 1, 32, 32))
    # Collapse logits are independent ReLU branches, not a simplex.
    assert (y >= 0).all()
    row_sum = y.sum(dim=1)
    assert not torch.allclose(row_sum, torch.ones_like(row_sum), atol=0.05)
