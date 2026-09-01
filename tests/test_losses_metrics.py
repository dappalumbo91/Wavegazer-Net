import torch

from wavegazer.losses import ce_dice_loss, dice_loss, softmax_cross_entropy, weighted_ce_loss
from wavegazer.metrics import dice_coefficient, intersection_over_union, pixel_error


def _perfect_pair(n_classes: int = 2):
    target = torch.zeros(1, 16, 16, dtype=torch.long)
    target[:, 4:12, 4:12] = 1
    logits = torch.zeros(1, n_classes, 16, 16)
    logits[:, 0] = 10.0
    logits[:, 1] = -10.0
    logits[:, 1, 4:12, 4:12] = 10.0
    logits[:, 0, 4:12, 4:12] = -10.0
    return logits, target


def test_perfect_prediction_metrics():
    logits, target = _perfect_pair()
    assert dice_coefficient(logits, target).item() > 0.99
    assert intersection_over_union(logits, target).item() > 0.99
    assert pixel_error(logits, target).item() < 1e-6


def test_dice_loss_zero_on_onehot_certainty():
    logits, target = _perfect_pair()
    loss = dice_loss(logits, target)
    assert loss.item() < 1e-3


def test_ce_and_compound_are_finite():
    logits = torch.randn(2, 3, 8, 8)
    target = torch.randint(0, 3, (2, 8, 8))
    weights = torch.ones(2, 8, 8)
    assert torch.isfinite(softmax_cross_entropy(logits, target))
    assert torch.isfinite(weighted_ce_loss(logits, target, weights))
    assert torch.isfinite(ce_dice_loss(logits, target))
