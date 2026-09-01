"""Shape and topology tests for the frozen U-Net baseline."""

import torch

from wavegazer.baseline_unet import (
    PAPER_INPUT_SIZE,
    PAPER_OUTPUT_SIZE,
    BaselineUNet,
    feature_shapes,
)


def test_modern_preserves_spatial_size():
    net = BaselineUNet(in_channels=1, n_classes=2, mode="modern")
    x = torch.randn(1, 1, 64, 64)
    y = net(x)
    assert y.shape == (1, 2, 64, 64)


def test_modern_rgb_and_multiclass():
    net = BaselineUNet(in_channels=3, n_classes=5, mode="modern")
    x = torch.randn(2, 3, 32, 32)
    y = net(x)
    assert y.shape == (2, 5, 32, 32)


def test_modern_bilinear_preserves_size():
    net = BaselineUNet(in_channels=1, n_classes=2, mode="modern", bilinear=True)
    x = torch.randn(1, 1, 48, 48)
    y = net(x)
    assert y.shape == (1, 2, 48, 48)


def test_paper_572_to_388():
    net = BaselineUNet(in_channels=1, n_classes=2, mode="paper")
    x = torch.randn(1, 1, PAPER_INPUT_SIZE, PAPER_INPUT_SIZE)
    y = net(x)
    assert y.shape == (1, 2, PAPER_OUTPUT_SIZE, PAPER_OUTPUT_SIZE)


def test_encoder_channel_schedule():
    shapes = feature_shapes(256, "modern")
    assert [c for c, _, _ in shapes] == [64, 128, 256, 512, 1024]
    assert shapes[-1][1:] == (16, 16)


def test_paper_encoder_spatial_matches_figure():
    # Figure 1 of the paper, after each pair of valid 3x3 convs.
    shapes = feature_shapes(PAPER_INPUT_SIZE, "paper")
    assert shapes[0] == (64, 568, 568)
    assert shapes[1] == (128, 280, 280)
    assert shapes[2] == (256, 136, 136)
    assert shapes[3] == (512, 64, 64)
    assert shapes[4] == (1024, 28, 28)


def test_conv_layer_count_is_23():
    # 2 per DoubleConv x 9 blocks + 4 transpose convs + 1x1 head = 23.
    net = BaselineUNet(mode="paper", bilinear=False)
    convs = [m for m in net.modules() if isinstance(m, (torch.nn.Conv2d, torch.nn.ConvTranspose2d))]
    assert len(convs) == 23


def test_he_init_is_finite():
    net = BaselineUNet(mode="modern")
    for p in net.parameters():
        assert torch.isfinite(p).all()
