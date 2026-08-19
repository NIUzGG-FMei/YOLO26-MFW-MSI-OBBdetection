"""Tests for the SDF-YOLO Sim-ACCoM and DIF modules."""

import pytest
import torch
import torch.nn.functional as F

from ultralytics.nn.modules.SDFYOLO import DIF, SimACCoM, SimAM, SpatialAttentionFree, SpatialAttentionMap
from ultralytics.nn.tasks import DetectionModel, parse_model
from ultralytics.utils import ROOT


def test_simam_matches_reference_formula():
    """SimAM should normalize each channel using its spatial variance."""
    x = torch.randn(2, 4, 5, 7)
    squared = (x - x.mean(dim=(2, 3), keepdim=True)).pow(2)
    variance = squared.sum(dim=(2, 3), keepdim=True) / (5 * 7 - 1)
    expected = x * torch.sigmoid(squared / (4 * (variance + 1e-4)) + 0.5)

    assert torch.allclose(SimAM()(x), expected)


def test_simam_handles_single_spatial_element():
    """A 1x1 feature should remain finite instead of dividing by zero."""
    output = SimAM()(torch.ones(2, 4, 1, 1))
    assert torch.isfinite(output).all()


@pytest.mark.parametrize("attention", [SpatialAttentionMap(7), SpatialAttentionFree()])
def test_spatial_attention_returns_broadcastable_map(attention):
    """Both documented SA variants should return one weight per spatial position."""
    output = attention(torch.randn(2, 16, 13, 17))
    assert output.shape == (2, 1, 13, 17)
    assert output.min() >= 0 and output.max() <= 1


@pytest.mark.parametrize(
    ("pos", "inputs", "expected_shape"),
    [
        (1, [torch.randn(1, 16, 21, 23), torch.randn(1, 32, 11, 12)], (1, 16, 21, 23)),
        (
            2,
            [torch.randn(1, 16, 41, 45), torch.randn(1, 32, 21, 23), torch.randn(1, 64, 11, 12)],
            (1, 32, 21, 23),
        ),
        (4, [torch.randn(1, 64, 21, 23), torch.randn(1, 128, 11, 12)], (1, 128, 11, 12)),
    ],
)
def test_simaccom_positions_and_odd_shapes(pos, inputs, expected_shape):
    """All position variants should align adjacent odd-sized features and preserve the current shape."""
    current_channels = expected_shape[1]
    module = SimACCoM(current_channels, pos=pos, k=3, dilations=(1, 2), depthwise=True)
    output = module(inputs)
    assert output.shape == expected_shape
    output.mean().backward()
    assert all(parameter.grad is not None for parameter in module.parameters())


def test_depthwise_simaccom_reduces_yolo26_stage_parameters():
    """The OBB-oriented option should substantially reduce context-branch parameters."""
    channels = (64, 128, 128, 256)
    dense = sum(sum(p.numel() for p in SimACCoM(c, pos=i + 1).parameters()) for i, c in enumerate(channels))
    depthwise = sum(
        sum(p.numel() for p in SimACCoM(c, pos=i + 1, depthwise=True).parameters()) for i, c in enumerate(channels)
    )
    assert depthwise < dense / 20


def test_dif_matches_paper_equation_with_interpolation():
    """DIF should calculate main + Conv1x1(Interp(aux))."""
    main = torch.randn(1, 8, 21, 23)
    auxiliary = torch.randn(1, 16, 11, 12)
    module = DIF(8, 16)
    expected = main + module.cv(F.interpolate(auxiliary, size=main.shape[2:], mode="nearest"))

    assert torch.allclose(module([main, auxiliary]), expected)


def test_dif_rejects_non_image_interpolation_mode():
    """DIF should reject modes whose dimensionality is incompatible with BCHW feature maps."""
    with pytest.raises(ValueError, match="interpolation mode"):
        DIF(8, 16, mode="linear")


def test_parser_injects_multi_input_channels():
    """Model parsing should derive current/main and auxiliary channels from nested from entries."""
    config = {
        "nc": 1,
        "backbone": [
            [-1, 1, "Conv", [16, 3, 2]],
            [-1, 1, "Conv", [32, 3, 2]],
            [[0, 1], 1, "SimACCoM", [1, 3, [1, 2], True]],
            [[2, 0], 1, "DIF", []],
        ],
        "head": [],
    }
    model, _ = parse_model(config, ch=3, verbose=False)

    assert model[2].c == 16
    assert model[2].depthwise
    assert model[3].c1 == 16
    assert model[3].c2 == 16


@pytest.mark.parametrize(
    "layer",
    [
        [[0, 1], 2, "SimACCoM", [1]],
        [[0], 1, "DIF", []],
    ],
)
def test_parser_rejects_invalid_multi_input_rows(layer):
    """Invalid repeats and from lists should fail during parsing with actionable errors."""
    config = {"nc": 1, "backbone": [[-1, 1, "Conv", [16, 3, 2]], layer], "head": []}
    with pytest.raises(ValueError):
        parse_model(config, ch=3, verbose=False)


def test_yolo26_obb_sdf_config_builds_without_p2_head():
    """The example config should consume all three SDF branches and retain only P3/P4/P5 OBB outputs."""
    model = DetectionModel(ROOT / "cfg/models/26/yolo26-obb-sdf.yaml", verbose=False)

    assert sum(isinstance(layer, SimACCoM) for layer in model.model) == 3
    assert sum(isinstance(layer, DIF) for layer in model.model) == 3
    assert model.stride.tolist() == [8.0, 16.0, 32.0]
    assert model.model[-1].__class__.__name__ == "OBB26"
