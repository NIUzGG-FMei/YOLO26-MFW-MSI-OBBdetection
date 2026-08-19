"""Tests for LVCA (Lightweight Voxel Coordinate Attention) module in ultralytics/nn/modules/LVCA.py"""
import sys
sys.path.insert(0, '/home/mofengwei/ultralytics')

import torch
import pytest
from ultralytics.nn.modules.LVCA import LVCA


def _make_input(b=1, c=64, h=32, w=32):
    return torch.randn(b, c, h, w)


class TestLVCAShape:
    """Output shape matches input shape for various channel/spatial sizes."""

    def test_default_mip(self):
        m = LVCA(64)
        y = m(_make_input(1, 64))
        assert y.shape == (1, 64, 32, 32)

    def test_explicit_mip(self):
        m = LVCA(64, mip=16)
        y = m(_make_input(1, 64))
        assert y.shape == (1, 64, 32, 32)

    def test_odd_spatial(self):
        m = LVCA(32)
        y = m(_make_input(2, 32, 37, 41))
        assert y.shape == (2, 32, 37, 41)

    def test_many_channels(self):
        m = LVCA(1024)
        y = m(_make_input(1, 1024, 20, 20))
        assert y.shape == (1, 1024, 20, 20)


class TestLVCABottleneck:
    """Bottleneck dimension follows the max(8, c1 // 32) default or the explicit mip."""

    def test_default_bottleneck(self):
        m = LVCA(64)
        assert m.cv1.conv.out_channels == 8  # 64 // 32

    def test_explicit_bottleneck(self):
        m = LVCA(64, mip=16)
        assert m.cv1.conv.out_channels == 16


class TestLVCAResidual:
    """Attention weights are multiplied across directions and added back as a residual."""

    def test_identity_when_zero_gate(self):
        m = LVCA(16, mip=4)
        m.eval()
        with torch.no_grad():
            x = torch.ones(1, 16, 8, 8)
            m.cv_h.weight.zero_()
            m.cv_h.bias.data.fill_(-10)  # sigmoid(-10) ~ 0
            m.cv_w.weight.zero_()
            m.cv_w.bias.data.fill_(-10)
            y = m(x)
        assert torch.allclose(y, x, atol=1e-3)  # residual keeps input when attention ~ 0


class TestLVCAGradient:
    """Gradients flow back through the module without NaNs."""

    def test_gradient(self):
        x = _make_input(1, 64, 16, 16).requires_grad_(True)
        y = LVCA(64)(x)
        y.sum().backward()
        assert x.grad is not None
        assert not torch.isnan(x.grad).any()


class TestLVCAEvalMode:
    """Module runs correctly in eval mode (BN uses running stats)."""

    def test_eval(self):
        m = LVCA(64).eval()
        with torch.no_grad():
            y = m(_make_input(1, 64))
        assert y.shape == (1, 64, 32, 32)
        assert not torch.isnan(y).any()


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
