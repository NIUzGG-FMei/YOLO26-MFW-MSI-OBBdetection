"""Tests for C3k2_PC block in ultralytics/nn/modules/block_My.py"""
import sys
sys.path.insert(0, '/home/mofengwei/ultralytics')

import torch
import pytest
from ultralytics.nn.modules.block_My import C3k2_PC


def _make_input(b=1, c=64, h=32, w=32):
    return torch.randn(b, c, h, w)


class TestC3k2PCShape:
    """Output shape is (B, c2, H, W) regardless of mode."""

    def test_default_mode(self):
        m = C3k2_PC(64, 128, n=1)
        x = _make_input(1, 64)
        y = m(x)
        assert y.shape == (1, 128, 32, 32)

    def test_c3k_mode(self):
        m = C3k2_PC(64, 128, n=1, c3k=True)
        x = _make_input(1, 64)
        y = m(x)
        assert y.shape == (1, 128, 32, 32)

    def test_attn_mode(self):
        # c must be divisible by num_heads; num_heads = max(c // 64, 1)
        # with e=0.5, c = c2*e = 64 => num_heads = 1, always safe
        m = C3k2_PC(64, 128, n=1, attn=True)
        x = _make_input(1, 64)
        y = m(x)
        assert y.shape == (1, 128, 32, 32)

    def test_same_channels(self):
        m = C3k2_PC(64, 64, n=2)
        x = _make_input(2, 64)
        y = m(x)
        assert y.shape == (2, 64, 32, 32)

    def test_batch_size(self):
        m = C3k2_PC(32, 64, n=1)
        x = _make_input(4, 32)
        y = m(x)
        assert y.shape == (4, 64, 32, 32)

    def test_multiple_blocks(self):
        m = C3k2_PC(64, 128, n=3)
        x = _make_input(1, 64)
        y = m(x)
        assert y.shape == (1, 128, 32, 32)


class TestC3k2PCKernels:
    """Custom kernel lists are accepted."""

    def test_single_kernel(self):
        m = C3k2_PC(64, 64, n=1, kk=[3])
        y = m(_make_input(1, 64))
        assert y.shape == (1, 64, 32, 32)

    def test_two_kernels(self):
        m = C3k2_PC(64, 64, n=1, kk=[3, 5])
        y = m(_make_input(1, 64))
        assert y.shape == (1, 64, 32, 32)

    def test_three_kernels(self):
        m = C3k2_PC(64, 64, n=1, kk=[3, 5, 7])
        y = m(_make_input(1, 64))
        assert y.shape == (1, 64, 32, 32)


class TestC3k2PCGradient:
    """Gradients flow back through all modes."""

    def _check_grad(self, m):
        x = _make_input(1, 64, 16, 16).requires_grad_(True)
        y = m(x)
        y.sum().backward()
        assert x.grad is not None
        assert not torch.isnan(x.grad).any()

    def test_grad_default(self):
        self._check_grad(C3k2_PC(64, 128, n=1))

    def test_grad_c3k(self):
        self._check_grad(C3k2_PC(64, 128, n=1, c3k=True))

    def test_grad_attn(self):
        self._check_grad(C3k2_PC(64, 128, n=1, attn=True))


class TestC3k2PCShortcut:
    """Shortcut flag affects Bottleneck_PC add path (c1==c2 required for residual)."""

    def test_shortcut_true_same_channels(self):
        # residual active: add=True because shortcut=True and c1==c2 inside bottleneck
        m = C3k2_PC(64, 64, n=1, shortcut=True)
        y = m(_make_input(1, 64))
        assert y.shape == (1, 64, 32, 32)

    def test_shortcut_false(self):
        m = C3k2_PC(64, 64, n=1, shortcut=False)
        y = m(_make_input(1, 64))
        assert y.shape == (1, 64, 32, 32)


class TestC3k2PCEvalMode:
    """Model runs correctly in eval mode (BN uses running stats)."""

    def test_eval_default(self):
        m = C3k2_PC(64, 128, n=1).eval()
        with torch.no_grad():
            y = m(_make_input(1, 64))
        assert y.shape == (1, 128, 32, 32)

    def test_eval_attn(self):
        m = C3k2_PC(64, 128, n=1, attn=True).eval()
        with torch.no_grad():
            y = m(_make_input(1, 64))
        assert y.shape == (1, 128, 32, 32)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
