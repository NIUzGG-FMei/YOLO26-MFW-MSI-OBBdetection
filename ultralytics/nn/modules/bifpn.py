# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""BiFPN feature fusion module."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .conv import Conv

__all__ = ("BiFPN", "BiFPN_Add2", "BiFPN_Add3")


class BiFPN(nn.Module):
    """Single-layer BiFPN fusing P3/P4/P5 features.

    One forward pass runs one top-down path followed by one bottom-up path with
    fast normalized feature fusion and per-node learnable weights.
    Inputs and outputs are lists ordered [P3, P4, P5].
    """

    def __init__(self, c_in: list[int], c_out: list[int]) -> None:
        """Initialize the BiFPN layer.

        Args:
            c_in: Input channels of [P3, P4, P5], width-scaled by parse_model.
            c_out: Output channels of [P3, P4, P5], width-scaled by parse_model.
        """
        super().__init__()
        if len(c_in) != 3 or len(c_out) != 3:
            raise ValueError(f"BiFPN expects 3 input/output levels, got c_in={c_in}, c_out={c_out}")
        c3, c4, c5 = c_in
        o3, o4, o5 = c_out
        self.eps = 1e-4

        # 1x1 projections aligning backbone channels to the BiFPN widths
        self.p3_in = Conv(c3, o3, 1)
        self.p4_in = Conv(c4, o4, 1)
        self.p5_in = Conv(c5, o5, 1)

        # Top-down path: P5 -> P4 -> P3
        self.p5_td = Conv(o5, o5, 1)
        self.p5_to_p4 = Conv(o5, o4, 1)
        self.p4_td = Conv(o4, o4, 3)
        self.p4_to_p3 = Conv(o4, o3, 1)
        self.p3_td = Conv(o3, o3, 3)

        # Bottom-up path: P3 -> P4 -> P5, plus the P4 -> P3 lateral
        self.p3_to_p4 = Conv(o3, o4, 1)
        self.p4_out = Conv(o4, o4, 3)
        self.p4_to_p5 = Conv(o4, o5, 1)
        self.p5_out = Conv(o5, o5, 3)
        self.p4_to_p3_out = Conv(o4, o3, 1)
        self.p3_out = Conv(o3, o3, 3)

        # Fast normalized fusion weights (one scalar per input per node)
        self.w4_td = nn.Parameter(torch.ones(2))
        self.w3_td = nn.Parameter(torch.ones(2))
        self.w4_out = nn.Parameter(torch.ones(3))
        self.w5_out = nn.Parameter(torch.ones(3))
        self.w3_out = nn.Parameter(torch.ones(3))

    def _fuse(self, w: torch.Tensor, xs: list[torch.Tensor]) -> torch.Tensor:
        """Apply fast normalized feature fusion, i.e. sum(w_i * x_i) / sum(w_i)."""
        w = w.relu()
        return sum(w[i] * x for i, x in enumerate(xs)) / (w.sum() + self.eps)

    def forward(self, x: list[torch.Tensor]) -> list[torch.Tensor]:
        """Fuse [P3, P4, P5] through one top-down and one bottom-up pass."""
        p3, p4, p5 = x
        p3 = self.p3_in(p3)
        p4 = self.p4_in(p4)
        p5 = self.p5_in(p5)

        # Top-down
        p5_td = self.p5_td(p5)
        p4_td = self.p4_td(
            self._fuse(self.w4_td, [p4, self.p5_to_p4(F.interpolate(p5_td, size=p4.shape[-2:], mode="nearest"))])
        )
        p3_td = self.p3_td(
            self._fuse(self.w3_td, [p3, self.p4_to_p3(F.interpolate(p4_td, size=p3.shape[-2:], mode="nearest"))])
        )

        # Bottom-up
        p4_out = self.p4_out(
            self._fuse(
                self.w4_out,
                [p4_td, p4, self.p3_to_p4(F.interpolate(p3_td, size=p4.shape[-2:], mode="nearest"))],
            )
        )
        p5_out = self.p5_out(
            self._fuse(
                self.w5_out,
                [p5_td, p5, self.p4_to_p5(F.interpolate(p4_out, size=p5.shape[-2:], mode="nearest"))],
            )
        )
        p3_out = self.p3_out(
            self._fuse(
                self.w3_out,
                [p3_td, p3, self.p4_to_p3_out(F.interpolate(p4_out, size=p3.shape[-2:], mode="nearest"))],
            )
        )
        return [p3_out, p4_out, p5_out]


class BiFPN_Add(nn.Module):
    """Learnable weighted add with per-input channel alignment and resize.

    The first input is the target branch: its spatial size and channel count define
    the output. Every other input is resized to the target size with nearest
    interpolation and projected to the target channels with a 1x1 Conv, then all
    inputs are combined with fast normalized fusion weights.
    """

    def __init__(self, c1s: list[int], eps: float = 1e-4) -> None:
        """Initialize the fusion node.

        Args:
            c1s: Input channel counts, first element is the target branch.
            eps: Small constant for fast normalized fusion stability.
        """
        super().__init__()
        if len(c1s) < 2:
            raise ValueError(f"BiFPN_Add requires at least 2 inputs, got c1s={c1s}")
        c2 = c1s[0]
        self.eps = eps
        self.w = nn.Parameter(torch.ones(len(c1s), dtype=torch.float32))
        self.cv = nn.ModuleList(Conv(c1, c2, 1) if c1 != c2 else nn.Identity() for c1 in c1s)

    def forward(self, x: list[torch.Tensor]) -> torch.Tensor:
        """Apply channel alignment, spatial alignment and weighted addition."""
        if len(x) != len(self.w):
            raise ValueError(f"Expected {len(self.w)} inputs, got {len(x)}")
        target_size = x[0].shape[-2:]
        xs = [cv(xi) for cv, xi in zip(self.cv, x)]
        xs = [xi if xi.shape[-2:] == target_size else F.interpolate(xi, size=target_size, mode="nearest") for xi in xs]
        w = self.w.relu()
        return sum(w[i] * xi for i, xi in enumerate(xs)) / (w.sum() + self.eps)


class BiFPN_Add2(BiFPN_Add):
    """BiFPN 2-input dynamic weighted fusion module."""

    def __init__(self, c1s: list[int], eps: float = 1e-4) -> None:
        """Initialize a 2-input fusion node."""
        if len(c1s) != 2:
            raise ValueError(f"BiFPN_Add2 expects exactly 2 inputs, got c1s={c1s}")
        super().__init__(c1s, eps)


class BiFPN_Add3(BiFPN_Add):
    """BiFPN 3-input dynamic weighted fusion module with a cross-node skip connection."""

    def __init__(self, c1s: list[int], eps: float = 1e-4) -> None:
        """Initialize a 3-input fusion node."""
        if len(c1s) != 3:
            raise ValueError(f"BiFPN_Add3 expects exactly 3 inputs, got c1s={c1s}")
        super().__init__(c1s, eps)
