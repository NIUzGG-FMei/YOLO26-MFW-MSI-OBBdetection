# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

"""Lightweight Voxel Coordinate Attention (LVCA) module.

Reproduction of the LVCA attention module proposed in "Attention Mechanism-Optimized Object Detection Algorithm for
Traffic Scenes Based on YOLO26". The module aggregates input features along the height and width directions with
directional average pooling, fuses them in a low-dimensional bottleneck (CBS), and produces a two-dimensional spatial
attention map by broadcasting the multiplication of the horizontal and vertical attention weights. A residual
connection stabilizes training by preventing gradient vanishing.
"""

import torch
import torch.nn as nn

from ultralytics.nn.modules.conv import Conv


class LVCA(nn.Module):
    """Lightweight Voxel Coordinate Attention module.

    Applies direction-aware feature aggregation and low-dimensional bottleneck feature fusion to recalibrate spatial
    features, enhancing small and occluded object representation while suppressing background noise.

    Attributes:
        pool_h (nn.AdaptiveAvgPool2d): Average pooling along the width direction, output shape (C, H, 1).
        pool_w (nn.AdaptiveAvgPool2d): Average pooling along the height direction, output shape (C, 1, W).
        cv1 (Conv): CBS bottleneck compressing channels from C to mip on the concatenated (H + W) feature.
        cv_h (nn.Conv2d): 1x1 convolution restoring channels from mip to C for the horizontal branch.
        cv_w (nn.Conv2d): 1x1 convolution restoring channels from mip to C for the vertical branch.

    References:
        L. Shang, "Attention Mechanism-Optimized Object Detection Algorithm for Traffic Scenes Based on YOLO26",
        in Proc. IEEE ICSP, 2026.
    """

    def __init__(self, c1: int, mip: int | None = None) -> None:
        """Initialize the LVCA module.

        Args:
            c1 (int): Number of input channels.
            mip (int | None): Intermediate bottleneck dimension. Defaults to max(8, c1 // 32) when None.
        """
        super().__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))  # (C, H, 1)
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))  # (C, 1, W)
        self.cv1 = Conv(c1, mip if mip is not None else max(8, c1 // 32), 1, 1)  # CBS bottleneck
        self.cv_h = nn.Conv2d(self.cv1.conv.out_channels, c1, 1, 1, bias=True)  # horizontal CS module
        self.cv_w = nn.Conv2d(self.cv1.conv.out_channels, c1, 1, 1, bias=True)  # vertical CS module

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply LVCA attention to the input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape (B, C, H, W).

        Returns:
            (torch.Tensor): Attention-recalibrated tensor of the same shape as the input.
        """
        h, w = x.shape[2:]
        x_h = self.pool_h(x)  # (B, C, H, 1)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)  # (B, C, 1, W)
        y = torch.cat([x_h, x_w], dim=2)  # (B, C, H + W, 1)
        y = self.cv1(y)  # (B, mip, H + W, 1)
        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)  # (B, mip, 1, W)
        a_h = self.cv_h(x_h).sigmoid()  # (B, C, H, 1)
        a_w = self.cv_w(x_w).sigmoid()  # (B, C, 1, W)
        att = a_h * a_w  # broadcast to (B, C, H, W)
        return x * att + x  # residual connection
