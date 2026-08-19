# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""SDF-YOLO modules (Sim-ACCoM & DIF).

Reproduction of the two core modules proposed in:

    L. Lu et al., "SDF-YOLO: A multi-scale dynamic feature fusion network for enhanced lightweight object
    detection in autonomous driving", Neurocomputing 681 (2026) 133328.

Implemented here:

1. Sim-ACCoM  (Simple Attention-guided Contextual and Cross-scale Module, Sec. 3.2 & Fig. 3, Eqs. (1)-(7))
2. DIF        (Dynamic Interpolation Fusion module, Sec. 3.3, Eq. (8))

Both modules accept a ``list`` of feature maps. In a model YAML, the ``from`` entry must therefore be a nested
list in the standard ``[from, repeats, module, args]`` row, for example ``[[-1, 6], 1, DIF, []]``. The forward
input order is the same as the order in ``from``. See each class docstring for the exact input order.

References:
    https://doi.org/10.1016/j.neucom.2026.133328
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .conv import Conv

__all__ = ("DIF", "SimACCoM", "SimAM", "SpatialAttentionFree", "SpatialAttentionMap")


class SpatialAttentionFree(nn.Module):
    """Parameter-free spatial attention map for lightweight experiments.

    The SDF-YOLO paper names a Spatial Attention (SA) operation but does not publish its formula. This fallback
    computes a spatial map without trainable parameters:

        SA(x) = sigmoid(mean_{c}(x))                shape: (B, 1, H, W)

    The resulting (B, 1, H, W) map is broadcast over all channels and element-wise multiplied with the
    enhanced current-branch feature ``f'_i`` (Eqs. (3)-(4)). This works even when the adjacent branch has a
    different channel count than the current branch (which is always the case in the SDF-YOLO backbone).

    Sim-ACCoM uses :class:`SpatialAttentionMap` by default because the paper's kernel-size ablation explicitly
    includes the SA convolution. Set ``sa_kernel=0`` to select this parameter-free approximation instead.
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return a (B, 1, H, W) attention map in [0, 1] from a (B, C, H, W) feature map."""
        return torch.sigmoid(x.mean(dim=1, keepdim=True))


class SpatialAttentionMap(nn.Module):
    """Generate a trainable spatial attention map from channel-average and channel-maximum descriptors.

    This follows the conventional spatial-attention interpretation of the SA blocks in SDF-YOLO Fig. 3. The
    paper does not provide the exact SA formula, so this implementation should be treated as the closest
    reproducible interpretation rather than author-code equivalence.
    """

    def __init__(self, kernel_size: int = 7) -> None:
        """Initialize a spatial attention map generator with a 3x3 or 7x7 convolution."""
        super().__init__()
        if kernel_size not in {3, 7}:
            raise ValueError(f"Spatial attention kernel size must be 3 or 7, got {kernel_size}")
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return a spatial attention map of shape (B, 1, H, W)."""
        descriptors = torch.cat((x.mean(dim=1, keepdim=True), x.amax(dim=1, keepdim=True)), dim=1)
        return self.conv(descriptors).sigmoid()


class SimAM(nn.Module):
    """SimAM: Simple, Parameter-free Attention Module.

    From: L. Yang et al., "SimAM: A Simple, Parameter-Free Attention Module for Convolutional Neural
    Networks", ICML 2021. Used inside Sim-ACCoM to remove redundant information from the enhanced feature
    ``f'_i`` (Eq. (5))::

        f''_i = SimAM(f'_i)

    The module contains no trainable parameters. ``channels`` is kept only for API compatibility with the
    original implementation and is not used internally.
    """

    def __init__(self, channels: int | None = None, e_lambda: float = 1e-4):
        """Initialize SimAM.

        Args:
            channels (int | None): Number of input channels (unused, kept for interface compatibility).
            e_lambda (float): Small constant added for numerical stability (default 1e-4, same as original).
        """
        super().__init__()
        self.activation = nn.Sigmoid()
        self.e_lambda = e_lambda

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Energy-based attention weighting on input (B, C, H, W) tensor, output has the same shape."""
        spatial_elements = x.shape[2] * x.shape[3]
        if spatial_elements <= 1:
            return x * self.activation(torch.full_like(x, 0.5))
        x_minus_mu_square = (x - x.mean(dim=[2, 3], keepdim=True)).pow(2)  # (B, C, H, W)
        variance = x_minus_mu_square.sum(dim=[2, 3], keepdim=True) / (spatial_elements - 1)
        y = x_minus_mu_square / (4 * (variance + self.e_lambda)) + 0.5
        return x * self.activation(y)


class SimACCoM(nn.Module):
    """Sim-ACCoM: Simple Attention-guided Contextual and Cross-scale Module (paper Sec. 3.2, Fig. 3).

    Enhances a "current branch" feature map ``f_i`` with (a) multi-scale contextual information extracted by
    dilated convolutions of increasing dilation rates, and (b) cross-scale information from the adjacent
    backbone levels ``f_{i-1}`` (previous/shallower level) and ``f_{i+1}`` (next/deeper level), gated by
    spatial attention and refined by the parameter-free SimAM module.

    Processing pipeline (paper Eqs. (1)-(7))::

        1) Feature enhancement (Eqs. (1)-(2)):
               f_{t,j}^dc = DConv(f_i, w_{3x3}, r_j),   r_j = j, j in {1,2,3,4}   # dilated conv + BN + ReLU
               f'_i       = Conv( Concat(f_1^dc..f_4^dc) )                         # channel restore (4x -> 1x)

        2) Attention weighting (Eqs. (3)-(5)):
               f_1 = SA( Down(f_{i-1}) ) * f'_i     # Down = 2x max pooling (previous level)
               f_2 = SA( UP  (f_{i+1}) ) * f'_i     # UP   = 2x nearest interpolation (next level)
               f''_i = SimAM(f'_i)

        3) Residual fusion (Eqs. (6)-(7)), pos-dependent:
               pos = 1 : out = f''_i + f_2 + f_i                    # only fuses the "next" branch
               pos = 2/3: out = f''_i + f_2 + f_1 + f_i             # fuses both adjacent branches
               pos = 4 : out = f''_i + f_1 + f_i                    # only fuses the "previous" branch

    Input/output shapes:
        - pos=1:  forward receives ``[cur, nxt]``   (2 tensors), out shape == cur.shape
        - pos=2/3:forward receives ``[prev, cur, nxt]`` (3 tensors), out shape == cur.shape
        - pos=4:  forward receives ``[prev, cur]``  (2 tensors), out shape == cur.shape

    All adjacent tensors keep their own channel counts. Their spatial sizes are aligned to the current branch,
    and their (B, 1, H, W) attention maps are broadcast over the enhanced current feature, so no channel
    conversion is needed on adjacent branches.

    Paper-optimal settings (Sec. 4.3.1): 7x7 dilated kernels with progressive dilation rates [1, 2, 3, 4].

    Args:
        c (int): Number of current-branch channels, auto-injected by ``parse_model`` from the corresponding
            ``from`` entry; do not write it in YAML arguments.
        pos (int): Module position index 1-4. Controls which adjacent branches are used, see fusion rules
            above. In SDF-YOLO, Sim-ACCoM1 uses the first backbone stage (stride 4), Sim-ACCoM2/3 use the
            second/third stages (stride 8/16), and Sim-ACCoM4 uses the deep SPPF-stage feature (stride 32).
        k (int): Kernel size of the dilated convolutions (default 7, the paper-optimal value).
        dilations (int | tuple[int, ...]): Dilation rates of the multi-scale branch, one conv per rate
            (default (1, 2, 3, 4), the paper-optimal progressive configuration).
        depthwise (bool): Use depthwise dilated convolutions instead of the paper-described standard dilated
            convolutions. Defaults to False for structural fidelity. Set True for the recommended YOLO26-OBB
            deployment trade-off; the restore convolution still performs cross-channel mixing.
        rk (int): Kernel size of the channel-restore convolution applied on the concatenated (4x) features
            (default 1, matching "Conv 1x1" in Fig. 3; the paper text Eq. (2) writes 3x3 - both are supported
            by setting rk=3).
        sa_kernel (int): Spatial-attention convolution kernel size. Defaults to 7, the paper-optimal setting.
            Set to 0 to use the parameter-free approximation because the paper does not publish the SA formula.
        e_lambda (float): Stability constant passed to the internal SimAM module (default 1e-4).

    YAML row arguments after all referenced backbone levels have already been computed. For the unmodified
    ``yolo26-obb.yaml`` backbone, P2/P3/P4/P5-pre-SPPF/deep-P5 are layers 2/4/6/8/10::

        [[2, 4], 1, SimACCoM, [1]]
        [[2, 4, 6], 1, SimACCoM, [2]]
        [[4, 6, 8], 1, SimACCoM, [3]]
        [[6, 10], 1, SimACCoM, [4]]

        # YOLO26-OBB speed-oriented form: preserve k=7 and dilation rates, but use depthwise branches.
        [[2, 4], 1, SimACCoM, [1, 7, [1, 2, 3, 4], True]]

    Note:
        Use ``n=1`` in the YAML row (the module cannot be repeated inside a ``nn.Sequential`` since it takes
        a list of tensors). These rows only create enhanced features; route each output into the neck through
        DIF. Inserting four rows before the head also shifts all subsequent absolute head indices by four.
    """

    def __init__(
        self,
        c: int,
        pos: int = 2,
        k: int = 7,
        dilations=(1, 2, 3, 4),
        depthwise: bool = False,
        rk: int = 1,
        sa_kernel: int = 7,
        e_lambda: float = 1e-4,
    ) -> None:
        """Initialize Sim-ACCoM with multi-scale dilated convolutions, spatial attention and SimAM."""
        super().__init__()
        if pos not in {1, 2, 3, 4}:
            raise ValueError(f"SimACCoM pos must be 1, 2, 3 or 4, got {pos}")
        dilations = (dilations,) if isinstance(dilations, int) else tuple(dilations)
        if not dilations or any(not isinstance(d, int) or d < 1 for d in dilations):
            raise ValueError(f"SimACCoM dilations must contain positive integers, got {dilations}")
        if not isinstance(k, int) or k < 1 or k % 2 == 0:
            raise ValueError(f"SimACCoM kernel size must be a positive odd integer, got {k}")
        if not isinstance(rk, int) or rk < 1 or rk % 2 == 0:
            raise ValueError(f"SimACCoM restore kernel size must be a positive odd integer, got {rk}")
        self.pos = pos
        self.c = c
        self.dilations = dilations
        self.depthwise = depthwise
        # (1) Multi-scale dilated convolutions (BN + ReLU), one per dilation rate (paper Eq. (1))
        groups = c if depthwise else 1
        self.dconvs = nn.ModuleList(Conv(c, c, k, 1, g=groups, d=d, act=nn.ReLU(inplace=True)) for d in dilations)
        # (2) Channel-restore convolution: concat of `n` branches (n*c channels) back to c (paper Eq. (2),
        #     "Restore (4x)" + "Conv 1x1" in Fig. 3)
        self.restore = Conv(c * len(dilations), c, rk, 1, act=nn.ReLU(inplace=True))
        # Independent SA blocks match the two branches drawn in paper Fig. 3.
        sa_cls = SpatialAttentionFree if sa_kernel == 0 else SpatialAttentionMap
        sa_args = () if sa_kernel == 0 else (sa_kernel,)
        self.sa_prev = sa_cls(*sa_args) if pos != 1 else None
        self.sa_next = sa_cls(*sa_args) if pos != 4 else None
        # SimAM module for redundancy removal (paper Eq. (5))
        self.simam = SimAM(e_lambda=e_lambda)

    def _align(self, feat: torch.Tensor, ref: torch.Tensor, kind: str) -> torch.Tensor:
        """Align ``feat`` spatial size to ``ref``: 2x max-pooling for 'down', 2x nearest-up for 'up'.

        If sizes already match, the feature is returned unchanged (robustness for custom wiring).
        """
        target_size = ref.shape[2:]
        if feat.shape[2:] == target_size:
            return feat
        if kind == "down":
            return F.adaptive_max_pool2d(feat, target_size)
        if kind == "up":
            return F.interpolate(feat, size=target_size, mode="nearest")
        raise ValueError(f"Unknown SimACCoM alignment kind {kind!r}")

    def forward(self, x: list[torch.Tensor]) -> torch.Tensor:
        """Fuse current and adjacent level features; input order depends on ``self.pos`` (see class docstring)."""
        expected_inputs = 2 if self.pos in {1, 4} else 3
        if not isinstance(x, (list, tuple)) or len(x) != expected_inputs:
            raise ValueError(f"SimACCoM pos={self.pos} expects {expected_inputs} input tensors")
        if self.pos == 1:  # inputs: [cur, nxt]
            cur, nxt = x
            prev = None
        elif self.pos == 4:  # inputs: [prev, cur]
            prev, cur = x
            nxt = None
        else:  # inputs: [prev, cur, nxt]
            prev, cur, nxt = x
        if cur.shape[1] != self.c:
            raise ValueError(f"SimACCoM expected {self.c} current-branch channels, got {cur.shape[1]}")

        # (1) Multi-scale contextual enhancement: concat of dilated outputs, then channel restore (Eqs. (1)-(2))
        fi = self.restore(torch.cat([dc(cur) for dc in self.dconvs], dim=1))  # f'_i

        # (2) Cross-scale attention weighting (Eqs. (3)-(5))
        out = self.simam(fi)  # f''_i
        if prev is not None:  # f_1 = SA(Down(f_{i-1})) * f'_i  (pos = 2, 3, 4)
            out = out + self.sa_prev(self._align(prev, cur, "down")) * fi
        if nxt is not None:  # f_2 = SA(UP(f_{i+1})) * f'_i     (pos = 1, 2, 3)
            out = out + self.sa_next(self._align(nxt, cur, "up")) * fi

        # (3) Residual connection with the original current branch (Eq. (6))
        return out + cur


class DIF(nn.Module):
    """DIF: Dynamic Interpolation Fusion module (paper Sec. 3.3, Eq. (8)).

    Fuses a primary (main) feature map from the neck with an auxiliary feature map from the corresponding
    backbone level (e.g. a Sim-ACCoM output). The auxiliary feature is first aligned to the primary feature
    by nearest-neighbor interpolation (upsampling in the SDF-YOLO wiring, interpolation generally), then a
    1x1 convolution maps its channels to the primary feature's channels, and the two maps are summed
    element-wise::

        f_out = f_1 + Conv( Interp(f_2) )                                    (paper Eq. (8))
        f_1: primary (main) feature map, e.g. neck C2f output
        f_2: auxiliary feature map, e.g. corresponding Sim-ACCoM output
        Interp: nearest-neighbor interpolation to the primary feature size
        Conv: 1x1 convolution for channel alignment

    The module is lightweight and stable during training; its trainable parameters are only the 1x1
    convolution's weight and optional bias. In SDF-YOLO seven DIF modules are deployed: six after the neck C2f
    modules (fusing the neck C2f output with the same-level Sim-ACCoM output) and one special DIF4 fusing the
    outputs of Sim-ACCoM4 and the SPPF.

    Args:
        c1 (int): Primary-branch channels, auto-injected by ``parse_model``; do not write it in YAML arguments.
        c2 (int | None): Auxiliary-branch channels, also auto-injected by ``parse_model``. When constructing
            the module directly in Python, it defaults to ``c1``.
        mode (str): Interpolation mode used to align the auxiliary feature ('nearest' | 'bilinear', etc.,
            default 'nearest' as in the paper).
        bias (bool): Whether the 1x1 convolution has a bias. Defaults to True.

    YAML usage (main branch is the first ``from`` entry, auxiliary branch is the second)::

        # After a neck block, fuse the main output with an existing Sim-ACCoM output at layer 18.
        [[-1, 18], 1, DIF, []]
        # Optional non-paper interpolation mode; main and auxiliary channels remain auto-injected.
        [[-1, 18], 1, DIF, ["bilinear"]]

    Note:
        Use ``n=1`` in the YAML row (the module takes a list of tensors and cannot be repeated).
    """

    def __init__(self, c1: int, c2: int | None = None, mode: str = "nearest", bias: bool = True) -> None:
        """Initialize DIF with a 1x1 channel-alignment convolution."""
        super().__init__()
        self.c1 = c1
        self.c2 = c1 if c2 is None else c2
        valid_modes = {"nearest", "nearest-exact", "bilinear", "bicubic", "area"}
        if mode not in valid_modes:
            raise ValueError(f"DIF interpolation mode must be one of {sorted(valid_modes)}, got {mode!r}")
        self.mode = mode
        self.cv = nn.Conv2d(self.c2, self.c1, 1, bias=bias)

    def forward(self, x: list[torch.Tensor]) -> torch.Tensor:
        """Fuse a ``[main, aux]`` tensor pair (main from the current layer, aux from an earlier layer)."""
        if not isinstance(x, (list, tuple)) or len(x) != 2:
            raise ValueError("DIF expects exactly two input tensors ordered as [main, auxiliary]")
        main, aux = x
        if main.shape[1] != self.c1 or aux.shape[1] != self.c2:
            raise ValueError(
                f"DIF expected main/aux channels ({self.c1}, {self.c2}), got ({main.shape[1]}, {aux.shape[1]})"
            )
        if aux.shape[2:] != main.shape[2:]:  # align spatial resolution to the primary feature (Eq. (8))
            align_corners = False if self.mode in {"bilinear", "bicubic"} else None
            aux = F.interpolate(aux, size=main.shape[2:], mode=self.mode, align_corners=align_corners)
        return main + self.cv(aux)
