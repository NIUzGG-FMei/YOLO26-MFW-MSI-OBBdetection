import torch
import torch.nn as nn
import torch.nn.functional as F


class PreNorm2d(nn.Module):
    """Apply per-sample layer normalization over the full [C, H, W] feature volume."""

    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.layer_norm(x, x.shape[1:])


class ChannelSelect(nn.Module):
    """Select a fixed subset of channels from an input tensor."""

    def __init__(self, channels: list[int] | tuple[int, ...]):
        super().__init__()
        if not channels:
            raise ValueError("`channels` must contain at least one index.")
        self.channels = tuple(int(c) for c in channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if max(self.channels) < x.shape[1]:
            return x[:, self.channels, :, :]

        # Ultralytics first builds a temporary 3-channel model from YAML before
        # the trainer rebuilds the final model with `data.yaml["channels"]`.
        # In that placeholder build, keep the tensor unchanged when it already
        # has the expected branch width to avoid index errors from fixed
        # multispectral channel selections such as [1, 2, 4].
        if x.shape[1] == len(self.channels):
            return x

        raise IndexError(
            f"ChannelSelect cannot pick channels {self.channels} from input with {x.shape[1]} channels."
        )


class MS_MSA(nn.Module):
    """
    PyTorch version of the spectral self-attention core from `mst.py`.

    This module is designed for direct use in detection backbones, so the
    interface uses `[B, C, H, W]` tensors instead of MindSpore's internal
    `[B, H, W, C]` convention.

    Notes:
    - The original MindSpore implementation is effectively used with
      `heads * dim_head == dim`. This version keeps that constraint explicit.
    - The attention is computed along the spectral/channel subspace, while a
      depthwise convolution branch provides local spatial context.
    """

    def __init__(self, dim: int, heads: int = 1, dim_head: int | None = None, rescale_init: float | None = None):
        super().__init__()
        if dim_head is None:
            if dim % heads != 0:
                raise ValueError(f"`dim` ({dim}) must be divisible by `heads` ({heads}).")
            dim_head = dim // heads

        inner_dim = heads * dim_head
        if inner_dim != dim:
            raise ValueError(
                "This extracted MS_MSA keeps the original module's effective "
                "shape rule: `heads * dim_head` must equal `dim`."
            )

        self.dim = dim
        self.heads = heads
        self.dim_head = dim_head

        self.to_q = nn.Linear(dim, inner_dim, bias=True)
        self.to_k = nn.Linear(dim, inner_dim, bias=True)
        self.to_v = nn.Linear(dim, inner_dim, bias=True)
        # Keep this as a 1D learnable vector so optimizers like Muon do not
        # misclassify it as a 2D+ matrix parameter.
        if rescale_init is None:
            rescale_init = 1.0
        self.rescale = nn.Parameter(torch.full((heads,), float(rescale_init)))
        self.proj = nn.Linear(inner_dim, dim, bias=True)
        self.pos_emb = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=True),
            nn.GELU(),
            nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor with shape [B, C, H, W]

        Returns:
            Tensor with shape [B, C, H, W]
        """
        b, c, h, w = x.shape

        x_hw = x.flatten(2).transpose(1, 2)  # [B, HW, C]
        q_inp = self.to_q(x_hw)
        k_inp = self.to_k(x_hw)
        v_inp = self.to_v(x_hw)

        q = q_inp.view(b, h * w, self.heads, self.dim_head).permute(0, 2, 3, 1)  # [B, heads, d, HW]
        k = k_inp.view(b, h * w, self.heads, self.dim_head).permute(0, 2, 3, 1)  # [B, heads, d, HW]
        v = v_inp.view(b, h * w, self.heads, self.dim_head).permute(0, 2, 3, 1)  # [B, heads, d, HW]

        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)

        attn = torch.matmul(k, q.transpose(-2, -1))  # [B, heads, d, d]
        attn = attn * self.rescale.view(1, self.heads, 1, 1)
        attn = attn.softmax(dim=-1)

        out = torch.matmul(attn, v)  # [B, heads, d, HW]
        out = out.permute(0, 3, 1, 2).contiguous().view(b, h * w, self.heads * self.dim_head)
        out_c = self.proj(out).transpose(1, 2).contiguous().view(b, c, h, w)

        v_map = v_inp.transpose(1, 2).contiguous().view(b, c, h, w)
        out_p = self.pos_emb(v_map)

        return out_c + out_p


def _resolve_heads(dim: int, preferred_heads: int) -> int:
    """Pick a valid head count for the current channel width."""
    heads = max(int(preferred_heads), 1)
    while heads > 1 and dim % heads != 0:
        heads -= 1
    return heads


class SpectralStage(nn.Module):
    """
    Spectral feature extraction stage with channel alignment and MS_MSA blocks.

    This wrapper is designed for YAML-based model parsing where a stage must be
    able to change the channel width like `C3k2`, while keeping `MS_MSA` as the
    main spectral modeling operator.
    """

    def __init__(self, c1: int, c2: int, n: int = 1, heads: int = 4, stable_rescale: bool = False):
        super().__init__()
        self.align = (
            nn.Sequential(
                nn.Conv2d(c1, c2, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(c2),
                nn.SiLU(),
            )
            if c1 != c2
            else nn.Identity()
        )
        valid_heads = _resolve_heads(c2, heads)
        rescale_init = (c2 // valid_heads) ** -0.5 if stable_rescale else 1.0
        self.blocks = nn.ModuleList(MS_MSA(c2, heads=valid_heads, rescale_init=rescale_init) for _ in range(n))
        self.ffn = nn.ModuleList(
            nn.Sequential(
                nn.Conv2d(c2, c2, kernel_size=3, stride=1, padding=1, groups=c2, bias=False),
                nn.BatchNorm2d(c2),
                nn.SiLU(),
                nn.Conv2d(c2, c2, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(c2),
            )
            for _ in range(n)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.align(x)
        for attn, ffn in zip(self.blocks, self.ffn):
            x = x + attn(x)
            x = x + ffn(x)
        return x


class SpectralInputMix(nn.Module):
    """Learn a same-width 1x1 spectral channel mixing before the spectral branch."""

    def __init__(self, c1: int, use_bn_act: bool = True):
        super().__init__()
        layers: list[nn.Module] = [nn.Conv2d(c1, c1, kernel_size=1, stride=1, padding=0, bias=not use_bn_act)]
        # Optional BN + SiLU branch kept for ablations; disable it when a pure
        # learnable 1x1 spectral mixing is desired.
        if use_bn_act:
            layers.extend([nn.BatchNorm2d(c1), nn.SiLU()])
        self.mix = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mix(x)


class GuidedEnhance(nn.Module):
    """
    Spectral-guided enhancement for spatial features.

    Computes: `spectral * spatial + spatial`, where the spectral branch acts as
    an additive gate on the spatial branch.
    """

    def __init__(self, apply_sigmoid: bool = False):
        super().__init__()
        self.apply_sigmoid = apply_sigmoid

    def forward(self, x: list[torch.Tensor] | tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        if len(x) != 2:
            raise ValueError(f"GuidedEnhance expects 2 inputs, but received {len(x)}.")
        spatial, spectral = x
        if spatial.shape != spectral.shape:
            raise ValueError(
                "GuidedEnhance requires spatial and spectral features to have the same shape, "
                f"but got {tuple(spatial.shape)} and {tuple(spectral.shape)}."
            )
        if self.apply_sigmoid:
            spectral = torch.sigmoid(spectral)
        return spectral * spatial + spatial


class GuidedEnhanceZeroInit(nn.Module):
    """
    Spectral-guided enhancement with zero-initialized 1x1 attention alignment.

    Computes: spatial * (2.0 * sigmoid(conv1x1(spectral))).
    The 1x1 conv is zero-initialized so the multiplier starts at 1.0.
    """

    def __init__(self, channels: int):
        super().__init__()
        self.align_conv = nn.Conv2d(channels, channels, kernel_size=1, bias=True)
        nn.init.constant_(self.align_conv.weight, 0)
        nn.init.constant_(self.align_conv.bias, 0)

    def forward(self, x: list[torch.Tensor] | tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        if len(x) != 2:
            raise ValueError(f"GuidedEnhanceZeroInit expects 2 inputs, but received {len(x)}.")
        spatial, spectral = x
        if spatial.shape != spectral.shape:
            raise ValueError(
                "GuidedEnhanceZeroInit requires spatial and spectral features to have the same shape, "
                f"but got {tuple(spatial.shape)} and {tuple(spectral.shape)}."
            )
        attn = torch.sigmoid(self.align_conv(spectral))
        return spatial * (2.0 * attn)


if __name__ == "__main__":
    x = torch.randn(2, 256, 32, 32)
    model = MS_MSA(dim=256, heads=4)
    y = model(x)
    print(y.shape)
