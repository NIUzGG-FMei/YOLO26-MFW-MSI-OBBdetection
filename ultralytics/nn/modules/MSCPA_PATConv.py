import timm
import torch
import torch.nn.functional as F
from exp.irpe import build_rpe, get_rpe_config
from torch import Tensor, nn

try:
    from mmcv.runner import _load_checkpoint
    from mmdet.models.builder import BACKBONES as det_BACKBONES
    from mmdet.utils import get_root_logger

    has_mmdet = True
except ImportError:
    # print("If for detection, please install mmdetection first")
    has_mmdet = False


def hard_sigmoid(x, inplace: bool = False):
    if inplace:
        return x.add_(3.0).clamp_(0.0, 6.0).div_(6.0)
    else:
        return F.relu6(x + 3.0) / 6.0


class LinearAttention(nn.Module):
    def __init__(self, dim, num_heads):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads

        self.qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        b, c, h, w = x.shape

        x = x.view(b, c, h * w).permute(0, 2, 1)  # (b, h*w, c)

        qkv = self.qkv(x).reshape(b, h * w, 3, self.num_heads, self.dim // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        key = F.softmax(k, dim=-1)
        query = F.softmax(q, dim=-2)
        context = key.transpose(-2, -1) @ v
        x = (query @ context).reshape(b, h * w, c)

        x = self.proj(x)

        x = x.permute(0, 2, 1).view(b, c, h, w)

        return x


def _make_divisible(v, divisor, min_value=None):
    """This function is taken from the original tf repo. It ensures that all layers have a channel number that is
    divisible by 4 It can be seen
    here: https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py.
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class SqueezeExcite(nn.Module):
    def __init__(
        self, in_chs, se_ratio=0.25, reduced_base_chs=None, act_layer=nn.ReLU, gate_fn=hard_sigmoid, divisor=4, **_
    ):
        super().__init__()
        self.gate_fn = gate_fn
        reduced_chs = _make_divisible((reduced_base_chs or in_chs) * se_ratio, divisor)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_reduce = nn.Conv2d(in_chs, reduced_chs, 1, bias=True)
        self.act1 = act_layer(inplace=True)
        self.conv_expand = nn.Conv2d(reduced_chs, in_chs, 1, bias=True)

    def forward(self, x):
        x_se = self.avg_pool(x)
        x_se = self.conv_reduce(x_se)
        x_se = self.act1(x_se)
        x_se = self.conv_expand(x_se)
        x = x * self.gate_fn(x_se)
        return x


class RPEAttention(nn.Module):
    """Attention with image relative position encoding."""

    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0.0, proj_drop=0.0, rpe_config=None):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        # image relative position encoding
        self.rpe_q, self.rpe_k, self.rpe_v = build_rpe(rpe_config, head_dim=head_dim, num_heads=num_heads)

    def forward(self, x):
        B, C, h, w = x.shape
        x = x.view(B, C, h * w).transpose(1, 2)
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        q *= self.scale

        attn = q @ k.transpose(-2, -1)

        # image relative position on keys
        if self.rpe_k is not None:
            # attn += self.rpe_k(q)
            attn += self.rpe_k(q, h, w)
        # image relative position on queries
        if self.rpe_q is not None:
            attn += self.rpe_q(k * self.scale).transpose(2, 3)

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        out = attn @ v

        # image relative position on values
        if self.rpe_v is not None:
            out += self.rpe_v(attn)

        x = out.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        x = x.transpose(1, 2).view(B, C, h, w)
        return x


class SRM(nn.Module):
    def __init__(self, channel):
        super().__init__()
        self.cfc1 = nn.Conv2d(channel, channel, kernel_size=(1, 2), bias=False)
        # self.cfc2 = nn.Conv2d(channel, channel, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(channel)
        self.sigmoid = nn.Hardsigmoid()

    def forward(self, x):
        b, c, _h, _w = x.shape
        # style pooling
        mean = x.reshape(b, c, -1).mean(-1).view(b, c, 1, 1)
        std = x.reshape(b, c, -1).std(-1).view(b, c, 1, 1)
        # max_value = torch.max(x.reshape(b, c, -1), -1)[0].view(b,c,1,1)
        u = torch.cat([mean, std], dim=-1)
        # style integration
        z = self.cfc1(u)
        # z = self.act(z)
        # z = self.cfc2(z)
        # z = self.bn(z)
        g = self.sigmoid(z)
        g = g.reshape(b, c, 1, 1)
        return x * g.expand_as(x)


class PATConv(nn.Module):
    def __init__(
        self, dim, n_div=4, forward_type="split_cat", use_attn=True, channel_type="se", patnet_t0=True
    ):  #'se' if i_stage <= 2 else 'self',
        super().__init__()
        self.dim_conv3 = dim // n_div
        self.dim = dim
        self.n_div = n_div
        self.dim_untouched = dim - self.dim_conv3
        self.partial_conv3 = nn.Conv2d(self.dim_conv3, self.dim_conv3, 3, 1, 1, bias=False)
        self.use_attn = use_attn
        self.channel_type = channel_type

        if use_attn:
            if channel_type == "self":
                self.partial_conv3 = nn.Conv2d(self.dim_conv3, self.dim_conv3, 3, 1, 1, bias=False)
                rpe_config = get_rpe_config(
                    ratio=20,
                    method="euc",
                    mode="bias",
                    shared_head=False,
                    skip=0,
                    rpe_on="k",
                )
                if patnet_t0:
                    num_heads = 4
                else:
                    num_heads = 6
                self.attn = RPEAttention(
                    self.dim_untouched, num_heads=num_heads, attn_drop=0.1, proj_drop=0.1, rpe_config=rpe_config
                )
                self.norm = timm.layers.LayerNorm2d(self.dim_untouched)
                # self.norm = timm.layers.LayerNorm2d(self.dim)
                self.forward = self.forward_atten
            elif channel_type == "se":
                self.partial_conv3 = nn.Conv2d(self.dim_conv3, self.dim_conv3, 3, 1, 1, bias=False)
                self.attn = SRM(self.dim_untouched)
                self.norm = nn.BatchNorm2d(self.dim_untouched)
                self.forward = self.forward_atten
        else:
            if forward_type == "slicing":
                self.forward = self.forward_slicing
            elif forward_type == "split_cat":
                self.forward = self.forward_split_cat
            else:
                raise NotImplementedError

    def forward_atten(self, x: Tensor) -> Tensor:
        if self.channel_type:
            # print(self.channel_type)
            if self.channel_type == "se":
                x1, x2 = torch.split(x, [self.dim_conv3, self.dim_untouched], dim=1)
                x1 = self.partial_conv3(x1)
                # x = self.partial_conv3(x)
                x2 = self.attn(x2)
                x2 = self.norm(x2)
                x = torch.cat((x1, x2), 1)
                # x = self.attn(x)
            else:
                x1, x2 = torch.split(x, [self.dim_conv3, self.dim_untouched], dim=1)
                x1 = self.partial_conv3(x1)
                x2 = self.norm(x2)
                x2 = self.attn(x2)
                x = torch.cat((x1, x2), 1)
        return x

    def forward_slicing(self, x: Tensor) -> Tensor:
        x1 = x.clone()  # !!! Keep the original input intact for the residual connection later
        x1[:, : self.dim_conv3, :, :] = self.partial_conv3(x1[:, : self.dim_conv3, :, :])
        return x1

    def forward_split_cat(self, x: Tensor) -> Tensor:
        x1, x2 = torch.split(x, [self.dim_conv3, self.dim_untouched], dim=1)
        x1 = self.partial_conv3(x1)
        x = torch.cat((x1, x2), 1)
        return x


class PSAConv(nn.Module):
    def __init__(self, dim, partial=0.5):
        super().__init__()
        self.dim = dim
        self.dim_conv = int(partial * dim)
        self.dim_untouched = dim - self.dim_conv

        self.conv = nn.Conv2d(self.dim_conv, self.dim_conv, 1, bias=False)
        self.conv_attn = nn.Conv2d(self.dim_untouched, self.dim_conv, 1, bias=False)
        self.norm = nn.BatchNorm2d(self.dim_untouched)
        self.norm2 = nn.BatchNorm2d(self.dim_conv)
        # self.act2 = nn.GELU()
        self.act = nn.Hardsigmoid()

    def forward(self, x):
        _b, _c, _h, _w = x.shape
        x1, x2 = torch.split(x, [self.dim_untouched, self.dim_conv], 1)
        weight = self.act(self.conv_attn(x1))
        x1 = x1 * weight
        x1 = self.norm(x1)
        # x2 = self.act2(x2)
        x2 = self.norm2(x2)
        x2 = self.conv(x2)
        x = torch.cat((x1, x2), 1)
        return x


class DepthwiseConv(nn.Module):
    def __init__(self, in_channels, kernel_size):
        super().__init__()
        self.depthwise = nn.Conv2d(
            in_channels, in_channels, kernel_size=kernel_size, groups=in_channels, padding=kernel_size // 2
        )
        self.relu = nn.ReLU()

    def forward(self, x):
        residual = x
        x = self.depthwise(x)
        x = x + residual
        x = self.relu(x)
        return x


class MSCPA(nn.Module):
    def __init__(self, dim, num_heads=4):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads

        self.dw_conv_3x3 = DepthwiseConv(dim // 4, kernel_size=3)
        self.dw_conv_5x5 = DepthwiseConv(dim // 4, kernel_size=5)
        self.dw_conv_7x7 = DepthwiseConv(dim // 4, kernel_size=7)
        self.dw_conv_9x9 = DepthwiseConv(dim // 4, kernel_size=9)

        self.P_cattention = PATConv(
            dim // 4,
            channel_type="se",
        )
        self.P_sfattention = PATConv(
            dim // 4,
            channel_type="self",
        )
        self.P_spattention = PSAConv(dim // 4)
        self.P_lattention = LinearAttention(dim // 4, 4)
        self.final_conv = nn.Conv2d(dim, dim, 1)

        self.scale_weights = nn.Parameter(torch.ones(4), requires_grad=True)

    def forward(self, input_):
        _b, c, _h, _w = input_.shape
        input_reshaped = input_
        split_size = c // 4
        x_3x3 = input_reshaped[:, :split_size, :, :]
        x_5x5 = input_reshaped[:, split_size : 2 * split_size, :, :]
        x_7x7 = input_reshaped[:, 2 * split_size : 3 * split_size :, :, :]
        x_9x9 = input_reshaped[:, 3 * split_size :, :, :]

        x_3x3 = self.dw_conv_3x3(x_3x3)
        x_5x5 = self.dw_conv_5x5(x_5x5)
        x_7x7 = self.dw_conv_7x7(x_7x7)
        x_9x9 = self.dw_conv_9x9(x_9x9)

        att_3x3 = self.P_lattention(x_3x3)
        att_5x5 = self.P_sfattention(x_5x5)
        att_7x7 = self.P_spattention(x_7x7)
        att_9x9 = self.P_cattention(x_9x9)

        processed_input = torch.cat(
            [
                att_3x3 * self.scale_weights[0],
                att_5x5 * self.scale_weights[1],
                att_7x7 * self.scale_weights[2],
                att_9x9 * self.scale_weights[3],
            ],
            dim=1,
        )
        final_output = self.final_conv(processed_input)

        return final_output


"""
MSCPA 多尺度卷积部分注意力模块通过结合多尺度卷积（3x3、5x5、7x7、9x9）
和多种注意力机制（如线性注意力、SE注意力、空间自注意力、位置感知自注意力），
有效地提升了模型对图像多尺度特征和复杂上下文关系的感知能力。它通过自适应加权和高效计算，
优化了特征提取与表示，特别适用于目标检测和图像分割等任务，增强了模型在处理大规模数据时的效率与精度。
"""
if __name__ == "__main__":
    input = torch.rand(1, 64, 32, 32)
    MSCPA = MSCPA(64)
    output = MSCPA(input)
    print("MSCPA input_size:", input.size())
    print("MSCPA output_size:", output.size())
