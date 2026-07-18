import einops
import torch
import torch.nn as nn
import torch.nn.functional as F


def autopad(k, p=None, d=1):
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p


class Conv(nn.Module):
    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        return self.act(self.conv(x))


class ConvBNAct(nn.Module):
    def __init__(
        self,
        in_chs,
        out_chs,
        kernel_size=3,
        stride=1,
        groups=1,
    ):
        super().__init__()
        self.conv = nn.Conv2d(
            in_chs, out_chs, kernel_size, stride, padding=(kernel_size - 1) // 2, groups=groups, bias=False
        )
        self.bn = nn.BatchNorm2d(out_chs)
        self.act = nn.GELU()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        return x


class CKConv(nn.Module):
    def __init__(self, c1, c2, kk=[3, 5, 7], s=1):
        super().__init__()

        if not isinstance(kk, list) or not all(ki in [3, 5, 7, 9] for ki in kk):
            raise ValueError("k must be a list containing 3, 5, and/or 7")

        self.kk = kk
        self.c1 = c1
        self.c2 = c2
        self.s = s
        self.branches = nn.ModuleDict()
        for ki in kk:
            self.branches[f"k{ki}_body"] = Conv(c2, c2, (3, 3), s=1, g=c2)
            self.branches[f"k{ki}_head_h"] = Conv(c2, c2, (1, ki), s=s, p=(0, (ki - 1) // 2), g=c2)
            self.branches[f"k{ki}_head_v"] = Conv(c2, c2, (ki, 1), s=s, p=((ki - 1) // 2, 0), g=c2)
            self.branches[f"k{ki}_conv2"] = nn.Conv2d(c2, c2, 1, groups=c2)
        self.conv_fuse = nn.Conv2d(len(kk) * c2, c2, 1, groups=16)

    def forward(self, x):
        outputs = []
        for ki in self.kk:
            y = self.branches[f"k{ki}_head_h"](x)
            y = self.branches[f"k{ki}_head_v"](y)
            ys = self.branches[f"k{ki}_body"](x)
            out = ys + y
            out = self.branches[f"k{ki}_conv2"](out)
            outputs.append(out)
        out = torch.cat(outputs, dim=1)
        out = self.conv_fuse(out)
        return out


class PConv(nn.Module):  # 风车卷积
    """Pinwheel-shaped Convolution using the Asymmetric Padding method."""

    def __init__(self, c1, c2, k=3, s=1):
        super().__init__()

        # 定义4种非对称填充方式，用于风车形状卷积的实现
        p = [(k, 0, 1, 0), (0, k, 0, 1), (0, 1, k, 0), (1, 0, 0, k)]  # 每个元组表示 (左, 上, 右, 下) 填充
        self.pad = [nn.ZeroPad2d(padding=(p[g])) for g in range(4)]  # 创建4个填充层

        # 定义水平方向卷积操作，卷积核大小为 (1, k)，步幅为 s，输出通道数为 c2 // 4
        self.cw = Conv(c1, c2 // 4, (1, k), s=s, p=0)

        # 定义垂直方向卷积操作，卷积核大小为 (k, 1)，步幅为 s，输出通道数为 c2 // 4
        self.ch = Conv(c1, c2 // 4, (k, 1), s=s, p=0)

        # 最终合并卷积结果的卷积层，卷积核大小为 (2, 2)，输出通道数为 c2
        self.cat = Conv(c2, c2, 2, s=1, p=0)

    def forward(self, x):
        # 对输入 x 进行不同填充和卷积操作，得到四个方向的特征
        yw0 = self.cw(self.pad[0](x))  # 水平方向，第一个填充方式
        yw1 = self.cw(self.pad[1](x))  # 水平方向，第二个填充方式
        yh0 = self.ch(self.pad[2](x))  # 垂直方向，第一个填充方式
        yh1 = self.ch(self.pad[3](x))  # 垂直方向，第二个填充方式

        # 将四个卷积结果在通道维度拼接，并通过一个额外的卷积层处理，最终输出
        return self.cat(torch.cat([yw0, yw1, yh0, yh1], dim=1))  # 在通道维度拼接，并通过 cat 卷积层处理


class EdgeConv(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels, kernel_size=3, bias=True):
        super().__init__()

        self.in_proj = nn.Conv2d(in_channels=in_channels, out_channels=mid_channels, kernel_size=1, bias=bias)
        self.w_conv = nn.Conv2d(
            mid_channels,
            mid_channels,
            kernel_size=(1, kernel_size),
            stride=1,
            padding=(0, kernel_size // 2),
            groups=mid_channels,
        )

        self.h_conv = nn.Conv2d(
            mid_channels,
            mid_channels,
            kernel_size=(kernel_size, 1),
            stride=1,
            padding=(kernel_size // 2, 0),
            groups=mid_channels,
        )

        self.out_proj = nn.Conv2d(in_channels=mid_channels * 2, out_channels=out_channels, kernel_size=1, bias=True)

    def forward(self, x):
        x = self.in_proj(x)
        x_w = self.w_conv(x)
        x_h = self.h_conv(x)
        x = torch.cat([x_w, x_h], dim=1)
        x = self.out_proj(x)
        return x


class GMSKConv(nn.Module):
    def __init__(self, in_dim, nbins=36, cell_size=(4, 4)):

        super().__init__()
        self.nbins = nbins
        self.cell_size = cell_size

        self.hog_feat = nn.Sequential(
            nn.Conv2d(nbins, in_dim, kernel_size=1),
            nn.Conv2d(in_dim, in_dim, kernel_size=3, padding=1, groups=in_dim, bias=False),
            nn.GroupNorm(in_dim // 8, in_dim),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )

        self.weight = nn.Sequential(CKConv(c1=in_dim, c2=in_dim), nn.GroupNorm(in_dim // 8, in_dim))

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1, stride=1),
            nn.GroupNorm(in_dim // 8, in_dim),
        )

        self.fuse_block = nn.Sequential(CKConv(c1=in_dim, c2=in_dim), nn.GroupNorm(in_dim // 8, in_dim))

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        residual = x

        x = image2patches(x)

        x_hog = self.get_hog_feature(x)
        x_hog = x_hog.to(dtype=x.dtype)
        x_hog = self.hog_feat(x_hog)

        x1 = self.sigmoid(self.weight(x + x_hog))
        x2 = self.conv(x)
        x = x1 * x2

        x = patches2image(x)

        x = x + residual
        x = self.fuse_block(x)

        return x

    def get_hog_feature(self, x):
        x_mean = x.mean(dim=1, keepdim=True)
        B, _, H, W = x_mean.shape
        device = x_mean.device

        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3).to(device)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).view(1, 1, 3, 3).to(device)
        dx = F.conv2d(x_mean.float(), sobel_x, padding=1)  # b, 1, h, w
        dy = F.conv2d(x_mean.float(), sobel_y, padding=1)

        # direction
        gradient_dir = torch.atan2(dy, dx)  # [-π，π]
        gradient_dir = torch.abs(gradient_dir)  # [0，π]

        # cells
        cell_h, cell_w = self.cell_size
        H_cells = int(H / cell_h)
        W_cells = int(W / cell_w)

        #
        dirs_crop = gradient_dir[:, :, : H_cells * cell_h, : W_cells * cell_w]

        # Slicing can produce a non-contiguous tensor, so use reshape instead of view.
        dirs = dirs_crop.reshape(B, H_cells, W_cells, -1)

        bin_with = torch.pi / self.nbins
        bin_indices = (dirs / bin_with).floor().long()
        bin_indices = torch.clamp(bin_indices, 0, self.nbins - 1)

        bin_indices_flat = bin_indices.reshape(B * H_cells * W_cells, dirs.shape[-1])
        weight = []
        for i in range(bin_indices_flat.shape[0]):
            bins = bin_indices_flat[i]
            count = torch.bincount(bins, minlength=self.nbins)
            weight.append(count)

        weight = torch.stack(weight, dim=0).reshape(B, H_cells, W_cells, -1) / 64  # B, H_cells , W_cells, self.bins

        start = torch.pi / (2 * self.nbins)
        hog_feature = (
            torch.linspace(start, torch.pi - start, self.nbins).to(device).repeat(B, H_cells, W_cells, 1) * weight
        )

        return hog_feature.permute(0, 3, 1, 2)


def image2patches(x):
    """B c (hg h) (wg w) -> (hg wg b) c h w."""
    x = einops.rearrange(x, "b c (hg h) (wg w) -> (hg wg b) c h w", hg=2, wg=2)
    return x


def patches2image(x):
    """(hg wg b) c h w -> b c (hg h) (wg w)."""
    x = einops.rearrange(x, "(hg wg b) c h w -> b c (hg h) (wg w)", hg=2, wg=2)
    return x


# 改进后的模块名 GMSKConv：梯度引导多尺度条带核卷积模块（Gradient-guided Multi-scale Strip-Kernel Convolution）

# 为什么把原来的方向引导改成“梯度引导”？
"""
#因为原始 DEGConv 的核心是用梯度方向信息引导边缘门控，而此时用 CKConv或是PConv风车卷积 替换 EdgeConv，使边缘提取从简单的横向/纵向卷积升级为多尺度条带交叉核卷积。
#与原模块在命名方面更好的区分，更吸引审稿人对此模块的兴趣。不过模块命名大家也可以自由发挥，模块名尽量不要取的大众化就好了，然后好好结合自己任务目前存在的问题去编故事。
"""


"""
GMSKConv模块：为了进一步增强 DEGConv 对多尺度边缘结构的建模能力，使用 CKConv或是PConv风车卷积 替换 EdgeConv。
原 EdgeConv 主要通过水平和垂直方向的深度卷积提取边缘信息，感受野较为固定；
而 CKConv或是PConv风车卷积 引入多尺度交叉卷积分支，能够在不同感受野范围内捕获细粒度边缘、宽尺度边界和复杂纹理结构。
结合 DEGConv 中的 HOG 方向引导机制，该改进可以生成更加判别性的边缘门控权重，并提升残差融合后的结构表达能力，
从而增强模型对小目标、弱边缘和复杂背景区域的特征感知能力。
"""

if __name__ == "__main__":
    # 创建一个随机输入张量 (batch_size=4, channels=128, height=256, width=256)
    input = torch.randn(4, 128, 256, 256)
    # 创建一个 GMSKConv 实例
    model = GMSKConv(in_dim=128, nbins=90, cell_size=(4, 4))
    # nbins=90,cell_size=(4, 4) 这两个参数是参考论文表5设置的，效果最好。

    output = model(input)
    # 打印输出形状
    print("GMSKConv input_size:", input.size())  # [4, 128, 256, 256]
    print("GMSKConv output_size:", output.size())  # [4, 128, 256, 256]
