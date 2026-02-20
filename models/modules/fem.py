import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from config import Config


config = Config()


class FEM(nn.Module):
    def __init__(self, in_channels=64, out_channels=64, inter_channels=64):
        super(FEM, self).__init__()
        inter_channels = in_channels // 4 if config.dec_channels_inter == 'adap' else 64
        self.conv = nn.Conv2d(in_channels, out_channels, 1, 1, 0)
        self.enable_mixer = out_channels >= 2 and out_channels % 2 == 0
        if self.enable_mixer:
            self.mixer_convs = nn.ModuleList([])
            min_ch = out_channels // 2
            for ks in [3, 5]:
                self.mixer_convs.append(DynamicContextFusion(min_ch, ks, ks * 3 + 2))
            self.mixer_conv_1x1 = Conv(out_channels, out_channels, k=1)

    def forward(self, x):
        x = self.conv(x)
        if self.enable_mixer:
            _, c, _, _ = x.size()
            x_group = torch.split(x, [c // 2, c // 2], dim=1)
            x_group = torch.cat([self.mixer_convs[i](x_group[i]) for i in range(len(self.mixer_convs))], dim=1)
            x = self.mixer_conv_1x1(x_group)
        return x


class DynamicContextFusion(nn.Module):
    def __init__(self, in_channels, square_kernel_size=3, band_kernel_size=11):
        super().__init__()
        self.dwconv = nn.ModuleList([
            nn.Conv2d(in_channels, in_channels, square_kernel_size, padding=square_kernel_size // 2, groups=in_channels),
            nn.Conv2d(in_channels, in_channels, kernel_size=(1, band_kernel_size), padding=(0, band_kernel_size // 2), groups=in_channels),
            nn.Conv2d(in_channels, in_channels, kernel_size=(band_kernel_size, 1), padding=(band_kernel_size // 2, 0), groups=in_channels),
        ])

        self.bn = nn.BatchNorm2d(in_channels)
        self.act = nn.SiLU()

        self.dkw = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels * 3, 1),
        )

    def forward(self, x):
        x_dkw = rearrange(self.dkw(x), 'bs (g ch) h w -> g bs ch h w', g=3)
        x_dkw = F.softmax(x_dkw, dim=0)
        x = torch.stack([self.dwconv[i](x) * x_dkw[i] for i in range(len(self.dwconv))]).sum(0)
        return self.act(self.bn(x))


class Conv(nn.Module):
    default_act = nn.SiLU()

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        return self.act(self.conv(x))


def autopad(k, p=None, d=1):
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]
    return p


