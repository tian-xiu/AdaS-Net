import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import deform_conv2d
from config import Config


config = Config()


class DeformableConv2d(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 stride=1,
                 padding=1,
                 bias=False):

        super(DeformableConv2d, self).__init__()
        
        assert type(kernel_size) == tuple or type(kernel_size) == int

        kernel_size = kernel_size if type(kernel_size) == tuple else (kernel_size, kernel_size)
        self.stride = stride if type(stride) == tuple else (stride, stride)
        self.padding = padding
        
        self.offset_conv = nn.Conv2d(in_channels,
                                     2 * kernel_size[0] * kernel_size[1],
                                     kernel_size=kernel_size,
                                     stride=stride,
                                     padding=self.padding,
                                     bias=True)

        nn.init.constant_(self.offset_conv.weight, 0.)
        nn.init.constant_(self.offset_conv.bias, 0.)
        
        self.modulator_conv = nn.Conv2d(in_channels,
                                     1 * kernel_size[0] * kernel_size[1],
                                     kernel_size=kernel_size,
                                     stride=stride,
                                     padding=self.padding,
                                     bias=True)

        nn.init.constant_(self.modulator_conv.weight, 0.)
        nn.init.constant_(self.modulator_conv.bias, 0.)

        self.regular_conv = nn.Conv2d(in_channels,
                                      out_channels=out_channels,
                                      kernel_size=kernel_size,
                                      stride=stride,
                                      padding=self.padding,
                                      bias=bias)

    def forward(self, x):
        offset = self.offset_conv(x)
        modulator = 2. * torch.sigmoid(self.modulator_conv(x))
        
        x = deform_conv2d(
            input=x,
            offset=offset,
            weight=self.regular_conv.weight,
            bias=self.regular_conv.bias,
            padding=self.padding,
            mask=modulator,
            stride=self.stride,
        )
        return x


def _norm_layer(channels):
    return nn.BatchNorm2d(channels) if config.batch_size > 1 else nn.Identity()


def _global_pool_block(in_channels, out_channels):
    return nn.Sequential(
        nn.AdaptiveAvgPool2d((1, 1)),
        nn.Conv2d(in_channels, out_channels, 1, stride=1, bias=False),
        _norm_layer(out_channels),
        nn.ReLU(inplace=True),
    )


class _SAPModule(nn.Module):
    def __init__(self, in_channels, planes, kernel_size, padding):
        super().__init__()
        self.atrous_conv = DeformableConv2d(
            in_channels,
            planes,
            kernel_size=kernel_size,
            stride=1,
            padding=padding,
            bias=False,
        )
        self.bn = _norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.atrous_conv(x)
        x = self.bn(x)

        return self.relu(x)


class SAP(nn.Module):
    def __init__(self, in_channels, out_channels=None, parallel_block_sizes=[1, 3, 7], rca_k_size=7):
        super().__init__()
        self.down_scale = 1
        if out_channels is None:
            out_channels = in_channels
        self.in_channelster = 256 // self.down_scale
        self.rca_k_size = rca_k_size

        self.aspp1 = _SAPModule(in_channels, self.in_channelster, 1, padding=0)
        self.aspp_deforms = nn.ModuleList([
            _SAPModule(in_channels, self.in_channelster, conv_size, padding=int(conv_size//2)) for conv_size in parallel_block_sizes
        ])

        self.global_avg_pool = _global_pool_block(in_channels, self.in_channelster)
        self.conv1 = nn.Conv2d(self.in_channelster * (2 + len(self.aspp_deforms)), out_channels, 1, bias=False)
        self.bn1 = _norm_layer(out_channels)
        self.relu = nn.ReLU(inplace=True)

        conv0_k = {7: 3, 11: 3, 23: 5, 35: 5, 41: 5, 53: 5}
        spatial_k = {7: 3, 11: 5, 23: 7, 35: 11, 41: 13, 53: 17}
        spatial_d = {7: 2, 11: 2, 23: 3, 35: 3, 41: 3, 53: 3}
        if rca_k_size not in conv0_k:
            raise ValueError(f'Unsupported rca_k_size: {rca_k_size}')

        c0 = conv0_k[rca_k_size]
        sk = spatial_k[rca_k_size]
        sd = spatial_d[rca_k_size]
        self.rca_conv0h = nn.Conv2d(out_channels, out_channels, kernel_size=(1, c0), stride=(1, 1), padding=(0, (c0 - 1) // 2), groups=out_channels)
        self.rca_conv0v = nn.Conv2d(out_channels, out_channels, kernel_size=(c0, 1), stride=(1, 1), padding=((c0 - 1) // 2, 0), groups=out_channels)
        self.rca_spatial_h = nn.Conv2d(out_channels, out_channels, kernel_size=(1, sk), stride=(1, 1), padding=(0, sd * (sk - 1) // 2), groups=out_channels, dilation=sd)
        self.rca_spatial_v = nn.Conv2d(out_channels, out_channels, kernel_size=(sk, 1), stride=(1, 1), padding=(sd * (sk - 1) // 2, 0), groups=out_channels, dilation=sd)

        self.rca_conv1 = nn.Conv2d(out_channels, out_channels, 1)
        self.dropout = nn.Dropout(0.5)

    def _apply_rca(self, x):
        residual = x
        attn = self.rca_conv0h(x)
        attn = self.rca_conv0v(attn)
        attn = self.rca_spatial_h(attn)
        attn = self.rca_spatial_v(attn)
        attn = self.rca_conv1(attn)
        return residual * attn

    def forward(self, x):
        x1 = self.aspp1(x)
        x_aspp_deforms = [aspp_deform(x) for aspp_deform in self.aspp_deforms]
        x5 = self.global_avg_pool(x)
        x5 = F.interpolate(x5, size=x1.size()[2:], mode='bilinear', align_corners=True)
        x = torch.cat((x1, *x_aspp_deforms, x5), dim=1)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self._apply_rca(x)

        return self.dropout(x)
