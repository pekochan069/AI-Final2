import torch
import torch.nn as nn
import numpy as np


class ConvBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride=1,
        padding=0,
        groups=1,
        bias=False,
    ):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                groups=groups,
                bias=bias,
            ),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
        )

    def forward(self, X):
        return self.layers(X)


class DepthWiseConv2d(nn.Module):
    def __init__(
        self, in_channels: int, kernel_size: int, stride=1, padding=0, bias=False
    ):
        super().__init__()
        self.conv = ConvBlock(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=kernel_size,
            stride=stride,
            groups=in_channels,
            bias=bias,
        )

    def forward(self, X):
        return self.conv(X)


class PointWiseConv2d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, bias: bool = False):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels=in_channels, out_channels=out_channels, kernel_size=1, bias=bias
        )
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, X):
        return self.bn(self.conv(X))


class SqueezeAndExcitation(nn.Module):
    def __init__(self, in_channels: int, se_ratio: float):
        super().__init__()
        squeezed_channels = max(1, int(in_channels * se_ratio))

        self.layers = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, squeezed_channels, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(squeezed_channels, in_channels, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, X):
        return X * self.layers(X)


class DropConnect(nn.Module):
    def __init__(self, p: float):
        super().__init__()
        self.p = p
        self.dropout = nn.Dropout(p)

    def forward(self, X):
        return self.dropout(X) * (1 - self.p)


class MBConvBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        expand_ratio: int = 1,
        kernel_size=3,
        stride=1,
        use_se=False,
        se_ratio=0.25,
        drop_connect_ratio=0.2,
    ):
        assert stride in (1, 2)
        assert kernel_size in (3, 5)
        assert expand_ratio > 0
        assert se_ratio >= 0 and se_ratio <= 1
        assert drop_connect_ratio >= 0 and drop_connect_ratio <= 1

        super().__init__()

        self.use_shortcut = in_channels == out_channels and stride == 1

        expanded_channels = in_channels * expand_ratio

        layers = []

        if expand_ratio > 1:
            layers.append(
                ConvBlock(in_channels, expanded_channels, kernel_size=1, bias=False)
            )

        layers.append(
            DepthWiseConv2d(
                expanded_channels, kernel_size=kernel_size, stride=stride, padding=1
            )
        )

        if use_se:
            layers.append(SqueezeAndExcitation(expanded_channels, se_ratio))

        layers.append(PointWiseConv2d(expanded_channels, out_channels, bias=False))

        if self.training and self.use_shortcut:
            layers.append(DropConnect(drop_connect_ratio))

        self.layers = nn.Sequential(*layers)

    def forward(self, X):
        if self.use_shortcut:
            return X + self.layers(X)
        else:
            return self.layers(X)


class Generator(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        pass


class Discriminator(nn.Module):
    # ResNet with Efficient scaling from EfficientNet
    def __init__(
        self,
        phi: int = 0,
        use_se=True,
        se_ratio=0.25,
        use_dropout=True,
        dropout_ratio=0.2,
    ):
        super().__init__()

        width_factor = int(round(pow(1.1, phi)))
        depth_factor = int(round(pow(1.2, phi)))

        layers = []

        block_args = [
            [1, 16, 1, 3, 1],
            [2, 32, 6, 3, 2],
            [2, 64, 6, 5, 2],
            [3, 128, 6, 3, 2],
        ]

        in_channels = int(round(32 * width_factor))
        out_channels = in_channels

        layers.append(ConvBlock(3, in_channels, kernel_size=3, stride=2, bias=False))

        for d, c, e, k, s in block_args:
            out_channels = int(round(c * width_factor))
            depth = int(round(d * depth_factor))

            for _ in range(depth):
                layers.append(
                    MBConvBlock(
                        in_channels,
                        out_channels,
                        expand_ratio=e,
                        kernel_size=k,
                        stride=s,
                        use_se=use_se,
                        se_ratio=se_ratio,
                        drop_connect_ratio=dropout_ratio,
                    )
                )

                in_channels = out_channels
                s = 1

            in_channels = out_channels

        layers.append(ConvBlock(in_channels, out_channels, kernel_size=1, bias=False))
        layers.append(nn.AdaptiveAvgPool2d((1, 1)))

        if use_dropout:
            layers.append(nn.Dropout(dropout_ratio))

        layers.append(nn.Linear(out_channels, 1))
        layers.append(nn.Sigmoid())

        self.layers = nn.Sequential(*layers)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                fan_out = m.weight.size(0)
                init_range = 1.0 / np.sqrt(fan_out)
                nn.init.uniform_(m.weight, -init_range, init_range)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, X):
        return self.layers(X)


class SRGAN(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        pass
