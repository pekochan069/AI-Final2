import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class ConvBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride=1,
        groups=1,
        bias=False,
    ):
        super().__init__()
        p = max(kernel_size - stride, 0)
        padding = [p // 2, p - p // 2, p // 2, p - p // 2]
        self.layers = nn.Sequential(
            nn.ZeroPad2d(padding),
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=0,
                groups=groups,
                bias=bias,
            ),
            nn.BatchNorm2d(out_channels),
        )
        self.gelu = nn.GELU()

    def forward(self, X):
        return self.gelu(self.layers(X))


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


class EfficientNet(nn.Module):
    def __init__(
        self,
        num_classes=10,
        width_factor=1.0,
        depth_factor=1.0,
        use_se=True,
        se_ratio=0.25,
        use_dropout=True,
        dropout_ratio=0.2,
    ):
        super().__init__()
        assert width_factor >= 1.0
        assert depth_factor >= 1.0
        assert se_ratio >= 0 and se_ratio <= 1
        assert dropout_ratio >= 0 and dropout_ratio <= 1

        layers = []

        block_args = [
            # depth, channels, expand_ratio, kernel, stride
            [1, 16, 1, 3, 1],
            [2, 24, 6, 3, 2],
            [2, 40, 6, 5, 2],
            [3, 80, 6, 3, 2],
            [3, 112, 6, 5, 1],
            [4, 192, 6, 5, 2],
            [1, 320, 6, 3, 1],
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

        out_channels = int(round(1280 * width_factor))

        layers.append(ConvBlock(in_channels, out_channels, kernel_size=1, bias=False))
        layers.append(nn.AdaptiveAvgPool2d((1, 1)))

        self.layers = nn.Sequential(*layers)

        classifier = []

        if use_dropout:
            classifier.append(nn.Dropout(dropout_ratio))

        classifier.append(nn.Linear(out_channels, num_classes))
        classifier.append(nn.Softmax(dim=1))

        self.classifier = nn.Sequential(*classifier)

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
        X = self.layers(X)
        X = X.mean([2, 3])
        X = self.classifier(X)
        return X
