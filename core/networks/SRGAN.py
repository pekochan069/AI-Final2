import torch.nn as nn
import torch.nn.functional as F


class GeneratorResBlock(nn.Module):
    def __init__(self):
        super().__init__()

        layers = []

        layers.append(
            nn.Conv2d(
                in_channels=64,
                out_channels=64,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            )
        )
        layers.append(nn.BatchNorm2d(64))
        layers.append(nn.PReLU())
        layers.append(
            nn.Conv2d(
                in_channels=64,
                out_channels=64,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            )
        )
        layers.append(nn.BatchNorm2d(64))

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return x + self.layers(x)


class Generator(nn.Module):
    def __init__(self, B: int = 16):
        super().__init__()

        layers1 = []
        layers2 = []
        layers3 = []

        layers1.append(
            nn.Conv2d(
                in_channels=3,
                out_channels=64,
                kernel_size=9,
                stride=1,
                padding=4,
                bias=False,
            )
        )
        layers1.append(nn.PReLU())

        for _ in range(B):
            layers2.append(GeneratorResBlock())

        layers2.append(
            nn.Conv2d(
                in_channels=64,
                out_channels=64,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            )
        )
        layers2.append(nn.BatchNorm2d(64))

        layers3.append(
            nn.Conv2d(
                in_channels=64,
                out_channels=256,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            )
        )
        layers3.append(nn.PixelShuffle(2))
        layers3.append(nn.PReLU())

        layers3.append(
            nn.Conv2d(
                in_channels=64,
                out_channels=256,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            )
        )
        layers3.append(nn.PixelShuffle(2))
        layers3.append(nn.PReLU())

        layers3.append(
            nn.Conv2d(
                in_channels=64,
                out_channels=3,
                kernel_size=9,
                stride=1,
                padding=4,
                bias=False,
            )
        )

        self.layers1 = nn.Sequential(*layers1)
        self.layers2 = nn.Sequential(*layers2)
        self.layers3 = nn.Sequential(*layers3)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, X):
        X = self.layers1(X)
        X = X + self.layers2(X)
        return self.layers3(X)


class DiscriminatorConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super().__init__()

        layers = []

        layers.append(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=1,
                bias=False,
            )
        )
        layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.LeakyReLU(0.2, inplace=True))

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()

        layers = []

        layers.append(
            nn.Conv2d(
                in_channels=3,
                out_channels=64,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            )
        )
        layers.append(nn.LeakyReLU(0.2, inplace=True))

        layers.append(DiscriminatorConvBlock(64, 64, 3, 2))
        layers.append(DiscriminatorConvBlock(64, 128, 3, 1))
        layers.append(DiscriminatorConvBlock(128, 128, 3, 2))
        layers.append(DiscriminatorConvBlock(128, 256, 3, 1))
        layers.append(DiscriminatorConvBlock(256, 256, 3, 2))
        layers.append(DiscriminatorConvBlock(256, 512, 3, 1))
        layers.append(DiscriminatorConvBlock(512, 512, 3, 2))
        layers.append(nn.AdaptiveAvgPool2d(1))

        layers.append(nn.Conv2d(512, 1024, 1, bias=False))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        layers.append(nn.Conv2d(1024, 1, 1, bias=False))
        layers.append(nn.Dropout2d(0.2, inplace=True))

        self.layers = nn.Sequential(*layers)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, X):
        X = self.layers(X)
        X = X.flatten(1)
        return F.sigmoid(X)
