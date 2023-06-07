import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torchvision


class FeatureExtractor(nn.Module):
    def __init__(self, net, feature_layer=31):
        super(FeatureExtractor, self).__init__()
        self.features = nn.Sequential(
            *list(net.features.children())[: (feature_layer + 1)]
        )

    def forward(self, x):
        return self.features(x)


class GResBlock(nn.Module):
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
    def __init__(self, B: int = 5, upscale_factor=4):
        super().__init__()

        assert upscale_factor in [2, 4, 8], "Upscale factor must be 2, 4, or 8"
        upscale_layers = {2: 1, 4: 2, 8: 3}

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
            layers2.append(GResBlock())

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

        for _ in range(upscale_layers[upscale_factor]):
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

        # for m in self.modules():
        #     if isinstance(m, nn.Conv2d):
        #         nn.init.kaiming_normal_(m.weight, mode="fan_in")

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight.data, 0.0, 0.02)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.normal_(m.weight.data, 1.0, 0.02)
                nn.init.constant_(m.bias.data, 0)

    def forward(self, X):
        X = self.layers1(X)
        X = X + self.layers2(X)
        return self.layers3(X)


class DConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super().__init__()

        layers = []

        layers.append(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=3,
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
        classifier = []

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

        layers.append(DConvBlock(64, 64, 2))
        layers.append(DConvBlock(64, 128, 1))
        layers.append(DConvBlock(128, 128, 2))
        layers.append(DConvBlock(128, 256, 1))
        layers.append(DConvBlock(256, 256, 2))
        layers.append(DConvBlock(256, 512, 1))
        layers.append(DConvBlock(512, 512, 2))

        classifier.append(nn.AdaptiveAvgPool2d(1))
        classifier.append(nn.Conv2d(512, 1024, 1, bias=False))
        classifier.append(nn.LeakyReLU(0.2, inplace=True))
        classifier.append(nn.Conv2d(1024, 1, 1, bias=False))
        classifier.append(nn.Dropout2d(0.25))

        self.layers = nn.Sequential(*layers)
        self.classifier = nn.Sequential(*classifier)

        # for m in self.modules():
        #     if isinstance(m, nn.Conv2d):
        #         nn.init.kaiming_normal_(m.weight, mode="fan_in")

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight.data, 0.0, 0.02)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.normal_(m.weight.data, 1.0, 0.02)
                nn.init.constant_(m.bias.data, 0)

    def forward(self, X):
        X = self.layers(X)
        X = self.classifier(X)
        X = X.flatten(1)
        return F.sigmoid(X)
