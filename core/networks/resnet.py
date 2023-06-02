import torch
import torch.nn as nn


class ResBlock(nn.Module):
    def __init__(self, filter: int, use_subsample=True, use_shortcut=True):
        super().__init__()

        self.activation_fn = nn.CELU()

        self.use_shortcut = use_shortcut

        m = 0.5 if use_subsample else 1

        self.conv1 = nn.Conv2d(
            int(filter * m),
            filter,
            kernel_size=3,
            stride=int(1 / m),
            padding=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(filter, track_running_stats=True)
        self.conv2 = nn.Conv2d(
            filter, filter, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(filter, track_running_stats=True)

        # down-sampling using pooling(kernel = 1, stride = 2)
        self.downsample = nn.AvgPool2d(1, 2)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_uniform_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, X):
        if self.use_shortcut:
            I = X

        X = self.conv1(X)
        X = self.bn1(X)
        X = self.activation_fn(X)
        X = self.conv2(X)
        X = self.bn2(X)

        if self.use_shortcut:
            X = self.shortcut(X, I)

        X = self.activation_fn(X)

        return X

    def shortcut(self, X, I):
        if X.shape != I.shape:
            d = self.downsample(I)
            return X + torch.cat((d, torch.mul(d, 0)), dim=1)
        else:
            return X + I


class ResNet(nn.Module):
    def __init__(self, n: int, use_shortcut=True):
        super().__init__()

        self.activation_fn = nn.CELU()

        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16, track_running_stats=True)

        self.blocks1 = nn.ModuleList(
            [
                ResBlock(16, use_subsample=False, use_shortcut=use_shortcut)
                for _ in range(n)
            ]
        )

        self.blocks2_1 = ResBlock(32, use_subsample=True, use_shortcut=use_shortcut)
        self.blocks2_rest = nn.ModuleList(
            [
                ResBlock(32, use_subsample=False, use_shortcut=use_shortcut)
                for _ in range(n - 1)
            ]
        )

        self.blocks3_1 = ResBlock(64, use_subsample=True, use_shortcut=use_shortcut)
        self.blocks3_rest = nn.ModuleList(
            [
                ResBlock(64, use_subsample=False, use_shortcut=use_shortcut)
                for _ in range(n - 1)
            ]
        )

        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64, 10, bias=True)
        self.softmax = nn.LogSoftmax(dim=-1)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.kaiming_uniform_(m.weight)
                m.bias.data.zero_()

    def forward(self, X):
        X = self.activation_fn(self.bn1(self.conv1(X)))

        for block in self.blocks1:
            X = block(X)

        X = self.blocks2_1(X)
        for block in self.blocks2_rest:
            X = block(X)

        X = self.blocks3_1(X)
        for block in self.blocks3_rest:
            X = block(X)

        X = self.pool(X)
        X = X.view(X.size(0), -1)

        return self.softmax(self.fc(X))
