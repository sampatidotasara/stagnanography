import torch
import torch.nn as nn


class ConvBNReLU(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super().__init__()

        padding = kernel_size // 2

        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                padding=padding,
                bias=False
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)


class MultiScaleBlock(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()

        branch_channels = out_channels // 3

        self.branch3 = ConvBNReLU(in_channels, branch_channels, 3)
        self.branch5 = ConvBNReLU(in_channels, branch_channels, 5)
        self.branch7 = ConvBNReLU(in_channels, branch_channels, 7)

        self.fuse = nn.Sequential(
            nn.Conv2d(
                branch_channels * 3,
                out_channels,
                kernel_size=1,
                bias=False
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=1,
                bias=False
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):

        b1 = self.branch3(x)
        b2 = self.branch5(x)
        b3 = self.branch7(x)

        out = torch.cat([b1, b2, b3], dim=1)

        out = self.fuse(out)

        out = out + self.shortcut(x)

        return out
