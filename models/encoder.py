import torch
import torch.nn as nn

from models.blocks import DoubleConv, Down, Up


class HidingNetwork(nn.Module):

    def __init__(self):
        super().__init__()

        self.inc = DoubleConv(6, 64)

        self.down1 = Down(64, 128)

        self.down2 = Down(128, 256)

        self.up1 = Up(256, 128)

        self.up2 = Up(128, 64)

        self.out = nn.Conv2d(64, 3, kernel_size=1)

    def forward(self, cover, secret):

        x = torch.cat([cover, secret], dim=1)

        x1 = self.inc(x)

        x2 = self.down1(x1)

        x3 = self.down2(x2)

        x = self.up1(x3, x2)

        x = self.up2(x, x1)

        stego = self.out(x)

        return stego
