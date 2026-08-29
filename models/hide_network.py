import torch
import torch.nn as nn
from models.multiscale_block import MultiScaleBlock


class HidingNetwork(nn.Module):

    def __init__(self):
        super().__init__()

        self.block1 = MultiScaleBlock(53, 64)
        self.block2 = MultiScaleBlock(64, 64)
        self.block3 = MultiScaleBlock(64, 64)
        self.block4 = MultiScaleBlock(64, 32)

        self.output = nn.Conv2d(32, 3, kernel_size=1)

    def forward(self, cover, secret_features):

        x = torch.cat([cover, secret_features], dim=1)

        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)

        stego = self.output(x)

        return stego
