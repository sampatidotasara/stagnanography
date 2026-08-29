import torch.nn as nn
from models.multiscale_block import MultiScaleBlock


class RevealNetwork(nn.Module):

    def __init__(self):
        super().__init__()

        self.block1 = MultiScaleBlock(3, 64)
        self.block2 = MultiScaleBlock(64, 64)
        self.block3 = MultiScaleBlock(64, 64)
        self.block4 = MultiScaleBlock(64, 32)

        self.output = nn.Conv2d(32, 3, kernel_size=1)

    def forward(self, stego):

        x = self.block1(stego)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)

        secret = self.output(x)

        return secret
