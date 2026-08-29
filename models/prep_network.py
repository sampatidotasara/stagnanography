import torch.nn as nn
from models.multiscale_block import MultiScaleBlock


class PreparationNetwork(nn.Module):

    def __init__(self):
        super().__init__()

        self.block1 = MultiScaleBlock(3, 64)
        self.block2 = MultiScaleBlock(64, 64)
        self.block3 = MultiScaleBlock(64, 64)

        # Convert features to 50 channels
        self.output = nn.Conv2d(64, 50, kernel_size=1)

    def forward(self, secret):

        x = self.block1(secret)
        x = self.block2(x)
        x = self.block3(x)

        features = self.output(x)

        return features
