import torch.nn as nn

from models.prep_network import PreparationNetwork
from models.hide_network import HidingNetwork
from models.reveal_network import RevealNetwork


class SteganographyModel(nn.Module):

    def __init__(self):
        super().__init__()

        self.prep = PreparationNetwork()

        self.hide = HidingNetwork()

        self.reveal = RevealNetwork()

    def forward(self, cover, secret):

        secret_features = self.prep(secret)

        stego = self.hide(cover, secret_features)

        recovered_secret = self.reveal(stego)

        return stego, recovered_secret
