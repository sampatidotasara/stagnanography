import torch

from models.encoder import HidingNetwork

model = HidingNetwork()

cover = torch.randn(2, 3, 256, 256)
secret = torch.randn(2, 3, 256, 256)

stego = model(cover, secret)

print("Stego Shape:", stego.shape)
