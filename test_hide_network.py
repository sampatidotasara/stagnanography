import torch

from models.hide_network import HidingNetwork

model = HidingNetwork()

cover = torch.randn(2, 3, 256, 256)

secret_features = torch.randn(2, 50, 256, 256)

stego = model(cover, secret_features)

print("Cover :", cover.shape)

print("Secret Features :", secret_features.shape)

print("Stego :", stego.shape)
