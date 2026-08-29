import torch

from models.reveal_network import RevealNetwork

model = RevealNetwork()

stego = torch.randn(2, 3, 256, 256)

secret = model(stego)

print("Stego :", stego.shape)
print("Recovered :", secret.shape)
