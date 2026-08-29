import torch

from models.prep_network import PreparationNetwork


model = PreparationNetwork()

secret = torch.randn(2, 3, 256, 256)

features = model(secret)

print("Input Shape :", secret.shape)
print("Output Shape:", features.shape)
