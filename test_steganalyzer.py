import torch

from models.steganalyzer import Steganalyzer

model = Steganalyzer()

x = torch.randn(8, 3, 256, 256)

y = model(x)

print("Input :", x.shape)
print("Output:", y.shape)
