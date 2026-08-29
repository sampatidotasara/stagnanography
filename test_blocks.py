import torch

from models.blocks import DoubleConv

x = torch.randn(1, 6, 256, 256)

model = DoubleConv(6, 64)

y = model(x)

print(y.shape)
