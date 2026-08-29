import torch

from models.multiscale_block import MultiScaleBlock

model = MultiScaleBlock(3, 64)

x = torch.randn(2, 3, 256, 256)

y = model(x)

print("Input :", x.shape)
print("Output:", y.shape)
