import torch

from models.steganography_model import SteganographyModel

model = SteganographyModel()

cover = torch.randn(2, 3, 256, 256)

secret = torch.randn(2, 3, 256, 256)

stego, recovered = model(cover, secret)

print("Cover      :", cover.shape)
print("Secret     :", secret.shape)
print("Stego      :", stego.shape)
print("Recovered  :", recovered.shape)
