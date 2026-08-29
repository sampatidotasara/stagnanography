import os

import numpy as np
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio
from skimage.metrics import structural_similarity


def load_image(path):
    img = Image.open(path).convert("RGB")
    return np.array(img)


cover = load_image("results/cover.png")
stego = load_image("results/stego.png")

secret = load_image("results/secret.png")
recovered = load_image("results/recovered.png")


# -----------------------------
# Cover vs Stego
# -----------------------------
cover_psnr = peak_signal_noise_ratio(
    cover,
    stego,
    data_range=255
)

cover_ssim = structural_similarity(
    cover,
    stego,
    channel_axis=2,
    data_range=255
)


# -----------------------------
# Secret vs Recovered
# -----------------------------
secret_psnr = peak_signal_noise_ratio(
    secret,
    recovered,
    data_range=255
)

secret_ssim = structural_similarity(
    secret,
    recovered,
    channel_axis=2,
    data_range=255
)


print("=" * 50)
print("Evaluation Results")
print("=" * 50)

print(f"Cover -> Stego PSNR : {cover_psnr:.2f} dB")
print(f"Cover -> Stego SSIM : {cover_ssim:.4f}")

print()

print(f"Secret -> Recovered PSNR : {secret_psnr:.2f} dB")
print(f"Secret -> Recovered SSIM : {secret_ssim:.4f}")

print("=" * 50)
