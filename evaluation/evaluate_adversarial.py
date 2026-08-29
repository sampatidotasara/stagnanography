import os

import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from skimage.metrics import peak_signal_noise_ratio
from skimage.metrics import structural_similarity

from models.steganography_model import SteganographyModel

# ============================================
# Configuration
# ============================================

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

MODEL_PATH = "checkpoints/adversarial_best.pth"

COVER_IMAGE = "test_images/cover.png"
SECRET_IMAGE = "test_images/secret.png"

RESULTS_DIR = "results"

os.makedirs(RESULTS_DIR, exist_ok=True)

# ============================================
# Image Transform
# ============================================

transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])

to_pil = transforms.ToPILImage()

# ============================================
# Load Images
# ============================================

cover = Image.open(COVER_IMAGE).convert("RGB")
secret = Image.open(SECRET_IMAGE).convert("RGB")

cover_tensor = transform(cover).unsqueeze(0).to(DEVICE)
secret_tensor = transform(secret).unsqueeze(0).to(DEVICE)

# ============================================
# Load Model
# ============================================

model = SteganographyModel().to(DEVICE)

model.load_state_dict(
    torch.load(
        MODEL_PATH,
        map_location=DEVICE
    )
)

model.eval()

# ============================================
# Inference
# ============================================

with torch.no_grad():

    stego_tensor, recovered_tensor = model(
        cover_tensor,
        secret_tensor
    )

# ============================================
# Save Images
# ============================================

stego_img = to_pil(
    stego_tensor.squeeze(0).cpu().clamp(0, 1)
)

recovered_img = to_pil(
    recovered_tensor.squeeze(0).cpu().clamp(0, 1)
)

stego_path = os.path.join(
    RESULTS_DIR,
    "adv_stego.png"
)

recovered_path = os.path.join(
    RESULTS_DIR,
    "adv_recovered.png"
)

stego_img.save(stego_path)
recovered_img.save(recovered_path)

print("Saved:", stego_path)
print("Saved:", recovered_path)

# ============================================
# Reload Saved Images
# ============================================

cover_np = np.array(
    cover.resize((256, 256))
)

secret_np = np.array(
    secret.resize((256, 256))
)

stego_np = np.array(stego_img)

recovered_np = np.array(recovered_img)

# ============================================
# Metrics
# ============================================

cover_psnr = peak_signal_noise_ratio(
    cover_np,
    stego_np,
    data_range=255
)

cover_ssim = structural_similarity(
    cover_np,
    stego_np,
    channel_axis=2,
    data_range=255
)

secret_psnr = peak_signal_noise_ratio(
    secret_np,
    recovered_np,
    data_range=255
)

secret_ssim = structural_similarity(
    secret_np,
    recovered_np,
    channel_axis=2,
    data_range=255
)

# ============================================
# Results
# ============================================

print("\n" + "=" * 60)
print("Adversarial Model Evaluation")
print("=" * 60)

print(f"Cover → Stego PSNR  : {cover_psnr:.2f} dB")
print(f"Cover → Stego SSIM  : {cover_ssim:.4f}")

print()

print(f"Secret → Recovered PSNR : {secret_psnr:.2f} dB")
print(f"Secret → Recovered SSIM : {secret_ssim:.4f}")

print("=" * 60)
