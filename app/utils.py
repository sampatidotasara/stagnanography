import os
import sys
import io

import torch
from PIL import Image
from torchvision import transforms

from skimage.metrics import (
    peak_signal_noise_ratio,
    structural_similarity
)

# -------------------------------------------------------
# Project Root
# -------------------------------------------------------

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

PROJECT_ROOT = os.path.abspath(
    os.path.join(CURRENT_DIR, "..")
)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# -------------------------------------------------------
# Import Model
# -------------------------------------------------------

from models.steganography_model import SteganographyModel


# -------------------------------------------------------
# Device
# -------------------------------------------------------

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# -------------------------------------------------------
# Model Path
# -------------------------------------------------------

MODEL_PATH = os.path.join(
    PROJECT_ROOT,
    "checkpoints",
    "adversarial_best.pth"
)


# -------------------------------------------------------
# Transform
# -------------------------------------------------------

transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])


_model = None


# -------------------------------------------------------
# Load Model
# -------------------------------------------------------

def load_model():

    global _model

    if _model is None:

        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(
                f"Model checkpoint not found: {MODEL_PATH}"
            )

        model = SteganographyModel()

        checkpoint = torch.load(
            MODEL_PATH,
            map_location=DEVICE,
            weights_only=True
        )

        # Handle checkpoint formats
        if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
            checkpoint = checkpoint["state_dict"]

        model.load_state_dict(checkpoint)

        model.to(DEVICE)

        model.eval()

        _model = model

    return _model


# -------------------------------------------------------
# PIL Image -> Tensor
# -------------------------------------------------------

def image_to_tensor(image):

    image = image.convert("RGB")

    tensor = transform(image)

    tensor = tensor.unsqueeze(0)

    return tensor.to(DEVICE)


# -------------------------------------------------------
# Tensor -> PIL Image
# -------------------------------------------------------

def tensor_to_image(tensor):

    tensor = tensor.squeeze(0)

    tensor = tensor.detach().cpu()

    tensor = torch.clamp(tensor, 0, 1)

    return transforms.ToPILImage()(tensor)


# -------------------------------------------------------
# Hide Secret
# -------------------------------------------------------

def hide_secret(cover_image, secret_image):

    model = load_model()

    cover = image_to_tensor(cover_image)

    secret = image_to_tensor(secret_image)

    with torch.inference_mode():

        stego, recovered = model(
            cover,
            secret
        )

    return (
        tensor_to_image(stego),
        tensor_to_image(recovered),
        cover.squeeze(0).cpu(),
        secret.squeeze(0).cpu(),
        stego.squeeze(0).cpu(),
        recovered.squeeze(0).cpu()
    )


# -------------------------------------------------------
# Tensor -> NumPy
# -------------------------------------------------------

def tensor_to_numpy(tensor):

    return (
        tensor.detach()
        .cpu()
        .permute(1, 2, 0)
        .numpy()
    )


# -------------------------------------------------------
# PSNR
# -------------------------------------------------------

def calculate_psnr(original, generated):

    return peak_signal_noise_ratio(
        tensor_to_numpy(original),
        tensor_to_numpy(generated),
        data_range=1.0
    )


# -------------------------------------------------------
# SSIM
# -------------------------------------------------------

def calculate_ssim(original, generated):

    return structural_similarity(
        tensor_to_numpy(original),
        tensor_to_numpy(generated),
        channel_axis=2,
        data_range=1.0
    )


# -------------------------------------------------------
# Evaluation
# -------------------------------------------------------

def evaluate(cover, secret, stego, recovered):

    return {

        "cover_psnr": round(
            calculate_psnr(cover, stego),
            2
        ),

        "cover_ssim": round(
            calculate_ssim(cover, stego),
            4
        ),

        "secret_psnr": round(
            calculate_psnr(secret, recovered),
            2
        ),

        "secret_ssim": round(
            calculate_ssim(secret, recovered),
            4
        )
    }


# -------------------------------------------------------
# Download Bytes
# -------------------------------------------------------

def get_download_bytes(image):

    buffer = io.BytesIO()

    image.save(
        buffer,
        format="PNG"
    )

    buffer.seek(0)

    return buffer.getvalue()
