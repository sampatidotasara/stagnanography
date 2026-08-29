from utils import (
    hide_secret,
    load_model,
    image_to_tensor,
    tensor_to_image
)

import torch


def encode_image(cover_image, secret_image):
    """
    Encode a secret image into a cover image.
    """
    return hide_secret(cover_image, secret_image)


def decode_image(stego_image):
    """
    Recover the hidden image from a stego image.
    """

    model = load_model()

    stego = image_to_tensor(stego_image)

    with torch.no_grad():
        recovered = model.reveal(stego)

    return tensor_to_image(recovered)
