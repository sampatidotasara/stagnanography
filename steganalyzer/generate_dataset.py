import os
from pathlib import Path

import torch
from torchvision import transforms
from PIL import Image

from dataset.div2k_dataset import DIV2KDataset
from models.steganography_model import SteganographyModel

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

OUTPUT_DIR = "steganalyzer_dataset"

os.makedirs(f"{OUTPUT_DIR}/cover", exist_ok=True)
os.makedirs(f"{OUTPUT_DIR}/stego", exist_ok=True)

model = SteganographyModel().to(DEVICE)
model.load_state_dict(torch.load("checkpoints/best_model.pth", map_location=DEVICE))
model.eval()

dataset = DIV2KDataset("data/DIV2K_train_HR")

to_pil = transforms.ToPILImage()

from tqdm import tqdm

with torch.no_grad():
    for i in tqdm(range(len(dataset)), desc="Generating Dataset"):

        cover, secret = dataset[i]

        cover_batch = cover.unsqueeze(0).to(DEVICE)
        secret_batch = secret.unsqueeze(0).to(DEVICE)

        stego, _ = model(cover_batch, secret_batch)

        cover_img = to_pil(cover)
        stego_img = to_pil(stego.squeeze(0).cpu())

        cover_img.save(f"{OUTPUT_DIR}/cover/{i:05d}.png")
        stego_img.save(f"{OUTPUT_DIR}/stego/{i:05d}.png")
print("Dataset generated successfully!")
