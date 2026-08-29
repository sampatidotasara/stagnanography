import os

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset.div2k_dataset import DIV2KDataset
from models.steganography_model import SteganographyModel
from models.steganalyzer import Steganalyzer


# ==========================================================
# Configuration
# ==========================================================

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

DATASET_PATH = "data/DIV2K_train_HR"

CHECKPOINT_DIR = "checkpoints"

GENERATOR_CHECKPOINT = os.path.join(
    CHECKPOINT_DIR,
    "best_model.pth"
)

STEGANALYZER_CHECKPOINT = os.path.join(
    CHECKPOINT_DIR,
    "steganalyzer_best.pth"
)

SAVE_MODEL = os.path.join(
    CHECKPOINT_DIR,
    "adversarial_best.pth"
)

EPOCHS = 20
BATCH_SIZE = 2

LR_GENERATOR = 1e-4
LR_STEGANALYZER = 1e-4

LAMBDA_ADV = 0.1


# ==========================================================
# Dataset
# ==========================================================

dataset = DIV2KDataset(DATASET_PATH)

loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0,
    pin_memory=torch.cuda.is_available()
)


# ==========================================================
# Models
# ==========================================================

generator = SteganographyModel().to(DEVICE)

generator.load_state_dict(
    torch.load(
        GENERATOR_CHECKPOINT,
        map_location=DEVICE
    )
)

steganalyzer = Steganalyzer().to(DEVICE)

steganalyzer.load_state_dict(
    torch.load(
        STEGANALYZER_CHECKPOINT,
        map_location=DEVICE
    )
)


# ==========================================================
# Loss Functions
# ==========================================================

mse_loss = nn.MSELoss()

bce_loss = nn.BCELoss()


# ==========================================================
# Optimizers
# ==========================================================

generator_optimizer = optim.Adam(
    generator.parameters(),
    lr=LR_GENERATOR
)

steganalyzer_optimizer = optim.Adam(
    steganalyzer.parameters(),
    lr=LR_STEGANALYZER
)


# ==========================================================
# Scheduler
# ==========================================================

generator_scheduler = optim.lr_scheduler.StepLR(
    generator_optimizer,
    step_size=10,
    gamma=0.5
)

steganalyzer_scheduler = optim.lr_scheduler.StepLR(
    steganalyzer_optimizer,
    step_size=10,
    gamma=0.5
)


# ==========================================================
# History
# ==========================================================

history = {
    "generator_loss": [],
    "cover_loss": [],
    "secret_loss": [],
    "adversarial_loss": [],
    "steganalyzer_loss": [],
    "steganalyzer_accuracy": []
}


best_generator_loss = float("inf")

print("=" * 60)
print("Adversarial Training")
print("=" * 60)
print("Device :", DEVICE)
print("Dataset Size :", len(dataset))
print("Epochs :", EPOCHS)
print("=" * 60)
# ==========================================================
# Training Loop
# ==========================================================

for epoch in range(EPOCHS):

    generator.train()
    steganalyzer.train()

    generator_loss_epoch = 0.0
    cover_loss_epoch = 0.0
    secret_loss_epoch = 0.0
    adversarial_loss_epoch = 0.0

    steganalyzer_loss_epoch = 0.0

    correct = 0
    total = 0

    loop = tqdm(
        loader,
        desc=f"Epoch {epoch+1}/{EPOCHS}"
    )

    for cover, secret in loop:

        cover = cover.to(DEVICE)
        secret = secret.to(DEVICE)

        # ==================================================
        # Generate Stego Image
        # ==================================================

        with torch.no_grad():

            stego, _ = generator(
                cover,
                secret
            )

        # ==================================================
        # Train Steganalyzer
        # ==================================================

        steganalyzer_optimizer.zero_grad()

        cover_prediction = steganalyzer(cover)

        cover_target = torch.zeros_like(
            cover_prediction
        )

        cover_loss_detector = bce_loss(
            cover_prediction,
            cover_target
        )

        stego_prediction = steganalyzer(
            stego.detach()
        )

        stego_target = torch.ones_like(
            stego_prediction
        )

        stego_loss_detector = bce_loss(
            stego_prediction,
            stego_target
        )

        detector_loss = (
            cover_loss_detector +
            stego_loss_detector
        ) / 2

        detector_loss.backward()

        steganalyzer_optimizer.step()

        steganalyzer_loss_epoch += detector_loss.item()

        # ==================================================
        # Detector Accuracy
        # ==================================================

        cover_pred = (
            cover_prediction > 0.5
        ).float()

        stego_pred = (
            stego_prediction > 0.5
        ).float()

        correct += (
            cover_pred == cover_target
        ).sum().item()

        correct += (
            stego_pred == stego_target
        ).sum().item()

        total += (
            cover_target.numel() +
            stego_target.numel()
        )
                # ==================================================
        # Train Generator (Freeze Steganalyzer)
        # ==================================================

        for param in steganalyzer.parameters():
            param.requires_grad = False

        generator_optimizer.zero_grad()

        # Forward pass through Generator
        stego, recovered = generator(
            cover,
            secret
        )

        # -----------------------------
        # Reconstruction Losses
        # -----------------------------
        cover_loss = mse_loss(
            stego,
            cover
        )

        secret_loss = mse_loss(
            recovered,
            secret
        )

        # -----------------------------
        # Adversarial Loss
        # Generator wants Steganalyzer
        # to classify Stego as Cover (0)
        # -----------------------------
        prediction = steganalyzer(stego)

        fake_labels = torch.zeros_like(
            prediction
        )

        adversarial_loss = bce_loss(
            prediction,
            fake_labels
        )

        # -----------------------------
        # Total Generator Loss
        # -----------------------------
        generator_loss = (
            cover_loss +
            secret_loss +
            LAMBDA_ADV * adversarial_loss
        )

        generator_loss.backward()

        generator_optimizer.step()

        # Enable Steganalyzer again
        for param in steganalyzer.parameters():
            param.requires_grad = True

        # -----------------------------
        # Statistics
        # -----------------------------
        generator_loss_epoch += generator_loss.item()
        cover_loss_epoch += cover_loss.item()
        secret_loss_epoch += secret_loss.item()
        adversarial_loss_epoch += adversarial_loss.item()

        loop.set_postfix(
            g_loss=f"{generator_loss.item():.4f}",
            d_loss=f"{detector_loss.item():.4f}",
            acc=f"{100 * correct / total:.2f}%"
        )

    # ======================================================
    # End of Epoch
    # ======================================================

    avg_generator_loss = generator_loss_epoch / len(loader)
    avg_cover_loss = cover_loss_epoch / len(loader)
    avg_secret_loss = secret_loss_epoch / len(loader)
    avg_adv_loss = adversarial_loss_epoch / len(loader)
    avg_detector_loss = steganalyzer_loss_epoch / len(loader)

    detector_accuracy = 100 * correct / total

    history["generator_loss"].append(avg_generator_loss)
    history["cover_loss"].append(avg_cover_loss)
    history["secret_loss"].append(avg_secret_loss)
    history["adversarial_loss"].append(avg_adv_loss)
    history["steganalyzer_loss"].append(avg_detector_loss)
    history["steganalyzer_accuracy"].append(detector_accuracy)

    print("\n" + "=" * 60)
    print(f"Epoch {epoch + 1}/{EPOCHS}")
    print("=" * 60)
    print(f"Generator Loss     : {avg_generator_loss:.6f}")
    print(f"Cover Loss         : {avg_cover_loss:.6f}")
    print(f"Secret Loss        : {avg_secret_loss:.6f}")
    print(f"Adversarial Loss   : {avg_adv_loss:.6f}")
    print(f"Detector Loss      : {avg_detector_loss:.6f}")
    print(f"Detector Accuracy  : {detector_accuracy:.2f}%")
    print("=" * 60)
        # ======================================================
    # Save Best Generator
    # ======================================================

    if avg_generator_loss < best_generator_loss:

        best_generator_loss = avg_generator_loss

        os.makedirs(CHECKPOINT_DIR, exist_ok=True)

        torch.save(
            generator.state_dict(),
            SAVE_MODEL
        )

        print("\n✅ Best adversarial model saved!")

    # ======================================================
    # Step Learning Rate Schedulers
    # ======================================================

    generator_scheduler.step()
    steganalyzer_scheduler.step()

    print(
        f"Generator LR   : "
        f"{generator_optimizer.param_groups[0]['lr']:.6f}"
    )

    print(
        f"Steganalyzer LR: "
        f"{steganalyzer_optimizer.param_groups[0]['lr']:.6f}"
    )


# ==========================================================
# Training Finished
# ==========================================================

print("\n" + "=" * 60)
print("Adversarial Training Finished")
print("=" * 60)

print(f"Best Generator Loss : {best_generator_loss:.6f}")
print(f"Best Model Saved At : {SAVE_MODEL}")

# ==========================================================
# Save Training History
# ==========================================================

import json

os.makedirs("results", exist_ok=True)

with open("results/adversarial_history.json", "w") as f:
    json.dump(history, f, indent=4)

print("Training history saved to results/adversarial_history.json")
