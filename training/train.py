import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm

from dataset.div2k_dataset import DIV2KDataset
from models.steganography_model import SteganographyModel


# ===========================
# Configuration
# ===========================

TRAIN_DIR = "data/DIV2K_train_HR"

BATCH_SIZE = 4
EPOCHS = 50
LEARNING_RATE = 1e-3

ALPHA = 1.0   # Cover loss weight
BETA = 1.0    # Secret loss weight

CHECKPOINT_DIR = "checkpoints"

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

os.makedirs(CHECKPOINT_DIR, exist_ok=True)

print("=" * 50)
print("Device :", DEVICE)
print("=" * 50)


# ===========================
# Dataset
# ===========================

train_dataset = DIV2KDataset(
    image_dir=TRAIN_DIR
)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0,
    pin_memory=True
)

print("Training Images :", len(train_dataset))


# ===========================
# Model
# ===========================

model = SteganographyModel().to(DEVICE)

print(model)


# ===========================
# Loss
# ===========================

cover_loss_fn = nn.MSELoss()

secret_loss_fn = nn.MSELoss()


# ===========================
# Optimizer
# ===========================

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)

scheduler = StepLR(
    optimizer,
    step_size=10,
    gamma=0.5
)


# ===========================
# Training
# ===========================

best_loss = float("inf")

history = []

print("\nTraining Started...\n")


for epoch in range(EPOCHS):

    model.train()

    epoch_loss = 0
    epoch_cover = 0
    epoch_secret = 0

    progress = tqdm(
        train_loader,
        desc=f"Epoch {epoch+1}/{EPOCHS}"
    )

    for cover, secret in progress:

        cover = cover.to(DEVICE)

        secret = secret.to(DEVICE)

        optimizer.zero_grad()

        stego, recovered = model(
            cover,
            secret
        )

        cover_loss = cover_loss_fn(
            stego,
            cover
        )

        secret_loss = secret_loss_fn(
            recovered,
            secret
        )

        total_loss = (
            ALPHA * cover_loss +
            BETA * secret_loss
        )

        total_loss.backward()

        optimizer.step()

        epoch_loss += total_loss.item()

        epoch_cover += cover_loss.item()

        epoch_secret += secret_loss.item()

        progress.set_postfix(
            loss=f"{total_loss.item():.4f}"
        )

    scheduler.step()

    avg_loss = epoch_loss / len(train_loader)

    avg_cover = epoch_cover / len(train_loader)

    avg_secret = epoch_secret / len(train_loader)

    history.append(avg_loss)

    print("\n--------------------------------")

    print(f"Epoch {epoch+1}/{EPOCHS}")

    print(f"Average Loss : {avg_loss:.6f}")

    print(f"Cover Loss   : {avg_cover:.6f}")

    print(f"Secret Loss  : {avg_secret:.6f}")

    print("--------------------------------\n")

    # Save every epoch

    torch.save(
        model.state_dict(),
        os.path.join(
            CHECKPOINT_DIR,
            f"epoch_{epoch+1}.pth"
        )
    )

    # Save best model

    if avg_loss < best_loss:

        best_loss = avg_loss

        torch.save(
            model.state_dict(),
            os.path.join(
                CHECKPOINT_DIR,
                "best_model.pth"
            )
        )

        print("Best model updated.")

print("\nTraining Finished!")

print("Best Loss :", best_loss)

torch.save(
    model.state_dict(),
    os.path.join(
        CHECKPOINT_DIR,
        "final_model.pth"
    )
)

print("Final model saved.")
