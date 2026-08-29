import os

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from tqdm import tqdm

from models.steganalyzer import Steganalyzer

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 16
EPOCHS = 20
LR = 1e-4


class StegoDataset(Dataset):
    def __init__(self, root):
        self.samples = []

        cover_dir = os.path.join(root, "cover")
        stego_dir = os.path.join(root, "stego")

        for file in os.listdir(cover_dir):
            self.samples.append((os.path.join(cover_dir, file), 0))

        for file in os.listdir(stego_dir):
            self.samples.append((os.path.join(stego_dir, file), 1))

        self.transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        img = self.transform(img)
        return img, torch.tensor([label], dtype=torch.float32)


dataset = StegoDataset("steganalyzer_dataset")

loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0
)

model = Steganalyzer().to(DEVICE)

criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=LR)

best_acc = 0

for epoch in range(EPOCHS):

    model.train()

    total_loss = 0
    correct = 0
    total = 0

    loop = tqdm(loader, desc=f"Epoch {epoch+1}/{EPOCHS}")

    for images, labels in loop:

        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        outputs = model(images)

        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        preds = (outputs > 0.5).float()

        correct += (preds == labels).sum().item()
        total += labels.size(0)

        loop.set_postfix(
            loss=loss.item(),
            accuracy=100 * correct / total
        )

    acc = 100 * correct / total

    print(
        f"Epoch {epoch+1}: "
        f"Loss={total_loss/len(loader):.4f} "
        f"Accuracy={acc:.2f}%"
    )

    if acc > best_acc:
        best_acc = acc

        os.makedirs("checkpoints", exist_ok=True)

        torch.save(
            model.state_dict(),
            "checkpoints/steganalyzer_best.pth"
        )

print("\nTraining Finished")
print(f"Best Accuracy: {best_acc:.2f}%")
