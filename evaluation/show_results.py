import os
import matplotlib.pyplot as plt
from PIL import Image

cover = Image.open("results/cover.png")
secret = Image.open("results/secret.png")
stego = Image.open("results/stego.png")
recovered = Image.open("results/recovered.png")

images = [
    ("Cover", cover),
    ("Secret", secret),
    ("Stego", stego),
    ("Recovered", recovered),
]

plt.figure(figsize=(12, 8))

for i, (title, img) in enumerate(images):
    plt.subplot(2, 2, i + 1)
    plt.imshow(img)
    plt.title(title)
    plt.axis("off")

plt.tight_layout()

os.makedirs("results", exist_ok=True)
plt.savefig("results/comparison.png", dpi=300, bbox_inches="tight")

print("Comparison image saved to results/comparison.png")
