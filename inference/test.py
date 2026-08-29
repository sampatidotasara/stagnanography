import os
import torch
from torchvision import transforms
from PIL import Image

from models.steganography_model import SteganographyModel

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])

to_pil = transforms.ToPILImage()

cover = Image.open("test_images/cover.png").convert("RGB")
secret = Image.open("test_images/secret.png").convert("RGB")

cover = transform(cover).unsqueeze(0).to(DEVICE)
secret = transform(secret).unsqueeze(0).to(DEVICE)

model = SteganographyModel().to(DEVICE)
model.load_state_dict(torch.load("checkpoints/best_model.pth", map_location=DEVICE))
model.eval()

with torch.no_grad():
    stego, recovered = model(cover, secret)

os.makedirs("results", exist_ok=True)

to_pil(cover.squeeze(0).cpu()).save("results/cover.png")
to_pil(secret.squeeze(0).cpu()).save("results/secret.png")
to_pil(stego.squeeze(0).cpu()).save("results/stego.png")
to_pil(recovered.squeeze(0).cpu()).save("results/recovered.png")

print("Inference completed!")
print("Results saved in results/")
