# evaluate.py
import os
import torch
import lpips
import numpy as np
from torchvision.utils import save_image
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
import torchvision.transforms as T

# Paths to image folders
CLEAN_DIR = "gan_split/clean256"
RECON_DIR = "gan_split/gen256"

# Load LPIPS loss model
lpips_fn = lpips.LPIPS(net='alex')  # options: 'alex', 'vgg', 'squeeze'
lpips_fn.eval()

EPOCH = "10"
CLEAN_PREFIX = f"clean{EPOCH}"
RECON_PREFIX = f"gen{EPOCH}"

# Preprocessing (to tensor)
transform = T.Compose([
    T.Resize((32, 32)),
    T.ToTensor()
])

def load_image(path):
    img = Image.open(path).convert("RGB")
    return transform(img)

# Collect matching filenames
file_names = sorted(os.listdir(CLEAN_DIR))

psnr_scores = []
ssim_scores = []
lpips_scores = []

for idx in range(16):  # assuming 8x8 grid → 64 images
    clean_path = os.path.join(CLEAN_DIR, f"{CLEAN_PREFIX}_{idx:02}.png")
    recon_path = os.path.join(RECON_DIR, f"{RECON_PREFIX}_{idx:02}.png")

    if not (os.path.exists(clean_path) and os.path.exists(recon_path)):
        continue

    clean = load_image(clean_path).unsqueeze(0)
    recon = load_image(recon_path).unsqueeze(0)

    # PSNR
    psnr_val = peak_signal_noise_ratio(clean.squeeze().numpy(), recon.squeeze().numpy(), data_range=1.0)
    psnr_scores.append(psnr_val)

    # SSIM
    ssim_val = structural_similarity(
        clean.squeeze().permute(1, 2, 0).numpy(),
        recon.squeeze().permute(1, 2, 0).numpy(),
        data_range=1.0,
        channel_axis=-1
    )
    ssim_scores.append(ssim_val)

    # LPIPS
    with torch.no_grad():
        lpips_val = lpips_fn(clean, recon).item()
    lpips_scores.append(lpips_val)

# Final Results
print(f"\n--- Evaluation for Epoch {EPOCH} ---")
print(f"PSNR :  {np.mean(psnr_scores):.2f} dB")
print(f"SSIM :  {np.mean(ssim_scores):.4f}")
print(f"LPIPS:  {np.mean(lpips_scores):.4f}")