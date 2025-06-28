import warnings
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import save_image
from models.unet_generator import UNetGenerator
from models.discriminator_dynamic import PatchGANDiscriminator
from utils.corruption_dataset import BlurCorruptionDataset  # Ensure correct dataset

import lpips

warnings.filterwarnings("ignore", category=UserWarning, module="torchvision")

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

os.makedirs("gan_results", exist_ok=True)
os.makedirs("saved_models", exist_ok=True)

G = UNetGenerator().to(device)
D = PatchGANDiscriminator().to(device)

bce_loss = nn.BCEWithLogitsLoss()
l1_loss = nn.L1Loss()
lpips_fn = lpips.LPIPS(net='alex').to(device)
lpips_fn.eval()

g_optimizer = optim.Adam(G.parameters(), lr=2e-4, betas=(0.5, 0.999))
d_optimizer = optim.Adam(D.parameters(), lr=2e-4, betas=(0.5, 0.999))

transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])

dataset = BlurCorruptionDataset("./data/mixed_dataset", transform=transform, blur_radius=2.5)
loader = DataLoader(dataset, batch_size=16, shuffle=True)

num_epochs = 10

for epoch in range(num_epochs):
    print(f"[INFO] Starting epoch {epoch + 1}")
    G.train()
    D.train()
    total_g_loss = 0
    total_d_loss = 0

    sample_noisy, sample_clean = next(iter(loader))
    sample_noisy, sample_clean = sample_noisy.to(device), sample_clean.to(device)

    for i, (noisy, clean) in enumerate(loader):
        noisy, clean = noisy.to(device), clean.to(device)
        print(f"batch {i}")
        real_out = D(clean)
        fake_images = G(noisy).detach()
        fake_out = D(fake_images)

        real_label = torch.ones_like(real_out).to(device)
        fake_label = torch.zeros_like(fake_out).to(device)

        D.zero_grad()
        d_loss_real = bce_loss(real_out, real_label)
        d_loss_fake = bce_loss(fake_out, fake_label)
        d_loss = (d_loss_real + d_loss_fake) / 2
        d_loss.backward()
        d_optimizer.step()

        G.zero_grad()
        generated = G(noisy)
        adv_loss = bce_loss(D(generated), real_label)
        rec_loss = l1_loss(generated, clean)
        perceptual = lpips_fn(generated, clean).mean()

        g_loss = rec_loss + 0.01 * adv_loss + 0.5 * perceptual
        g_loss.backward()
        g_optimizer.step()

        total_g_loss += g_loss.item()
        total_d_loss += d_loss.item()

    print(f"Epoch {epoch+1} | G_loss: {total_g_loss / len(loader):.4f} | D_loss: {total_d_loss / len(loader):.4f}")

    torch.save(G.state_dict(), f"saved_models/gan_generator_epoch{epoch + 1}.pth")
    torch.save(D.state_dict(), f"saved_models/gan_discriminator_epoch{epoch + 1}.pth")

    with torch.no_grad():
        generated = G(sample_noisy)
        save_image(sample_noisy, f"gan_results/noisy_epoch{epoch+1}.png", nrow=4)
        save_image(sample_clean, f"gan_results/clean_epoch{epoch+1}.png", nrow=4)
        save_image(generated, f"gan_results/generated_epoch{epoch+1}.png", nrow=4)