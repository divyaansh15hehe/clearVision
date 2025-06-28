import os
from PIL import Image, ImageFilter
from torch.utils.data import Dataset
from torchvision import transforms

class BlurCorruptionDataset(Dataset):
    def __init__(self, image_dir, transform=None, blur_radius=1.65):
        self.image_paths = sorted([
            os.path.join(image_dir, fname)
            for fname in os.listdir(image_dir)
            if fname.lower().endswith(('.jpg', '.png'))
        ])
        self.transform = transform or transforms.ToTensor()
        self.blur_radius = blur_radius  # Adjustable, 2.5 is moderate blur

    def __getitem__(self, idx):
        clean = Image.open(self.image_paths[idx]).convert("RGB")

        corrupted = clean.filter(ImageFilter.GaussianBlur(radius=self.blur_radius))

        if self.transform:
            clean = self.transform(clean)
            corrupted = self.transform(corrupted)

        return corrupted, clean

    def __len__(self):
        return len(self.image_paths)