import torch
import torch.nn as nn

class PatchGANDiscriminator(nn.Module):
    def __init__(self, in_channels=3, features=64):
        super(PatchGANDiscriminator, self).__init__()

        self.net = nn.Sequential(
            nn.Conv2d(in_channels, features, 4, 2, 1),       # 256 -> 128
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(features, features * 2, 4, 2, 1),       # 128 -> 64
            nn.BatchNorm2d(features * 2),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(features * 2, features * 4, 4, 2, 1),   # 64 -> 32
            nn.BatchNorm2d(features * 4),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(features * 4, features * 8, 4, 2, 1),   # 32 -> 16
            nn.BatchNorm2d(features * 8),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(features * 8, 1, kernel_size=4, stride=1, padding=1)  # -> [B, 1, 15, 15]
        )

    def forward(self, x):
        return self.net(x)