import torch
import torch.nn as nn

class UNetGenerator(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, features=64):
        super(UNetGenerator, self).__init__()

        # Encoder (Downsampling)
        self.down1 = self.contract_block(in_channels, features, use_bn=False)      # 256 -> 128
        self.down2 = self.contract_block(features, features * 2)                   # 128 -> 64
        self.down3 = self.contract_block(features * 2, features * 4)               # 64 -> 32
        self.down4 = self.contract_block(features * 4, features * 8)               # 32 -> 16
        self.down5 = self.contract_block(features * 8, features * 8)               # 16 -> 8
        self.down6 = self.contract_block(features * 8, features * 8)               # 8 -> 4
        self.down7 = self.contract_block(features * 8, features * 8)               # 4 -> 2
        self.down8 = self.contract_block(features * 8, features * 8)               # 2 -> 1

        # Decoder (Upsampling)
        self.up1 = self.expand_block(features * 8, features * 8)                   # 1 -> 2
        self.up2 = self.expand_block(features * 16, features * 8)                  # 2 -> 4
        self.up3 = self.expand_block(features * 16, features * 8)                  # 4 -> 8
        self.up4 = self.expand_block(features * 16, features * 8)                  # 8 -> 16
        self.up5 = self.expand_block(features * 16, features * 4)                  # 16 -> 32
        self.up6 = self.expand_block(features * 8, features * 2)                   # 32 -> 64
        self.up7 = self.expand_block(features * 4, features)                       # 64 -> 128
        self.up8 = self.expand_block(features * 2, features // 2)                  # 128 -> 256

        self.final_conv = nn.Conv2d(features // 2, out_channels, kernel_size=1)
        self.tanh = nn.Tanh()

    def forward(self, x):
        d1 = self.down1(x)   # 128x128
        d2 = self.down2(d1)  # 64x64
        d3 = self.down3(d2)  # 32x32
        d4 = self.down4(d3)  # 16x16
        d5 = self.down5(d4)  # 8x8
        d6 = self.down6(d5)  # 4x4
        d7 = self.down7(d6)  # 2x2
        d8 = self.down8(d7)  # 1x1

        u1 = self.up1(d8)                      # 2x2
        u2 = self.up2(torch.cat([u1, d7], 1))  # 4x4
        u3 = self.up3(torch.cat([u2, d6], 1))  # 8x8
        u4 = self.up4(torch.cat([u3, d5], 1))  # 16x16
        u5 = self.up5(torch.cat([u4, d4], 1))  # 32x32
        u6 = self.up6(torch.cat([u5, d3], 1))  # 64x64
        u7 = self.up7(torch.cat([u6, d2], 1))  # 128x128
        u8 = self.up8(torch.cat([u7, d1], 1))  # 256x256

        return self.tanh(self.final_conv(u8))

    def contract_block(self, in_channels, out_channels, kernel_size=4, stride=2, padding=1, use_bn=True):
        layers = [nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)]
        if use_bn:
            layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        return nn.Sequential(*layers)

    def expand_block(self, in_channels, out_channels, kernel_size=4, stride=2, padding=1):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )