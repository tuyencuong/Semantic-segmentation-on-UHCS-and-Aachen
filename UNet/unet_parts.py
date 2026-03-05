# """Parts of the U-Net model. Adapted from https://github.com/milesial/Pytorch-UNet/blob/master/unet/unet_parts.py"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv(nn.Module):
    """(convolution => [GroupNorm] => LeakyReLU) * 2 with Residual Connection"""

    def __init__(self, in_channels, out_channels, mid_channels=None, num_groups=8, dropout_p=0.3):  # Increase dropout
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels

        # Main convolutions
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(num_groups=num_groups, num_channels=mid_channels),
            nn.LeakyReLU(inplace=True),
            nn.Dropout(p=dropout_p),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(num_groups=num_groups, num_channels=out_channels),
            nn.LeakyReLU(inplace=True),
            nn.Dropout(p=dropout_p)
        )

        # Residual connection for input-output mismatch
        self.residual = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False) if in_channels != out_channels else nn.Identity()

    def forward(self, x):
        return self.double_conv(x) + self.residual(x)



class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels, num_groups=8, dropout_p=0.3):  # Increase dropout
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            DoubleConv(in_channels, out_channels, num_groups=num_groups, dropout_p=dropout_p)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

class Up(nn.Module):
    """Upscaling then double conv with Residual Connection"""

    def __init__(self, in_channels, out_channels, skip_channels=None, bilinear=True, num_groups=8, dropout_p=0.3):  # Increase dropout
        super().__init__()
        self.bilinear = bilinear
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)

        # Double convolution
        self.conv = DoubleConv(in_channels + (skip_channels or out_channels), out_channels, 
                               mid_channels=(in_channels + (skip_channels or out_channels)) // 2, 
                               num_groups=num_groups, 
                               dropout_p=dropout_p)

        # 1x1 convolution for residual alignment
        self.residual_conv = nn.Conv2d(skip_channels or in_channels, out_channels, kernel_size=1)

    def forward(self, x1, x2):
        # Upsample x1
        x1 = self.up(x1)

        # Handle dimension mismatches for concatenation
        diffY = x2.size(2) - x1.size(2)
        diffX = x2.size(3) - x1.size(3)
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])

        # Concatenate and apply convolution
        x = torch.cat([x2, x1], dim=1)
        conv_out = self.conv(x)

        # Align residual channels
        residual = self.residual_conv(x2)

        return conv_out + residual


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)