"""Legacy UHCS UNet variant kept for backward checkpoint compatibility."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import vgg16


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None, num_groups=8, dropout_p=0.3):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels

        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(min(num_groups, mid_channels), mid_channels),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Dropout(p=dropout_p),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(min(num_groups, out_channels), out_channels),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Dropout(p=dropout_p),
        )

        self.residual = (
            nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.GroupNorm(min(num_groups, out_channels), out_channels),
            )
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x):
        return self.double_conv(x) + self.residual(x)


class AttentionGate(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super().__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1),
            nn.GroupNorm(min(8, F_int), F_int),
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1),
            nn.GroupNorm(min(8, F_int), F_int),
        )
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1),
            nn.Sigmoid(),
        )
        self.relu = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi


class Up(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=True, num_groups=8, dropout_p=0.3):
        super().__init__()
        self.bilinear = bilinear
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)

        self.attention = AttentionGate(
            F_g=in_channels // 2 if bilinear else in_channels,
            F_l=in_channels // 2,
            F_int=in_channels // 4,
        )
        self.conv = DoubleConv(
            in_channels,
            out_channels,
            mid_channels=in_channels // 2,
            num_groups=num_groups,
            dropout_p=dropout_p,
        )
        self.residual_conv = (
            nn.Conv2d(in_channels // 2, out_channels, kernel_size=1)
            if in_channels // 2 != out_channels
            else nn.Identity()
        )

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        x2_attended = self.attention(x1, x2)
        x = torch.cat([x2_attended, x1], dim=1)
        conv_out = self.conv(x)
        residual = self.residual_conv(x2_attended)
        return conv_out + residual


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class SEBlock(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(SEBlock, self).__init__()
        self.fc1 = nn.Linear(in_channels, in_channels // reduction)
        self.fc2 = nn.Linear(in_channels // reduction, in_channels)
        self.act = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.sigmoid = nn.Sigmoid()
        self.norm = nn.LayerNorm(in_channels)

    def forward(self, x):
        b, c, _, _ = x.size()
        y = torch.mean(x, dim=(2, 3))
        y = self.fc1(y)
        y = self.act(y)
        y = self.fc2(y)
        y = self.sigmoid(y).view(b, c, 1, 1)
        x = x * y
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        x = x.permute(0, 3, 1, 2)
        return x


class UNetVgg16UHCSLegacy(nn.Module):
    """Original UHCS model variant; keep module names for strict checkpoint loading."""

    def __init__(self, n_classes, bilinear=True, encoder_pretrained=True):
        super(UNetVgg16UHCSLegacy, self).__init__()

        from torchvision.models.vgg import VGG16_Weights

        try:
            weights = VGG16_Weights.DEFAULT if encoder_pretrained else None
            encoder = vgg16(weights=weights).features
        except Exception:
            encoder = vgg16(weights=None).features

        self.inc = nn.Sequential(encoder[:4], SEBlock(64))
        self.down1 = nn.Sequential(encoder[4:9], SEBlock(128))
        self.down2 = nn.Sequential(encoder[9:16], SEBlock(256))
        self.down3 = nn.Sequential(encoder[16:23], SEBlock(512))
        self.down4 = nn.Sequential(encoder[23:30], SEBlock(512))

        self.bottleneck_attention = AttentionGate(F_g=512, F_l=512, F_int=256)

        self.up1 = Up(1024, 256, bilinear)
        self.up2 = Up(512, 128, bilinear)
        self.up3 = Up(256, 64, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)

        self.encoder = nn.ModuleList([self.inc, self.down1, self.down2, self.down3, self.down4])
        self.decoder = nn.ModuleList([self.bottleneck_attention, self.up1, self.up2, self.up3, self.up4, self.outc])

        self.align_x4 = nn.Conv2d(512, 256, kernel_size=1, bias=False)
        self.align_x3 = nn.Conv2d(256, 128, kernel_size=1, bias=False)
        self.align_x2 = nn.Conv2d(128, 64, kernel_size=1, bias=False)
        self.align_x1 = nn.Conv2d(64, 64, kernel_size=1, bias=False)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x5 = self.bottleneck_attention(x5, x5)

        x4_residual = self.align_x4(x4)
        x3_residual = self.align_x3(x3)
        x2_residual = self.align_x2(x2)
        x1_residual = self.align_x1(x1)

        d4 = self.up1(x5, x4) + x4_residual
        d3 = self.up2(d4, x3) + x3_residual
        d2 = self.up3(d3, x2) + x2_residual
        d1 = self.up4(d2, x1) + x1_residual
        out = self.outc(d1)
        return out

