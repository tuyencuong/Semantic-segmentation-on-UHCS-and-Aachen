import torch
import torch.nn as nn
from torchvision.models import vgg16
# from torchvision.models import vgg19
from .unet_parts import DoubleConv, Up, OutConv
import torch.nn.functional as F


class ECA(nn.Module):
    """Efficient Channel Attention"""
    def __init__(self, in_channels, k_size=3):
        super(ECA, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.avg_pool(x).squeeze(-1).transpose(-1, -2)
        y = self.conv(y).transpose(-1, -2).unsqueeze(-1)
        y = self.sigmoid(y)
        return x * y

class ASPP(nn.Module):
    """Atrous Spatial Pyramid Pooling"""
    def __init__(self, in_channels, out_channels):
        super(ASPP, self).__init__()
        self.atrous_block1 = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.atrous_block6 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=6, dilation=6)
        self.atrous_block12 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=12, dilation=12)
        self.atrous_block18 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=18, dilation=18)
        self.global_avg_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, kernel_size=1)
        )
        self.conv = nn.Conv2d(out_channels * 5, out_channels, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x1 = self.atrous_block1(x)
        x2 = self.atrous_block6(x)
        x3 = self.atrous_block12(x)
        x4 = self.atrous_block18(x)
        x5 = self.global_avg_pool(x)
        x5 = F.interpolate(x5, size=x.shape[2:], mode='bilinear', align_corners=True)
        x = torch.cat((x1, x2, x3, x4, x5), dim=1)
        x = self.conv(x)
        return self.relu(x)

class UNetVgg16(nn.Module):
    """UNet with VGG16 backbone, ASPP, and ECA attention"""

    def __init__(self, n_classes, encoder_pretrained=True):
        super(UNetVgg16, self).__init__()
        # Keep checkpoint-compatible layer names while allowing offline usage.
        from torchvision.models.vgg import VGG16_Weights
        try:
            weights = VGG16_Weights.DEFAULT if encoder_pretrained else None
            encoder = vgg16(weights=weights).features
        except Exception:
            # Fallback when pretrained weights cannot be fetched.
            encoder = vgg16(weights=None).features

        # Encoder - Remove ECA from encoder to reduce computation
        self.inc = encoder[:4]  # Remove ECA(64)
        self.down1 = encoder[4:9]  # Remove ECA(128)
        self.down2 = encoder[9:16]  # Remove ECA(256)
        self.down3 = encoder[16:23]  # Remove ECA(512)
        self.down4 = encoder[23:30]  # Remove ECA(512)

        # Combine encoder layers into a single attribute
        self.encoder = nn.ModuleList([self.inc, self.down1, self.down2, self.down3, self.down4])

        # ASPP in bottleneck - Reduce complexity
        self.aspp = ASPP(512, 256)
     
        # Add ECA only where it's most effective (after ASPP)
        self.eca_bottleneck = ECA(256)

        # Decoder
        self.up1 = Up(256, 256, skip_channels=512)
        self.up2 = Up(256, 128, skip_channels=256)
        self.up3 = Up(128, 64, skip_channels=128)
        self.up4 = Up(64, 64, skip_channels=64)
        self.outc = OutConv(64, n_classes)
        # Group decoder layers into a single attribute
        self.decoder = nn.ModuleList([self.up1, self.up2, self.up3, self.up4, self.outc])

        # Initialize weights for decoder (non-pretrained) layers
        self._init_weights_decoder()

    def _init_weights_decoder(self):
        # Only initialize decoder weights since encoder uses pretrained weights
        for m in self.decoder.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)

    def forward(self, x):
        # Encoder
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        # Bottleneck with ASPP and single ECA
        x5 = self.aspp(x5)
        x5 = self.eca_bottleneck(x5)

        # Decoder
        d4 = self.up1(x5, x4)
        d3 = self.up2(d4, x3)
        d2 = self.up3(d3, x2)
        d1 = self.up4(d2, x1)

        # Output
        return self.outc(d1)
