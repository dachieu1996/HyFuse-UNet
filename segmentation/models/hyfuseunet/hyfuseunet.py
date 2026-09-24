import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels),
        )

    def forward(self, x):
        return self.block(x)


class SpectralConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class SpectralDown(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.MaxPool2d(2),
            SpectralConv(in_channels, out_channels),
        )

    def forward(self, x):
        return self.block(x)


class Up(nn.Module):
    def __init__(
        self,
        in_channels,
        rgb_skip_channels,
        hsi_spatial_skip_channels,
        hsi_spectral_skip_channels,
        out_channels,
    ):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.rgb_proj = nn.Sequential(
            nn.Conv2d(rgb_skip_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        self.hsi_spatial_proj = nn.Sequential(
            nn.Conv2d(hsi_spatial_skip_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        self.hsi_spectral_proj = nn.Sequential(
            nn.Conv2d(hsi_spectral_skip_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        self.conv = DoubleConv(out_channels * 4, out_channels)

    def forward(self, x, rgb_skip, hsi_spatial_skip, hsi_spectral_skip):
        x = self.up(x)

        if x.shape[-2:] != rgb_skip.shape[-2:]:
            x = F.interpolate(
                x,
                size=rgb_skip.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )

        if hsi_spatial_skip.shape[-2:] != rgb_skip.shape[-2:]:
            hsi_spatial_skip = F.interpolate(
                hsi_spatial_skip,
                size=rgb_skip.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )

        if hsi_spectral_skip.shape[-2:] != rgb_skip.shape[-2:]:
            hsi_spectral_skip = F.interpolate(
                hsi_spectral_skip,
                size=rgb_skip.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )

        rgb_skip = self.rgb_proj(rgb_skip)
        hsi_spatial_skip = self.hsi_spatial_proj(hsi_spatial_skip)
        hsi_spectral_skip = self.hsi_spectral_proj(hsi_spectral_skip)
        x = torch.cat([x, rgb_skip, hsi_spatial_skip, hsi_spectral_skip], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class HyFuseUNet(nn.Module):
    def __init__(self, rgb_channels, hsi_channels, num_classes, width=16):
        super().__init__()
        widths = [width, width * 2, width * 4, width * 8]

        self.rgb_inc = DoubleConv(rgb_channels, widths[0])
        self.rgb_down1 = Down(widths[0], widths[1])
        self.rgb_down2 = Down(widths[1], widths[2])
        self.rgb_down3 = Down(widths[2], widths[3])

        self.hsi_spatial_inc = DoubleConv(hsi_channels, widths[0])
        self.hsi_spatial_down1 = Down(widths[0], widths[1])
        self.hsi_spatial_down2 = Down(widths[1], widths[2])
        self.hsi_spatial_down3 = Down(widths[2], widths[3])

        self.hsi_spectral_inc = SpectralConv(hsi_channels, widths[0])
        self.hsi_spectral_down1 = SpectralDown(widths[0], widths[1])
        self.hsi_spectral_down2 = SpectralDown(widths[1], widths[2])
        self.hsi_spectral_down3 = SpectralDown(widths[2], widths[3])

        self.bridge = DoubleConv(widths[3] * 3, widths[3])
        self.up1 = Up(widths[3], widths[2], widths[2], widths[2], widths[2])
        self.up2 = Up(widths[2], widths[1], widths[1], widths[1], widths[1])
        self.up3 = Up(widths[1], widths[0], widths[0], widths[0], widths[0])
        self.outc = OutConv(widths[0], num_classes)

    def forward(self, inputs):
        rgb, hsi = inputs

        rgb1 = self.rgb_inc(rgb)
        rgb2 = self.rgb_down1(rgb1)
        rgb3 = self.rgb_down2(rgb2)
        rgb4 = self.rgb_down3(rgb3)

        hsi_spatial1 = self.hsi_spatial_inc(hsi)
        hsi_spatial2 = self.hsi_spatial_down1(hsi_spatial1)
        hsi_spatial3 = self.hsi_spatial_down2(hsi_spatial2)
        hsi_spatial4 = self.hsi_spatial_down3(hsi_spatial3)

        hsi_spectral1 = self.hsi_spectral_inc(hsi)
        hsi_spectral2 = self.hsi_spectral_down1(hsi_spectral1)
        hsi_spectral3 = self.hsi_spectral_down2(hsi_spectral2)
        hsi_spectral4 = self.hsi_spectral_down3(hsi_spectral3)

        x = self.bridge(torch.cat([rgb4, hsi_spatial4, hsi_spectral4], dim=1))
        x = self.up1(x, rgb3, hsi_spatial3, hsi_spectral3)
        x = self.up2(x, rgb2, hsi_spatial2, hsi_spectral2)
        x = self.up3(x, rgb1, hsi_spatial1, hsi_spectral1)
        return self.outc(x)
