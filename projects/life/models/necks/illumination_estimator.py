# Copyright (c) 2022 Tengyu Ma
import torch
import torch.nn as nn
from mmdet3d.models.builder import NECKS


class EnhanceNetwork(nn.Module):
    """Residual illumination estimator with shared refinement weights."""

    def __init__(self, layers, channels):
        super(EnhanceNetwork, self).__init__()

        kernel_size = 3
        dilation = 1
        padding = int((kernel_size - 1) / 2) * dilation

        self.in_conv = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=channels,
                kernel_size=kernel_size,
                stride=1,
                padding=padding,
            ),
            nn.ReLU(),
        )

        self.conv = nn.Sequential(
            nn.Conv2d(
                in_channels=channels,
                out_channels=channels,
                kernel_size=kernel_size,
                stride=1,
                padding=padding,
            ),
            nn.BatchNorm2d(channels),
            nn.ReLU(),
        )

        self.blocks = nn.ModuleList()
        for _ in range(layers):
            self.blocks.append(self.conv)

        self.out_conv = nn.Sequential(
            nn.Conv2d(
                in_channels=channels,
                out_channels=1,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.Sigmoid(),
        )

    def forward(self, input):
        fea = self.in_conv(input)
        for conv in self.blocks:
            fea = fea + conv(fea)

        fea = self.out_conv(fea)
        illu = fea + input
        illu = torch.clamp(illu, 0.0001, 1)
        return illu


@NECKS.register_module()
class IlluminationEstimator(nn.Module):
    """Frozen SCI illumination predictor used by the image encoder."""

    def __init__(self, enhance_layer=1, enhance_channel=1):
        super().__init__()
        self.enhance = EnhanceNetwork(enhance_layer, enhance_channel)

    def forward(self, gray):
        return self.enhance(gray)
