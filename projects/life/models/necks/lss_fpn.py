# Copyright (c) Phigent Robotics. All rights reserved.
import torch
import torch.nn as nn
from mmcv.cnn import build_norm_layer

from mmdet.models import NECKS


@NECKS.register_module()
class FPN_LSS(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        scale_factor=4,
        input_feature_index=(0, 2),
        norm_cfg=dict(type='BN'),
        extra_upsample=2,
    ):
        super(FPN_LSS, self).__init__()
        self.input_feature_index = input_feature_index

        self.up = nn.Upsample(
            scale_factor=scale_factor, mode='bilinear', align_corners=True
        )

        channels_factor = 2
        self.conv = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels * channels_factor,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            build_norm_layer(norm_cfg, out_channels * channels_factor)[1],
            nn.ReLU(inplace=True),
            nn.Conv2d(
                out_channels * channels_factor,
                out_channels * channels_factor,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            build_norm_layer(norm_cfg, out_channels * channels_factor)[1],
            nn.ReLU(inplace=True),
        )

        self.up2 = nn.Sequential(
            nn.Upsample(
                scale_factor=extra_upsample,
                mode='bilinear',
                align_corners=True,
            ),
            nn.Conv2d(
                out_channels * channels_factor,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            build_norm_layer(norm_cfg, out_channels)[1],
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=1, padding=0),
        )

    def forward(self, feats):
        x2, x1 = (
            feats[self.input_feature_index[0]],
            feats[self.input_feature_index[1]],
        )
        x1 = self.up(x1)
        x1 = torch.cat([x2, x1], dim=1)
        x = self.conv(x1)
        x = self.up2(x)
        return x
