# Copyright (c) OpenMMLab. All rights reserved.
import torch.nn as nn
from mmcv.cnn import ConvModule
from mmcv.runner import BaseModule, auto_fp16
from mmdet3d.models.builder import NECKS


@NECKS.register_module()
class CustomFPN(BaseModule):
    """Project the single DINO semantic feature into the stereo view encoder."""

    def __init__(
        self,
        in_channels,
        out_channels,
        init_cfg=dict(type='Xavier', layer='Conv2d', distribution='uniform'),
    ):
        super().__init__(init_cfg)
        self.fp16_enabled = False
        self.lateral_convs = nn.ModuleList(
            [
                ConvModule(
                    in_channels, out_channels, 1, act_cfg=None, inplace=False
                )
            ]
        )
        self.fpn_convs = nn.ModuleList(
            [
                ConvModule(
                    out_channels,
                    out_channels,
                    3,
                    padding=1,
                    act_cfg=None,
                    inplace=False,
                )
            ]
        )

    @auto_fp16()
    def forward(self, inputs):
        return [self.fpn_convs[0](self.lateral_convs[0](inputs[0]))]
