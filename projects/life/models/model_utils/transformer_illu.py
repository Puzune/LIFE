# Copyright (c) 2022-2023, NVIDIA Corporation & Affiliates. All rights reserved.
#
# This work is made available under the Nvidia Source Code License-NC.
# To view a copy of this license, visit
# https://github.com/NVlabs/FB-BEV/blob/main/LICENSE

import torch
import torch.nn as nn
from mmcv.cnn.bricks.transformer import build_transformer_layer_sequence
from mmdet.models.utils.builder import TRANSFORMER
from mmcv.runner.base_module import BaseModule
from mmcv.runner import auto_fp16
from .spatial_cross_attention_depth import DA_MSDeformableAttention


@TRANSFORMER.register_module()
class IDP_transformer(BaseModule):
    """Cross-attend BEV queries to camera context using illumination and depth."""

    def __init__(self, encoder, embed_dims):
        super().__init__()
        self.encoder = build_transformer_layer_sequence(encoder)
        self.embed_dims = embed_dims
        self.fp16_enabled = False

    def init_weights(self):
        for parameter in self.parameters():
            if parameter.dim() > 1:
                nn.init.xavier_uniform_(parameter)
        for module in self.modules():
            if isinstance(module, DA_MSDeformableAttention):
                module.init_weights()

    @auto_fp16(apply_to=('mlvl_feats', 'bev_queries', 'bev_pos'))
    def forward(
        self,
        mlvl_feats,
        bev_queries,
        bev_pos,
        cam_params,
        pred_img_depth,
        illu_map,
    ):
        feature = mlvl_feats[0]
        _, _, _, height, width = feature.shape
        feature = feature.flatten(3).permute(1, 0, 3, 2)
        feature = feature.permute(0, 2, 1, 3)
        bev_pos = bev_pos.flatten(2).permute(2, 0, 1)
        spatial_shapes = torch.as_tensor(
            [(height, width)], dtype=torch.long, device=bev_pos.device
        )
        level_start_index = spatial_shapes.new_zeros((1,))
        return self.encoder(
            bev_queries,
            feature,
            feature,
            bev_pos,
            cam_params,
            pred_img_depth,
            illu_map,
            spatial_shapes,
            level_start_index,
        )
