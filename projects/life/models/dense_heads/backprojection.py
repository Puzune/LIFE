# Copyright (c) 2022-2023, NVIDIA Corporation & Affiliates. All rights reserved.
#
# This work is made available under the Nvidia Source Code License-NC.
# To view a copy of this license, visit
# https://github.com/NVlabs/FB-BEV/blob/main/LICENSE


# Modified by Zhiqi Li.

import torch.nn as nn
from mmcv.runner.base_module import BaseModule
from mmdet.models import HEADS
from mmcv.cnn.bricks.transformer import build_positional_encoding
from mmcv.runner import auto_fp16
from mmdet.models.utils import build_transformer


@HEADS.register_module()
class IDP(BaseModule):
    """Illumination-aware depth projection from camera context to BEV."""

    def __init__(self, transformer, positional_encoding, bev_h=200, bev_w=200):
        super().__init__()
        self.bev_h = bev_h
        self.bev_w = bev_w
        self.fp16_enabled = False
        self.positional_encoding = build_positional_encoding(
            positional_encoding
        )
        self.transformer = build_transformer(transformer)
        self.bev_embedding = nn.Embedding(
            bev_h * bev_w, self.transformer.embed_dims
        )

    def init_weights(self):
        self.transformer.init_weights()

    @auto_fp16(apply_to=('mlvl_feats',))
    def forward(
        self, mlvl_feats, img_inputs, lss_bev, pred_img_depth, illu_map
    ):
        batch = mlvl_feats[0].size(0)
        dtype = mlvl_feats[0].dtype
        bev_queries = self.bev_embedding.weight.to(dtype)
        bev_queries = bev_queries.unsqueeze(1).repeat(1, batch, 1)
        bev_queries = bev_queries + lss_bev.flatten(2).permute(2, 0, 1)
        bev_pos = self.positional_encoding(
            batch, self.bev_h, self.bev_w, bev_queries.device
        ).to(dtype)
        sensor2egos, intrins, post_rots, post_trans, bda = img_inputs[1:6]
        cam_params = [
            sensor2egos[:, :, :3, :3],
            sensor2egos[:, :, :3, 3],
            intrins,
            post_rots,
            post_trans,
            bda,
        ]
        bev, illumination = self.transformer(
            mlvl_feats,
            bev_queries,
            bev_pos,
            cam_params,
            pred_img_depth,
            illu_map,
        )
        bev = (
            bev.permute(0, 2, 1)
            .view(batch, -1, self.bev_h, self.bev_w)
            .contiguous()
        )
        return bev, illumination
