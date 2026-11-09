# Copyright (c) 2022-2023, NVIDIA Corporation & Affiliates. All rights reserved.
#
# This work is made available under the Nvidia Source Code License-NC.
# To view a copy of this license, visit
# https://github.com/NVlabs/FB-BEV/blob/main/LICENSE

import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import xavier_init, constant_init
from mmcv.cnn.bricks.registry import ATTENTION
from mmcv.cnn.bricks.transformer import build_attention
import math
from mmcv.runner import force_fp32

from mmcv.runner.base_module import BaseModule

from .multi_scale_deformable_attn_function import (
    MultiScaleDeformableAttnFunction_fp32,
)


@ATTENTION.register_module()
class DA_SpatialCrossAttention(BaseModule):
    """Camera cross-attention with depth-weighted BEV height anchors."""

    def __init__(
        self,
        deformable_attention,
        dbound,
        embed_dims=256,
        num_cams=6,
        dropout=0.1,
    ):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.fp16_enabled = False
        self.deformable_attention = build_attention(deformable_attention)
        self.embed_dims = embed_dims
        self.num_cams = num_cams
        self.dbound = dbound
        self.output_proj = nn.Linear(embed_dims, embed_dims)
        self.init_weight()

    def init_weight(self):
        xavier_init(self.output_proj, distribution='uniform', bias=0.0)

    @force_fp32(
        apply_to=('query', 'key', 'value', 'query_pos', 'reference_points_cam')
    )
    def forward(
        self,
        query,
        key,
        value,
        query_pos,
        reference_points_cam,
        spatial_shapes,
        level_start_index,
        bev_query_depth,
        pred_img_depth,
        per_cam_mask_list,
    ):
        """Return BEV queries [B, Q, C] from camera features [N, S, B, C]."""
        B, N, DC, H, W = pred_img_depth.shape

        bev_query_depth = bev_query_depth.permute(1, 0, 2, 3, 4)
        pred_img_depth = pred_img_depth.view(B * N, DC, H, W)
        pred_img_depth = pred_img_depth.flatten(2).permute(0, 2, 1)

        inp_residual = query
        slots = torch.zeros_like(query)
        query = query + query_pos

        bs = query.size(0)

        D = reference_points_cam.size(3)
        indexes = [[] for _ in range(bs)]

        max_len = 0
        for j in range(bs):
            for per_cam_mask in per_cam_mask_list:
                index_query_per_img = (
                    per_cam_mask[j].sum(-1).nonzero().squeeze(-1)
                )
                indexes[j].append(index_query_per_img)
                max_len = max(max_len, len(index_query_per_img))

        # Batch only the BEV queries visible to each camera to save memory.
        queries_rebatch = query.new_zeros(
            [bs, self.num_cams, max_len, self.embed_dims]
        )
        reference_points_rebatch = reference_points_cam.new_zeros(
            [bs, self.num_cams, max_len, D, 2]
        )
        bev_query_depth_rebatch = reference_points_cam.new_zeros(
            [bs, self.num_cams, max_len, D, 1]
        )

        for j in range(bs):
            for i, reference_points_per_img in enumerate(reference_points_cam):
                index_query_per_img = indexes[j][i]
                queries_rebatch[j, i, : len(index_query_per_img)] = query[
                    j, index_query_per_img
                ]
                bev_query_depth_rebatch[j, i, : len(index_query_per_img)] = (
                    bev_query_depth[j, i, index_query_per_img]
                )

                reference_points_rebatch[j, i, : len(index_query_per_img)] = (
                    reference_points_per_img[j, index_query_per_img]
                )

        num_cams, l, bs, _ = key.shape
        value = value.permute(2, 0, 1, 3).reshape(
            bs * self.num_cams, l, self.embed_dims
        )

        bev_query_depth_rebatch = (
            bev_query_depth_rebatch - self.dbound[0]
        ) / self.dbound[2]
        bev_query_depth_rebatch = torch.clip(
            torch.floor(bev_query_depth_rebatch), 0, DC - 1
        ).to(torch.long)

        bev_query_depth_rebatch = F.one_hot(
            bev_query_depth_rebatch.squeeze(-1), num_classes=DC
        )

        queries = self.deformable_attention(
            query=queries_rebatch.view(
                bs * self.num_cams, max_len, self.embed_dims
            ),
            value=value,
            reference_points=reference_points_rebatch.view(
                bs * self.num_cams, max_len, D, 2
            ),
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
            bev_query_depth=bev_query_depth_rebatch.view(
                bs * self.num_cams, max_len, D, DC
            ),
            pred_img_depth=pred_img_depth,
        ).view(bs, self.num_cams, max_len, self.embed_dims)

        for j in range(bs):
            for i in range(num_cams):
                index_query_per_img = indexes[j][i]
                slots[j, index_query_per_img] += queries[
                    j, i, : len(index_query_per_img)
                ]

        count = per_cam_mask_list.sum(-1) > 0
        count = count.permute(1, 2, 0).sum(-1)
        count = torch.clamp(count, min=1.0)
        slots = slots / count[..., None]

        slots = self.output_proj(slots)
        return self.dropout(slots) + inp_residual


@ATTENTION.register_module()
class DA_MSDeformableAttention(BaseModule):
    """Apply depth-weighted deformable attention around projected height anchors.

    Split num_points samples evenly across num_Z_anchors height anchors
    for each head and feature level.
    """

    def __init__(
        self,
        embed_dims=256,
        num_heads=8,
        num_levels=4,
        num_points=8,
        num_Z_anchors=4,
        im2col_step=64,
        init_cfg=None,
    ):
        super().__init__(init_cfg)
        if embed_dims % num_heads != 0:
            raise ValueError(
                f'embed_dims must be divisible by num_heads, '
                f'but got {embed_dims} and {num_heads}'
            )
        dim_per_head = embed_dims // num_heads
        self.fp16_enabled = False
        self.num_Z_anchors = num_Z_anchors

        def _is_power_of_2(n):
            if (not isinstance(n, int)) or (n < 0):
                raise ValueError(
                    'invalid input for _is_power_of_2: {} (type: {})'.format(
                        n, type(n)
                    )
                )
            return (n & (n - 1) == 0) and n != 0

        if not _is_power_of_2(dim_per_head):
            warnings.warn(
                "You'd better set embed_dims in "
                'MultiScaleDeformAttention to make '
                'the dimension of each attention head a power of 2 '
                'which is more efficient in our CUDA implementation.'
            )

        self.im2col_step = im2col_step
        self.embed_dims = embed_dims
        self.num_levels = num_levels
        self.num_heads = num_heads
        self.num_points = num_points
        self.sampling_offsets = nn.Linear(
            embed_dims, num_heads * num_levels * num_points * 2
        )
        self.attention_weights = nn.Linear(
            embed_dims, num_heads * num_levels * num_points
        )
        self.value_proj = nn.Linear(embed_dims, embed_dims)

        self.init_weights()

    def init_weights(self):
        constant_init(self.sampling_offsets, 0.0)
        thetas = torch.arange(self.num_heads, dtype=torch.float32) * (
            2.0 * math.pi / self.num_heads
        )

        self.each_anchor_points = self.num_points // self.num_Z_anchors

        grid_init = torch.stack([thetas.cos(), thetas.sin()], -1)
        grid_init = (
            (grid_init / grid_init.abs().max(-1, keepdim=True)[0])
            .view(self.num_heads, 1, 1, 1, 2)
            .repeat(
                1,
                self.num_levels,
                self.each_anchor_points,
                self.num_Z_anchors,
                1,
            )
        )
        for i in range(self.each_anchor_points):
            for j in range(self.num_Z_anchors):
                grid_init[:, :, i, j, :] *= i + 1

        self.sampling_offsets.bias.data = grid_init.view(-1)
        constant_init(self.attention_weights, val=0.0, bias=0.0)
        xavier_init(self.value_proj, distribution='uniform', bias=0.0)
        self._is_init = True

    @force_fp32()
    def forward(
        self,
        query,
        value,
        reference_points,
        spatial_shapes,
        level_start_index,
        bev_query_depth,
        pred_img_depth,
        query_pos=None,
    ):
        """Sample query features [B, Q, C] around normalized anchors [B, Q, Z, 2]."""
        if query_pos is not None:
            query = query + query_pos
        bs, num_query, _ = query.shape
        bs, num_value, _ = value.shape
        assert (spatial_shapes[:, 0] * spatial_shapes[:, 1]).sum() == num_value

        value = self.value_proj(value)
        value = value.view(bs, num_value, self.num_heads, -1)

        sampling_offsets = self.sampling_offsets(query).view(
            bs, num_query, self.num_heads, self.num_levels, self.num_points, 2
        )
        attention_weights = self.attention_weights(query).view(
            bs, num_query, self.num_heads, self.num_levels * self.num_points
        )
        attention_weights = attention_weights.softmax(-1)

        attention_weights = attention_weights.view(
            bs, num_query, self.num_heads, self.num_levels, self.num_points
        )

        if reference_points.shape[-1] != 2:
            raise ValueError(
                'Projected height anchors require two image coordinates.'
            )
        offset_normalizer = torch.stack(
            [spatial_shapes[..., 1], spatial_shapes[..., 0]], -1
        )

        bs, num_query, num_Z_anchors, xy = reference_points.shape
        reference_points = reference_points[:, :, None, None, None, :, :]

        sampling_offsets = (
            sampling_offsets / offset_normalizer[None, None, None, :, None, :]
        )
        bs, num_query, num_heads, num_levels, num_all_points, xy = (
            sampling_offsets.shape
        )
        sampling_offsets = sampling_offsets.view(
            bs,
            num_query,
            num_heads,
            num_levels,
            num_all_points // num_Z_anchors,
            num_Z_anchors,
            xy,
        )
        sampling_locations = reference_points + sampling_offsets
        bs, num_query, num_heads, num_levels, num_points, num_Z_anchors, xy = (
            sampling_locations.shape
        )
        assert num_all_points == num_points * num_Z_anchors

        sampling_locations = sampling_locations.view(
            bs, num_query, num_heads, num_levels, num_all_points, xy
        )

        if not value.is_cuda:
            raise RuntimeError(
                'LIFE depth-aware deformable attention requires CUDA tensors.'
            )
        MultiScaleDeformableAttnFunction = (
            MultiScaleDeformableAttnFunction_fp32
        )
        depth_reference_points = reference_points.reshape(
            bs, num_query * num_Z_anchors, 1, 1, 1, 2
        ).contiguous()
        depth_attention_weights = torch.ones_like(
            depth_reference_points[..., 0]
        ).contiguous()

        depth_weights = MultiScaleDeformableAttnFunction.apply(
            pred_img_depth.unsqueeze(2).contiguous(),
            spatial_shapes[0:1],
            level_start_index[0:1],
            depth_reference_points,
            depth_attention_weights,
            self.im2col_step,
        ).reshape(bs, num_query, num_Z_anchors, -1)

        depth_weights = (depth_weights * bev_query_depth).sum(-1)
        depth_weights = (
            depth_weights.unsqueeze(2)
            .repeat(1, 1, num_points, 1)
            .reshape(bs, num_query, num_all_points)
        )

        attention_weights = (
            attention_weights * depth_weights[:, :, None, None, :]
        )
        output = MultiScaleDeformableAttnFunction.apply(
            value,
            spatial_shapes,
            level_start_index,
            sampling_locations,
            attention_weights,
            self.im2col_step,
        )
        return output
