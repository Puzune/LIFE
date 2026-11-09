# Copyright (c) 2022-2023, NVIDIA Corporation & Affiliates. All rights reserved.
#
# This work is made available under the Nvidia Source Code License-NC.
# To view a copy of this license, visit
# https://github.com/NVlabs/FB-BEV/blob/main/LICENSE

import copy
import torch
from mmcv.cnn import build_norm_layer
from mmcv.cnn.bricks.registry import (
    TRANSFORMER_LAYER,
    TRANSFORMER_LAYER_SEQUENCE,
)
from mmcv.cnn.bricks.transformer import (
    build_attention,
    build_feedforward_network,
    build_transformer_layer,
)
from mmcv.runner import force_fp32, auto_fp16
from mmcv.runner.base_module import BaseModule, ModuleList


@TRANSFORMER_LAYER_SEQUENCE.register_module()
class IDP_transformer_encoder(BaseModule):
    """Project illumination and refine BEV using camera cross-attention."""

    def __init__(
        self, transformerlayers, num_layers, grid_config, data_config
    ):
        super().__init__()
        self.layers = ModuleList(
            [
                build_transformer_layer(copy.deepcopy(transformerlayers))
                for _ in range(num_layers)
            ]
        )
        self.x_bound = grid_config['x']
        self.y_bound = grid_config['y']
        self.z_bound = grid_config['z']
        self.image_size = data_config['input_size']
        self.fp16_enabled = False

    def get_reference_points(self, device, dtype):
        """Return configured ego-frame centers [bev_h, bev_w, Z, 3]."""
        X = (
            torch.arange(*self.x_bound, dtype=torch.float)
            + self.x_bound[-1] / 2
        )
        Y = (
            torch.arange(*self.y_bound, dtype=torch.float)
            + self.y_bound[-1] / 2
        )
        Z = (
            torch.arange(*self.z_bound, dtype=torch.float)
            + self.z_bound[-1] / 2
        )
        Y, X, Z = torch.meshgrid([Y, X, Z])
        return torch.stack([X, Y, Z], dim=-1).to(dtype).to(device)

    @force_fp32(apply_to=('reference_points', 'cam_params'))
    def point_sampling_with_illu_weight(
        self, reference_points, cam_params=None, illu_map=None
    ):
        rots, trans, intrins, post_rots, post_trans, bda = cam_params
        B, N, _ = trans.shape
        eps = 1e-5
        ogfH, ogfW = self.image_size
        reference_points = reference_points[None, None].repeat(
            B, N, 1, 1, 1, 1
        )
        reference_points = (
            torch.inverse(bda)
            .view(B, 1, 1, 1, 1, 3, 3)
            .matmul(reference_points.unsqueeze(-1))
            .squeeze(-1)
        )
        reference_points -= trans.view(B, N, 1, 1, 1, 3)
        combine = rots.matmul(torch.inverse(intrins)).inverse()
        reference_points_cam = (
            combine.view(B, N, 1, 1, 1, 3, 3)
            .matmul(reference_points.unsqueeze(-1))
            .squeeze(-1)
        )
        reference_points_cam = torch.cat(
            [
                reference_points_cam[..., 0:2]
                / torch.maximum(
                    reference_points_cam[..., 2:3],
                    torch.ones_like(reference_points_cam[..., 2:3]) * eps,
                ),
                reference_points_cam[..., 2:3],
            ],
            5,
        )
        reference_points_cam = (
            post_rots.view(B, N, 1, 1, 1, 3, 3)
            .matmul(reference_points_cam.unsqueeze(-1))
            .squeeze(-1)
        )
        reference_points_cam += post_trans.view(B, N, 1, 1, 1, 3)
        reference_points_cam[..., 0] /= ogfW
        reference_points_cam[..., 1] /= ogfH

        mask = reference_points_cam[..., 2:3] > eps
        mask = (
            mask
            & (reference_points_cam[..., 0:1] > eps)
            & (reference_points_cam[..., 0:1] < (1.0 - eps))
            & (reference_points_cam[..., 1:2] > eps)
            & (reference_points_cam[..., 1:2] < (1.0 - eps))
        )

        reference_points_cam_scaled = reference_points_cam.clone()

        reference_points_cam_scaled[..., 0] *= ogfW
        reference_points_cam_scaled[..., 1] *= ogfH
        indices = reference_points_cam_scaled[
            ..., :2
        ].long()
        indices[..., 0] = torch.clamp(indices[..., 0], 0, ogfW - 1)
        indices[..., 1] = torch.clamp(indices[..., 1], 0, ogfH - 1)

        indices_flat = indices.view(B, N, -1, 2)

        illu_weights = illu_map[
            torch.arange(B)[:, None, None],
            torch.arange(N)[None, :, None],
            indices_flat[..., 1],
            indices_flat[..., 0],
        ]

        B, N, H, W, D, _ = reference_points_cam.shape
        reference_points_cam = reference_points_cam.permute(
            1, 0, 2, 3, 4, 5
        ).reshape(N, B, H * W, D, 3)
        # Preserve batch-camera pairing when converting [B, N] to [N, B].
        illu_weights = (
            illu_weights.reshape(B, N, H * W, D)
            .permute(1, 0, 2, 3)
            .unsqueeze(-1)
        )
        mask = (
            mask.permute(1, 0, 2, 3, 4, 5)
            .reshape(N, B, H * W, D, 1)
            .squeeze(-1)
        )
        return (
            reference_points_cam[..., :2],
            mask,
            reference_points_cam[..., 2:3],
            illu_weights,
        )

    def get_illu_bev_weight(self, per_cam_mask_list, illu_weights):
        """Average illumination over visible cameras on the fixed 200x200 BEV grid.

        Inputs use [6, B, 40000, Z] visibility and [6, B, 40000, Z, 1]
        illumination. Output shape is [B, 1, 200, 200].
        """
        _, B, _, num_points_pillar = per_cam_mask_list.shape
        illu_weights_bev_avg_list = []
        for b in range(B):

            mask = per_cam_mask_list[:, b, ...].reshape(
                6, 200, 200, num_points_pillar
            )
            mask_per_cam = mask.sum(-1) > 0

            illu_weights_bev = illu_weights[:, b, ...].reshape(
                6, 200, 200, num_points_pillar
            )
            illu_weights_bev = illu_weights_bev.mean(-1)
            illu_weights_bev = torch.where(
                mask_per_cam,
                illu_weights_bev,
                torch.tensor(
                    0.0,
                    dtype=illu_weights_bev.dtype,
                    device=illu_weights_bev.device,
                ),
            )

            illu_weights_bev_all = illu_weights_bev.sum(axis=0)

            count_all = mask_per_cam.sum(axis=0).float()
            count_all[count_all == 0] = 1.0

            illu_weights_bev_avg = (
                illu_weights_bev_all / count_all
            )

            illu_weights_bev_avg_list.append(illu_weights_bev_avg)

        illu_weights_bev_avg_batch = torch.stack(
            illu_weights_bev_avg_list, dim=0
        ).unsqueeze(
            1
        )

        return illu_weights_bev_avg_batch

    @auto_fp16()
    def forward(
        self,
        bev_query,
        key,
        value,
        bev_pos,
        cam_params,
        pred_img_depth,
        illu_map,
        spatial_shapes,
        level_start_index,
    ):
        """Return refined queries [B, Q, C] and illumination BEV [B, 1, 200, 200]."""
        reference = self.get_reference_points(
            bev_query.device, bev_query.dtype
        )
        _, _, H, W = illu_map.shape
        illumination = illu_map.view(-1, 6, H, W)
        reference_points_cam, mask, bev_depth, illumination_weights = (
            self.point_sampling_with_illu_weight(
                reference, cam_params, illumination
            )
        )
        illumination_bev = self.get_illu_bev_weight(mask, illumination_weights)
        output = bev_query.permute(1, 0, 2)
        bev_pos = bev_pos.permute(1, 0, 2)
        for layer in self.layers:
            output = layer(
                output,
                key,
                value,
                bev_pos,
                reference_points_cam,
                mask,
                bev_depth,
                pred_img_depth,
                spatial_shapes,
                level_start_index,
            )
        return output, illumination_bev


@TRANSFORMER_LAYER.register_module()
class BEVFormerEncoderLayer(BaseModule):
    """Camera cross-attention, layer normalization, and BEV feedforward block."""

    def __init__(self, attn_cfgs, ffn_cfgs, norm_cfg=dict(type='LN')):
        super().__init__()
        if len(attn_cfgs) != 1:
            raise ValueError(
                'LIFE uses one camera cross-attention module per layer.'
            )
        self.attentions = ModuleList([build_attention(attn_cfgs[0])])
        embed_dims = self.attentions[0].embed_dims
        if ffn_cfgs['embed_dims'] != embed_dims:
            raise ValueError(
                'Attention and feedforward feature dimensions must match.'
            )
        self.ffns = ModuleList([build_feedforward_network(ffn_cfgs)])
        self.norms = ModuleList(
            [build_norm_layer(norm_cfg, embed_dims)[1] for _ in range(2)]
        )
        self.fp16_enabled = False

    @force_fp32()
    def forward(
        self,
        query,
        key,
        value,
        bev_pos,
        reference_points_cam,
        per_cam_mask_list,
        bev_query_depth,
        pred_img_depth,
        spatial_shapes,
        level_start_index,
    ):
        query = self.attentions[0](
            query,
            key,
            value,
            query_pos=bev_pos,
            reference_points_cam=reference_points_cam,
            per_cam_mask_list=per_cam_mask_list,
            bev_query_depth=bev_query_depth,
            pred_img_depth=pred_img_depth,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
        )
        query = self.norms[0](query)
        query = self.ffns[0](query)
        return self.norms[1](query)
