# Copyright (c) OpenMMLab. All rights reserved.
import torch
from torch import nn
from mmcv.cnn import ConvModule
from mmcv.runner import BaseModule
import numpy as np
from mmdet3d.models.builder import HEADS, build_loss
from ..losses.occupancy_losses import (
    sem_scal_loss_with_mask,
    geo_scal_loss_with_mask,
)

nusc_class_frequencies = np.array(
    [
        944004,
        1897170,
        152386,
        2391677,
        16957802,
        724139,
        189027,
        2074468,
        413451,
        2384460,
        5916653,
        175883646,
        4275424,
        51393615,
        61411620,
        105975596,
        116424404,
        1892500630,
    ]
)


@HEADS.register_module()
class predictor(BaseModule):
    def __init__(
        self,
        in_dim=256,
        out_dim=256,
        Dz=16,
        weight_ce=1,
        weight_geo=1,
        weight_sem=1,
        num_classes=18,
        loss_occ=None,
    ):
        super(predictor, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.Dz = Dz
        out_channels = out_dim
        self.final_conv = ConvModule(
            self.in_dim,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=True,
            conv_cfg=dict(type='Conv2d'),
        )
        self.predicter = nn.Sequential(
            nn.Linear(self.out_dim, self.out_dim * 2),
            nn.Softplus(),
            nn.Linear(self.out_dim * 2, num_classes * Dz),
        )

        self.num_classes = num_classes

        class_weights = torch.from_numpy(
            1 / np.log(nusc_class_frequencies[:num_classes] + 0.001)
        )
        self.cls_weights = class_weights
        loss_occ['class_weight'] = class_weights

        self.loss_occ = build_loss(loss_occ)

        self.weight_ce = weight_ce
        self.weight_geo = weight_geo
        self.weight_sem = weight_sem

    def forward(self, img_feats):
        """Map BEV features [B, C, Dy, Dx] to logits [B, Dx, Dy, Dz, C_cls]."""
        occ_pred = self.final_conv(img_feats).permute(0, 3, 2, 1)
        bs, Dx, Dy = occ_pred.shape[:3]
        occ_pred = self.predicter(occ_pred)
        occ_pred = occ_pred.view(bs, Dx, Dy, self.Dz, self.num_classes)

        return occ_pred

    def loss(self, occ_pred, voxel_semantics, mask_camera):
        loss = dict()
        voxel_semantics = voxel_semantics.long()
        mask_camera = mask_camera.to(torch.int32)
        voxel_semantics = voxel_semantics.reshape(
            -1
        )
        preds = occ_pred.reshape(
            -1, self.num_classes
        )
        mask_camera = mask_camera.reshape(
            -1
        )

        valid_voxels = voxel_semantics[mask_camera.bool()]
        num_total_samples = 0
        for i in range(self.num_classes):
            num_total_samples += (valid_voxels == i).sum() * self.cls_weights[
                i
            ]

        loss_occ = self.loss_occ(
            cls_score=preds,
            label=voxel_semantics,
            weight=mask_camera,
            avg_factor=num_total_samples,
        )
        loss['loss_occ'] = self.weight_ce * loss_occ
        loss['loss_voxel_sem_scal'] = (
            self.weight_sem
            * sem_scal_loss_with_mask(preds, voxel_semantics, mask_camera)
        )
        loss['loss_voxel_geo_scal'] = (
            self.weight_geo
            * geo_scal_loss_with_mask(
                preds, voxel_semantics, mask_camera, non_empty_idx=17
            )
        )

        return loss

    def get_occ(self, occ_pred):
        occ_res = occ_pred.argmax(-1)
        occ_res = occ_res.cpu().numpy().astype(np.uint8)
        return list(occ_res)
