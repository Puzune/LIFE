# Copyright (c) Phigent Robotics. All rights reserved.
import torch
from mmcv.runner import force_fp32
from mmdet3d.models import CenterPoint, builder


class StereoBEVBase(CenterPoint):
    """BEVDet geometry, stereo inputs, and BEV encoding used by LIFE."""

    def __init__(
        self,
        img_backbone,
        img_neck,
        img_view_transformer,
        img_bev_encoder_backbone,
        img_bev_encoder_neck,
        pre_process,
        num_adj=1,
        train_cfg=None,
        test_cfg=None,
        init_cfg=None,
    ):
        super().__init__(
            img_backbone=img_backbone,
            img_neck=img_neck,
            pts_bbox_head=None,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            init_cfg=init_cfg,
        )
        self.img_view_transformer = builder.build_neck(img_view_transformer)
        self.img_bev_encoder_backbone = builder.build_backbone(
            img_bev_encoder_backbone
        )
        self.img_bev_encoder_neck = builder.build_neck(img_bev_encoder_neck)
        self.pre_process_net = builder.build_backbone(pre_process)
        self.temporal_frame = num_adj + 1
        self.extra_ref_frames = 1
        self.num_frame = self.temporal_frame + self.extra_ref_frames

    @force_fp32()
    def bev_encoder(self, x):
        x = self.img_bev_encoder_backbone(x)
        x = self.img_bev_encoder_neck(x)
        return x

    def prepare_inputs(self, img_inputs):
        """Unpack camera-major RGB and frame-major geometry into per-frame inputs.

        RGB includes an extra stereo reference frame. Returned geometry covers
        only the key and history BEV frames.
        """
        B, N, C, H, W = img_inputs[0].shape
        N = N // self.num_frame
        imgs = img_inputs[0].view(
            B, N, self.num_frame, C, H, W
        )
        imgs = torch.split(imgs, 1, 2)
        imgs = [
            t.squeeze(2) for t in imgs
        ]
        sensor2egos, ego2globals, intrins, post_rots, post_trans, bda = (
            img_inputs[1:7]
        )

        sensor2egos = sensor2egos.view(B, self.num_frame, N, 4, 4)
        ego2globals = ego2globals.view(B, self.num_frame, N, 4, 4)

        keyego2global = ego2globals[:, 0, 0, ...].unsqueeze(1).unsqueeze(1)
        global2keyego = torch.inverse(keyego2global.double())
        # Map each camera into the key-frame ego coordinate system.
        sensor2keyegos = (
            global2keyego
            @ ego2globals[:, : self.temporal_frame].double()
            @ sensor2egos[:, : self.temporal_frame].double()
        )
        sensor2keyegos = sensor2keyegos.float()

        sensor2egos_cv, ego2globals_cv = sensor2egos, ego2globals
        sensor2egos_curr = sensor2egos_cv[
            :, : self.temporal_frame, ...
        ].double()
        ego2globals_curr = ego2globals_cv[
            :, : self.temporal_frame, ...
        ].double()
        sensor2egos_adj = sensor2egos_cv[
            :, 1 : self.temporal_frame + 1, ...
        ].double()
        ego2globals_adj = ego2globals_cv[
            :, 1 : self.temporal_frame + 1, ...
        ].double()

        # Map each camera into the preceding stereo reference camera.
        curr2adjsensor = (
            torch.inverse(ego2globals_adj @ sensor2egos_adj)
            @ ego2globals_curr
            @ sensor2egos_curr
        )
        curr2adjsensor = (
            curr2adjsensor.float()
        )
        curr2adjsensor = torch.split(curr2adjsensor, 1, 1)
        curr2adjsensor = [p.squeeze(1) for p in curr2adjsensor]
        assert len(curr2adjsensor) == self.temporal_frame

        extra = [
            sensor2keyegos,
            intrins.view(B, self.num_frame, N, 3, 3)[
                :, : self.temporal_frame
            ],
            post_rots.view(B, self.num_frame, N, 3, 3)[
                :, : self.temporal_frame
            ],
            post_trans.view(B, self.num_frame, N, 3)[
                :, : self.temporal_frame
            ],
        ]
        extra = [torch.split(t, 1, 1) for t in extra]
        extra = [[p.squeeze(1) for p in t] for t in extra]
        sensor2keyegos, intrins, post_rots, post_trans = extra
        return (
            imgs,
            sensor2keyegos,
            intrins,
            post_rots,
            post_trans,
            bda,
            curr2adjsensor,
        )

    def extract_stereo_ref_feat(self, images):
        """Extract stride-4 reference features without the semantic backbone."""
        batch, views, channels, height, width = images.shape
        images = images.view(batch * views, channels, height, width)
        return self.img_backbone.forward_stereo(images)

    def forward_test(self, img_inputs, img_gray, img_metas, rescale=True):
        """Unpack the MMDetection3D test wrapper for occupancy prediction."""
        for value, name in [
            (img_inputs, 'img_inputs'),
            (img_metas, 'img_metas'),
        ]:
            if not isinstance(value, list):
                raise TypeError(f'{name} must be a list, got {type(value)}.')
        if len(img_inputs) != len(img_metas):
            raise ValueError(
                'Image inputs and metadata must have matching augmentations.'
            )
        if len(img_inputs) != 1 or isinstance(img_inputs[0][0], list):
            raise NotImplementedError(
                'LIFE uses single-scale occupancy evaluation.'
            )
        return self.simple_test(img_inputs[0], img_gray)
