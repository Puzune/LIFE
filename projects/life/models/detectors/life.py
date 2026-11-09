# Copyright (c) Phigent Robotics. All rights reserved.
import torch
from mmdet3d.models import DETECTORS, build_neck
from mmdet3d.models.builder import build_head, build_loss

from .stereo_bev import StereoBEVBase


@DETECTORS.register_module()
class LIFE(StereoBEVBase):
    """Illumination-guided stereo occupancy prediction with BEV regularization."""

    def __init__(
        self,
        light_enhance,
        gamma_correction,
        illumination_refinement,
        occ_head,
        sigreg_loss,
        init_weights_path='ckpts/init.pth',
        **kwargs
    ):
        super().__init__(**kwargs)
        self.init_weights_path = init_weights_path
        self._pretrained_initialized = False
        self.sigreg_loss = build_loss(sigreg_loss)
        self.light_enhance = build_neck(light_enhance)
        for parameter in self.light_enhance.parameters():
            parameter.requires_grad = False
        self.light_enhance.eval()
        self.correction_module = build_neck(gamma_correction)
        self.illumination_refinement = build_neck(illumination_refinement)
        self.occ_head = build_head(occ_head)
        self.pts_bbox_head = None

    def init_weights(self):
        """Initialize trainable modules and load the two frozen branches."""
        if self._pretrained_initialized:
            return
        checkpoint = torch.load(
            self.init_weights_path, map_location='cpu', weights_only=True
        )
        if not self._is_init:
            super().init_weights()
        for prefix, module in (
            ('img_backbone.stereo_branch.', self.img_backbone.stereo_branch),
            ('light_enhance.', self.light_enhance),
        ):
            state_dict = {
                key[len(prefix):]: value
                for key, value in checkpoint.items()
                if key.startswith(prefix)
            }
            module.load_state_dict(state_dict, strict=True)
        self._pretrained_initialized = True

    def train(self, mode=True):
        super().train(mode)
        self.light_enhance.eval()
        return self

    @staticmethod
    def denormalize(tensor):
        mean = tensor.new_tensor([123.675, 116.28, 103.53]).view(1, -1, 1, 1)
        std = tensor.new_tensor([58.395, 57.12, 57.375]).view(1, -1, 1, 1)
        return tensor * std + mean

    @staticmethod
    def normalize(tensor):
        mean = tensor.new_tensor([123.675, 116.28, 103.53]).view(1, -1, 1, 1)
        std = tensor.new_tensor([58.395, 57.12, 57.375]).view(1, -1, 1, 1)
        return (tensor - mean) / std

    def image_encoder(self, img, gray):
        """Encode [B, N, 3, H, W] images using their illumination maps."""
        batch, views, channels, height, width = img.shape
        images = img.view(batch * views, channels, height, width)
        originals = self.denormalize(images)
        gray = gray.view(batch * views, 1, height, width)
        with torch.no_grad():
            illumination = self.light_enhance(gray)
        enhanced = torch.clamp(originals / illumination, 0, 255)
        enhanced = self.correction_module(originals, enhanced, illumination)
        stereo_features, semantic = self.img_backbone(self.normalize(enhanced))
        features = self.img_neck([semantic])[0]
        _, output_channels, output_height, output_width = features.shape
        features = features.view(
            batch, views, output_channels, output_height, output_width
        )
        return features, stereo_features, illumination

    def prepare_bev_feat(
        self,
        img,
        sensor2keyego,
        intrin,
        post_rot,
        post_tran,
        bda,
        mlp_input,
        feat_prev_iv,
        k2s_sensor,
        gray,
    ):
        """Project semantic features, stereo depth, and illumination into BEV."""
        features, stereo_features, illumination = self.image_encoder(img, gray)
        metas = dict(
            k2s_sensor=k2s_sensor,
            intrins=intrin,
            post_rots=post_rot,
            post_trans=post_tran,
            frustum=self.img_view_transformer.cv_frustum.to(features),
            cv_feat_list=[feat_prev_iv, stereo_features],
        )
        outputs = self.img_view_transformer(
            [
                features,
                sensor2keyego,
                intrin,
                post_rot,
                post_tran,
                bda,
                mlp_input,
            ],
            metas,
            illumination,
        )
        bev, depth, illumination_bev = outputs
        bev = self.pre_process_net(bev)[0]
        return bev, depth, stereo_features, illumination_bev

    def prepare_gray_images(self, gray):
        """Unpack the camera-major grayscale sequence into per-frame tensors."""
        if isinstance(gray, list):
            gray = gray[0]
        batch, packed_views, channels, height, width = gray.shape
        views = packed_views // self.temporal_frame
        gray = gray.view(
            batch, views, self.temporal_frame, channels, height, width
        )
        return [frame.squeeze(2) for frame in torch.split(gray, 1, dim=2)]

    def extract_img_feat(self, img_inputs, img_gray):
        """Fuse the packed key/history sequence into occupancy BEV features."""
        (
            images,
            sensor2keyegos,
            intrins,
            post_rots,
            post_trans,
            bda,
            curr2adjsensor,
        ) = self.prepare_inputs(img_inputs)
        gray = self.prepare_gray_images(img_gray)
        bev_frames, illumination_frames = [], []
        depth_key_frame = None
        with torch.no_grad():
            previous_stereo = self.extract_stereo_ref_feat(
                images[self.temporal_frame]
            )
        for frame_id in range(self.temporal_frame - 1, -1, -1):
            key_frame = frame_id == 0
            sensor2keyego = sensor2keyegos[frame_id]
            mlp_input = self.img_view_transformer.get_mlp_input(
                sensor2keyegos[0],
                intrins[frame_id],
                post_rots[frame_id],
                post_trans[frame_id],
                bda,
            )
            inputs = (
                images[frame_id],
                sensor2keyego,
                intrins[frame_id],
                post_rots[frame_id],
                post_trans[frame_id],
                bda,
                mlp_input,
                previous_stereo,
                curr2adjsensor[frame_id],
                gray[frame_id],
            )
            if key_frame:
                bev, depth, stereo, illumination = self.prepare_bev_feat(
                    *inputs
                )
                depth_key_frame = depth
            else:
                with torch.no_grad():
                    bev, depth, stereo, illumination = self.prepare_bev_feat(
                        *inputs
                    )
            bev_frames.append(bev)
            illumination_frames.append(illumination)
            if not key_frame:
                previous_stereo = stereo

        bev_frames = self.illumination_refinement(
            bev_frames, illumination_frames
        )
        fused_bev = self.bev_encoder(torch.cat(bev_frames, dim=1))
        return [fused_bev], depth_key_frame

    def extract_feat(self, img_inputs, img_gray):
        return self.extract_img_feat(img_inputs, img_gray)

    def forward_train(
        self,
        img_inputs,
        img_gray,
        gt_depth,
        voxel_semantics,
        mask_camera,
        img_metas=None,
    ):
        """Compute stereo depth, occupancy, and fused-BEV SIGReg losses."""
        features, depth = self.extract_feat(img_inputs, img_gray)
        bev = features[0]
        losses = dict(
            loss_depth=self.img_view_transformer.get_depth_loss(
                gt_depth, depth
            )
        )
        losses['loss_sigreg'], losses['sigreg_raw'] = self.sigreg_loss(bev)
        losses.update(
            self.forward_occ_train(bev, voxel_semantics, mask_camera)
        )
        return losses

    def forward_occ_train(self, features, voxel_semantics, mask_camera):
        prediction = self.occ_head(features)
        assert voxel_semantics.min() >= 0 and voxel_semantics.max() <= 17
        return self.occ_head.loss(prediction, voxel_semantics, mask_camera)

    def simple_test(self, img_inputs, img_gray):
        features, _ = self.extract_feat(img_inputs, img_gray)
        return self.occ_head.get_occ(self.occ_head(features[0]))
