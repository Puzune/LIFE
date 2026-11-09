# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn.functional as F
from mmcv.runner import BaseModule, force_fp32
from mmdet3d.models.builder import NECKS, build_head
from torch.cuda.amp.autocast_mode import autocast

from ...ops import bev_pool_v2
from ..model_utils import DepthNet


@NECKS.register_module()
class LSS_forward_BEVStereo(BaseModule):
    """Stereo lift-splat projection with illumination-aware BEV refinement."""

    def __init__(
        self,
        grid_config,
        input_size,
        in_channels,
        out_channels,
        depthnet_cfg,
        backward_projection,
        downsample=16,
        loss_depth_weight=0.05,
    ):
        super().__init__()
        self.grid_config = grid_config
        self.downsample = downsample
        self.create_grid_infos(
            grid_config['x'], grid_config['y'], grid_config['z']
        )
        self.frustum = self.create_frustum(
            grid_config['depth'], input_size, downsample
        )
        self.out_channels = out_channels
        self.depth_net = DepthNet(
            in_channels=in_channels,
            mid_channels=in_channels,
            context_channels=out_channels,
            depth_channels=self.D,
            **depthnet_cfg
        )
        self.loss_depth_weight = loss_depth_weight
        self.cv_frustum = self.create_frustum(
            grid_config['depth'], input_size, 4
        )
        self.backward_projection = build_head(backward_projection)

    def create_grid_infos(self, x, y, z):
        self.grid_lower_bound = torch.Tensor(
            [cfg[0] for cfg in [x, y, z]]
        )
        self.grid_interval = torch.Tensor(
            [cfg[2] for cfg in [x, y, z]]
        )
        self.grid_size = torch.Tensor(
            [(cfg[1] - cfg[0]) / cfg[2] for cfg in [x, y, z]]
        )

    def create_frustum(self, depth_cfg, input_size, downsample):
        """Create pixel/depth coordinates [D, H, W, 3] in [u, v, depth] order."""
        H_in, W_in = input_size
        H_feat, W_feat = H_in // downsample, W_in // downsample
        d = (
            torch.arange(*depth_cfg, dtype=torch.float)
            .view(-1, 1, 1)
            .expand(-1, H_feat, W_feat)
        )
        self.D = d.shape[0]

        x = (
            torch.linspace(0, W_in - 1, W_feat, dtype=torch.float)
            .view(1, 1, W_feat)
            .expand(self.D, H_feat, W_feat)
        )
        y = (
            torch.linspace(0, H_in - 1, H_feat, dtype=torch.float)
            .view(1, H_feat, 1)
            .expand(self.D, H_feat, W_feat)
        )

        return torch.stack((x, y, d), -1)

    def get_ego_coor(self, sensor2ego, cam2imgs, post_rots, post_trans, bda):
        """Lift frustum points into the BEV-augmented ego frame [B, N, D, H, W, 3]."""
        B, N, _, _ = sensor2ego.shape

        # Undo image augmentation before lifting pixels into 3D.
        points = self.frustum.to(sensor2ego) - post_trans.view(
            B, N, 1, 1, 1, 3
        )
        points = (
            torch.inverse(post_rots)
            .view(B, N, 1, 1, 1, 3, 3)
            .matmul(points.unsqueeze(-1))
        )

        points = torch.cat(
            (points[..., :2, :] * points[..., 2:3, :], points[..., 2:3, :]), 5
        )
        combine = sensor2ego[:, :, :3, :3].matmul(torch.inverse(cam2imgs))
        points = combine.view(B, N, 1, 1, 1, 3, 3).matmul(points).squeeze(-1)
        points += sensor2ego[:, :, :3, 3].view(B, N, 1, 1, 1, 3)

        points = (
            bda.view(B, 1, 1, 1, 1, 3, 3)
            .matmul(points.unsqueeze(-1))
            .squeeze(-1)
        )
        return points

    def voxel_pooling_v2(self, coor, depth, feat):
        """Pool depth-weighted features into BEV [B, C*Dz, Dy, Dx].

        coor uses augmented ego coordinates [B, N, D, H, W, 3]. depth and feat
        use [B, N, D, H, W] and [B, N, C, H, W]. Empty grids return zeros.
        """
        (
            ranks_bev,
            ranks_depth,
            ranks_feat,
            interval_starts,
            interval_lengths,
        ) = self.voxel_pooling_prepare_v2(coor)

        if ranks_feat is None:
            print(
                'warning ---> no points within the predefined '
                'bev receptive field'
            )
            dummy = torch.zeros(
                size=[
                    feat.shape[0],
                    feat.shape[2],
                    int(self.grid_size[2]),
                    int(self.grid_size[1]),
                    int(self.grid_size[0]),
                ]
            ).to(
                feat
            )
            dummy = torch.cat(dummy.unbind(dim=2), 1)
            return dummy

        feat = feat.permute(0, 1, 3, 4, 2)
        bev_feat_shape = (
            depth.shape[0],
            int(self.grid_size[2]),
            int(self.grid_size[1]),
            int(self.grid_size[0]),
            feat.shape[-1],
        )
        bev_feat = bev_pool_v2(
            depth,
            feat,
            ranks_depth,
            ranks_feat,
            ranks_bev,
            bev_feat_shape,
            interval_starts,
            interval_lengths,
        )
        bev_feat = torch.cat(bev_feat.unbind(dim=2), 1)
        return bev_feat

    def voxel_pooling_prepare_v2(self, coor):
        """Return voxel/depth/feature indices and intervals sorted by output voxel.

        Indices have one entry per retained point. Intervals delimit points
        in the same voxel. All five outputs are None if no points are in bounds.
        """
        B, N, D, H, W, _ = coor.shape
        num_points = B * N * D * H * W
        ranks_depth = torch.range(
            0, num_points - 1, dtype=torch.int, device=coor.device
        )
        ranks_feat = torch.range(
            0, num_points // D - 1, dtype=torch.int, device=coor.device
        )
        ranks_feat = ranks_feat.reshape(B, N, 1, H, W)
        ranks_feat = ranks_feat.expand(
            B, N, D, H, W
        ).flatten()

        coor = (coor - self.grid_lower_bound.to(coor)) / self.grid_interval.to(
            coor
        )
        coor = coor.long().view(
            num_points, 3
        )
        batch_idx = (
            torch.range(0, B - 1)
            .reshape(B, 1)
            .expand(B, num_points // B)
            .reshape(num_points, 1)
            .to(coor)
        )
        coor = torch.cat(
            (coor, batch_idx), 1
        )

        kept = (
            (coor[:, 0] >= 0)
            & (coor[:, 0] < self.grid_size[0])
            & (coor[:, 1] >= 0)
            & (coor[:, 1] < self.grid_size[1])
            & (coor[:, 2] >= 0)
            & (coor[:, 2] < self.grid_size[2])
        )
        if len(kept) == 0:
            return None, None, None, None, None


        coor, ranks_depth, ranks_feat = (
            coor[kept],
            ranks_depth[kept],
            ranks_feat[kept],
        )

        # Group points by output voxel for the CUDA reduction.
        ranks_bev = coor[:, 3] * (
            self.grid_size[2] * self.grid_size[1] * self.grid_size[0]
        )
        ranks_bev += coor[:, 2] * (self.grid_size[1] * self.grid_size[0])
        ranks_bev += coor[:, 1] * self.grid_size[0] + coor[:, 0]
        order = ranks_bev.argsort()
        ranks_bev, ranks_depth, ranks_feat = (
            ranks_bev[order],
            ranks_depth[order],
            ranks_feat[order],
        )

        kept = torch.ones(
            ranks_bev.shape[0], device=ranks_bev.device, dtype=torch.bool
        )
        kept[1:] = ranks_bev[1:] != ranks_bev[:-1]
        interval_starts = torch.where(kept)[0].int()
        if len(interval_starts) == 0:
            return None, None, None, None, None
        interval_lengths = torch.zeros_like(interval_starts)
        interval_lengths[:-1] = interval_starts[1:] - interval_starts[:-1]
        interval_lengths[-1] = ranks_bev.shape[0] - interval_starts[-1]

        return (
            ranks_bev.int().contiguous(),
            ranks_depth.int().contiguous(),
            ranks_feat.int().contiguous(),
            interval_starts.int().contiguous(),
            interval_lengths.int().contiguous(),
        )

    def view_transform(self, input, depth, tran_feat):
        """Pool per-camera context using depth and calibrated ego coordinates."""
        B, N, _, H, W = input[0].shape
        coor = self.get_ego_coor(*input[1:6])
        bev_feat = self.voxel_pooling_v2(
            coor,
            depth.view(B, N, self.D, H, W),
            tran_feat.view(B, N, self.out_channels, H, W),
        )
        return bev_feat, depth

    def get_mlp_input(self, sensor2ego, intrin, post_rot, post_tran, bda):
        """Pack intrinsics, image/BEV augmentation, and extrinsics into [B, N, 27]."""
        B, N, _, _ = sensor2ego.shape
        bda = bda.view(B, 1, 3, 3).repeat(
            1, N, 1, 1
        )
        mlp_input = torch.stack(
            [
                intrin[:, :, 0, 0],
                intrin[:, :, 1, 1],
                intrin[:, :, 0, 2],
                intrin[:, :, 1, 2],
                post_rot[:, :, 0, 0],
                post_rot[:, :, 0, 1],
                post_tran[:, :, 0],
                post_rot[:, :, 1, 0],
                post_rot[:, :, 1, 1],
                post_tran[:, :, 1],
                bda[:, :, 0, 0],
                bda[:, :, 0, 1],
                bda[:, :, 1, 0],
                bda[:, :, 1, 1],
                bda[:, :, 2, 2],
            ],
            dim=-1,
        )
        sensor2ego = sensor2ego[:, :, :3, :].reshape(B, N, -1)
        mlp_input = torch.cat(
            [mlp_input, sensor2ego], dim=-1
        )
        return mlp_input

    def forward_bev_feat(self, input, stereo_metas):
        """Return lifted BEV, per-camera context, and stereo depth probabilities."""
        x, mlp_input = input[0], input[6]

        B, N, C, H, W = x.shape
        x = x.view(B * N, C, H, W)
        x = self.depth_net(
            x, mlp_input, stereo_metas
        )
        depth_digit = x[:, : self.D, ...]
        tran_feat = x[
            :, self.D : self.D + self.out_channels, ...
        ]
        depth = depth_digit.softmax(dim=1)
        bev_feat, depth = self.view_transform(input, depth, tran_feat)

        return bev_feat, tran_feat, depth

    def get_downsampled_gt_depth(self, gt_depths):
        """Encode the nearest nonzero depth per patch as a one-hot bin."""
        B, N, H, W = gt_depths.shape
        gt_depths = gt_depths.view(
            B * N,
            H // self.downsample,
            self.downsample,
            W // self.downsample,
            self.downsample,
            1,
        )
        gt_depths = gt_depths.permute(0, 1, 3, 5, 2, 4).contiguous()
        gt_depths = gt_depths.view(-1, self.downsample * self.downsample)
        gt_depths_tmp = torch.where(
            gt_depths == 0.0, 1e5 * torch.ones_like(gt_depths), gt_depths
        )
        gt_depths = torch.min(gt_depths_tmp, dim=-1).values
        gt_depths = gt_depths.view(
            B * N, H // self.downsample, W // self.downsample
        )

        # Reserve bin zero for invalid depths.
        gt_depths = (
            gt_depths
            - (self.grid_config['depth'][0] - self.grid_config['depth'][2])
        ) / self.grid_config['depth'][2]

        gt_depths = torch.where(
            (gt_depths < self.D + 1) & (gt_depths >= 0.0),
            gt_depths,
            torch.zeros_like(gt_depths),
        )
        gt_depths = F.one_hot(gt_depths.long(), num_classes=self.D + 1).view(
            -1, self.D + 1
        )[
            :, 1:
        ]
        return gt_depths.float()

    @force_fp32()
    def get_depth_loss(self, depth_labels, depth_preds):
        depth_labels = self.get_downsampled_gt_depth(
            depth_labels
        )
        depth_preds = (
            depth_preds.permute(0, 2, 3, 1).contiguous().view(-1, self.D)
        )
        fg_mask = torch.max(depth_labels, dim=1).values > 0.0
        depth_labels = depth_labels[fg_mask]
        depth_preds = depth_preds[fg_mask]
        with autocast(enabled=False):
            depth_loss = F.binary_cross_entropy(
                depth_preds,
                depth_labels,
                reduction='none',
            ).sum() / max(1.0, fg_mask.sum())
        return self.loss_depth_weight * depth_loss

    def forward(self, input, stereo_metas=None, illu_map=None):
        """Return illumination-refined BEV, depth, and BEV illumination weights."""
        bev_forward_feat, context, depth = self.forward_bev_feat(
            input, stereo_metas
        )

        _, C, H, W = context.shape

        _, Cd, _, _ = depth.shape

        bev_feat_refined, illu_bev_weight = self.backward_projection(
            mlvl_feats=[context.view(-1, 6, C, H, W)],
            pred_img_depth=depth.view(-1, 6, Cd, H, W),
            img_inputs=input,
            lss_bev=bev_forward_feat,
            illu_map=illu_map,
        )

        illu_bev_weight = torch.clamp(
            illu_bev_weight, min=0.0, max=1.0
        )
        bev_feat = (
            bev_feat_refined * (illu_bev_weight + 0.50) + bev_forward_feat
        )

        return bev_feat, depth, illu_bev_weight
