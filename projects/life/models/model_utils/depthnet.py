import torch
import torch.nn as nn
import torch.nn.functional as F
from mmdet.models.backbones.resnet import BasicBlock
from torch.utils.checkpoint import checkpoint

try:
    import triton
    import triton.language as tl
except ImportError:  # pragma: no cover - optional runtime dependency
    triton = None
    tl = None


if triton is not None:

    @triton.jit
    def _cost_volume_fused_nhwc_kernel(
        prev,
        curr,
        grid,
        out,
        total,
        C: tl.constexpr,
        D: tl.constexpr,
        H: tl.constexpr,
        W: tl.constexpr,
        BIAS: tl.constexpr,
        INVALID_CHANNEL: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_C: tl.constexpr,
    ):
        offsets_m = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
        mask_m = offsets_m < total

        w_idx = offsets_m % W
        h_idx = (offsets_m // W) % H
        d_idx = (offsets_m // (H * W)) % D
        bn_idx = offsets_m // (D * H * W)

        grid_base = ((bn_idx * D * H + d_idx * H + h_idx) * W + w_idx) * 2
        gx = tl.load(grid + grid_base, mask=mask_m, other=-2.0)
        gy = tl.load(grid + grid_base + 1, mask=mask_m, other=-2.0)

        x = (gx + 1.0) * ((W - 1) * 0.5)
        y = (gy + 1.0) * ((H - 1) * 0.5)
        x0 = tl.floor(x).to(tl.int32)
        y0 = tl.floor(y).to(tl.int32)
        x1 = x0 + 1
        y1 = y0 + 1
        wx = x - x0
        wy = y - y0

        bn = bn_idx[:, None]
        h = h_idx[:, None]
        w = w_idx[:, None]
        x0_b = x0[:, None]
        x1_b = x1[:, None]
        y0_b = y0[:, None]
        y1_b = y1[:, None]
        wx_b = wx[:, None]
        wy_b = wy[:, None]
        mask_b = mask_m[:, None]

        accum = tl.zeros((BLOCK_M,), dtype=tl.float32)
        channels = tl.arange(0, BLOCK_C)
        for channel_start in tl.static_range(0, C, BLOCK_C):
            c = channel_start + channels
            mask_c = c < C
            c_b = c[None, :]

            curr_offset = ((bn * H + h) * W + w) * C + c_b
            curr_val = tl.load(
                curr + curr_offset, mask=mask_b & mask_c[None, :], other=0.0
            ).to(tl.float32)

            base00 = ((bn * H + y0_b) * W + x0_b) * C + c_b
            base01 = ((bn * H + y0_b) * W + x1_b) * C + c_b
            base10 = ((bn * H + y1_b) * W + x0_b) * C + c_b
            base11 = ((bn * H + y1_b) * W + x1_b) * C + c_b

            m00 = (
                mask_b
                & mask_c[None, :]
                & (x0_b >= 0)
                & (x0_b < W)
                & (y0_b >= 0)
                & (y0_b < H)
            )
            m01 = (
                mask_b
                & mask_c[None, :]
                & (x1_b >= 0)
                & (x1_b < W)
                & (y0_b >= 0)
                & (y0_b < H)
            )
            m10 = (
                mask_b
                & mask_c[None, :]
                & (x0_b >= 0)
                & (x0_b < W)
                & (y1_b >= 0)
                & (y1_b < H)
            )
            m11 = (
                mask_b
                & mask_c[None, :]
                & (x1_b >= 0)
                & (x1_b < W)
                & (y1_b >= 0)
                & (y1_b < H)
            )

            v00 = tl.load(prev + base00, mask=m00, other=0.0).to(tl.float32)
            v01 = tl.load(prev + base01, mask=m01, other=0.0).to(tl.float32)
            v10 = tl.load(prev + base10, mask=m10, other=0.0).to(tl.float32)
            v11 = tl.load(prev + base11, mask=m11, other=0.0).to(tl.float32)

            sample = (
                v00 * (1.0 - wx_b) * (1.0 - wy_b)
                + v01 * wx_b * (1.0 - wy_b)
                + v10 * (1.0 - wx_b) * wy_b
                + v11 * wx_b * wy_b
            )
            accum += tl.sum(tl.abs(sample - curr_val), axis=1)

        if BIAS != 0.0:
            ic = INVALID_CHANNEL
            invalid_off00 = ((bn_idx * H + y0) * W + x0) * C + ic
            invalid_off01 = ((bn_idx * H + y0) * W + x1) * C + ic
            invalid_off10 = ((bn_idx * H + y1) * W + x0) * C + ic
            invalid_off11 = ((bn_idx * H + y1) * W + x1) * C + ic
            invalid_m00 = mask_m & (x0 >= 0) & (x0 < W) & (y0 >= 0) & (y0 < H)
            invalid_m01 = mask_m & (x1 >= 0) & (x1 < W) & (y0 >= 0) & (y0 < H)
            invalid_m10 = mask_m & (x0 >= 0) & (x0 < W) & (y1 >= 0) & (y1 < H)
            invalid_m11 = mask_m & (x1 >= 0) & (x1 < W) & (y1 >= 0) & (y1 < H)
            invalid_v00 = tl.load(
                prev + invalid_off00, mask=invalid_m00, other=0.0
            ).to(tl.float32)
            invalid_v01 = tl.load(
                prev + invalid_off01, mask=invalid_m01, other=0.0
            ).to(tl.float32)
            invalid_v10 = tl.load(
                prev + invalid_off10, mask=invalid_m10, other=0.0
            ).to(tl.float32)
            invalid_v11 = tl.load(
                prev + invalid_off11, mask=invalid_m11, other=0.0
            ).to(tl.float32)
            invalid_sample = (
                invalid_v00 * (1.0 - wx) * (1.0 - wy)
                + invalid_v01 * wx * (1.0 - wy)
                + invalid_v10 * (1.0 - wx) * wy
                + invalid_v11 * wx * wy
            )
            accum += tl.where(invalid_sample == 0.0, BIAS, 0.0)

        tl.store(out + offsets_m, accum, mask=mask_m)


def triton_fused_cost_volume(prev, curr, grid, B, N, D, H, W, bias):
    if triton is None:
        raise RuntimeError('Stereo cost-volume fusion requires Triton')
    if not prev.is_cuda or not curr.is_cuda or not grid.is_cuda:
        raise RuntimeError('Stereo cost-volume fusion requires CUDA tensors')
    if torch.is_grad_enabled():
        raise RuntimeError(
            'Stereo cost-volume fusion is not allowed with ' 'autograd enabled'
        )

    prev = prev.contiguous()
    curr = curr.contiguous()
    grid = grid.contiguous()
    _, C, _, _ = curr.shape
    output = curr.new_empty(B * N, D, H, W)
    total = B * N * D * H * W
    invalid_channel = C - 4 if bias != 0 else 0
    prev_nhwc = prev.permute(0, 2, 3, 1).contiguous()
    curr_nhwc = curr.permute(0, 2, 3, 1).contiguous()
    _cost_volume_fused_nhwc_kernel[(triton.cdiv(total, 16),)](
        prev_nhwc,
        curr_nhwc,
        grid,
        output,
        total,
        C,
        D,
        H,
        W,
        float(bias),
        invalid_channel,
        BLOCK_M=16,
        BLOCK_C=32,
        num_warps=4,
    )
    return output


class _ASPPModule(nn.Module):
    def __init__(
        self, inplanes, planes, kernel_size, padding, dilation, BatchNorm
    ):
        super(_ASPPModule, self).__init__()
        self.atrous_conv = nn.Conv2d(
            inplanes,
            planes,
            kernel_size=kernel_size,
            stride=1,
            padding=padding,
            dilation=dilation,
            bias=False,
        )
        self.bn = BatchNorm(planes)
        self.relu = nn.ReLU()

        self._init_weight()

    def forward(self, x):
        x = self.atrous_conv(x)
        x = self.bn(x)

        return self.relu(x)

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


class ASPP(nn.Module):
    def __init__(self, inplanes, mid_channels=256, BatchNorm=nn.BatchNorm2d):
        super(ASPP, self).__init__()
        dilations = [1, 6, 12, 18]
        self.aspp1 = _ASPPModule(
            inplanes,
            mid_channels,
            1,
            padding=0,
            dilation=dilations[0],
            BatchNorm=BatchNorm,
        )
        self.aspp2 = _ASPPModule(
            inplanes,
            mid_channels,
            3,
            padding=dilations[1],
            dilation=dilations[1],
            BatchNorm=BatchNorm,
        )
        self.aspp3 = _ASPPModule(
            inplanes,
            mid_channels,
            3,
            padding=dilations[2],
            dilation=dilations[2],
            BatchNorm=BatchNorm,
        )
        self.aspp4 = _ASPPModule(
            inplanes,
            mid_channels,
            3,
            padding=dilations[3],
            dilation=dilations[3],
            BatchNorm=BatchNorm,
        )

        self.global_avg_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(inplanes, mid_channels, 1, stride=1, bias=False),
            BatchNorm(mid_channels),
            nn.ReLU(),
        )
        self.conv1 = nn.Conv2d(int(mid_channels * 5), inplanes, 1, bias=False)
        self.bn1 = BatchNorm(inplanes)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self._init_weight()

    def forward(self, x):
        x1 = self.aspp1(x)
        x2 = self.aspp2(x)
        x3 = self.aspp3(x)
        x4 = self.aspp4(x)
        x5 = self.global_avg_pool(x)
        x5 = F.interpolate(
            x5, size=x4.size()[2:], mode='bilinear', align_corners=True
        )
        x = torch.cat((x1, x2, x3, x4, x5), dim=1)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        return self.dropout(x)

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


class Mlp(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.ReLU,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x


class SELayer(nn.Module):
    def __init__(self, channels, act_layer=nn.ReLU, gate_layer=nn.Sigmoid):
        super().__init__()
        self.conv_reduce = nn.Conv2d(channels, channels, 1, bias=True)
        self.act1 = act_layer()
        self.conv_expand = nn.Conv2d(channels, channels, 1, bias=True)
        self.gate = gate_layer()

    def forward(self, x, x_se):
        x_se = self.conv_reduce(x_se)
        x_se = self.act1(x_se)
        x_se = self.conv_expand(x_se)
        return x * self.gate(x_se)


class DepthNet(nn.Module):
    def __init__(
        self,
        in_channels,
        mid_channels,
        context_channels,
        depth_channels,
        with_cp=False,
        bias=5.0,
        aspp_mid_channels=96,
    ):
        super(DepthNet, self).__init__()
        self.reduce_conv = nn.Sequential(
            nn.Conv2d(
                in_channels, mid_channels, kernel_size=3, stride=1, padding=1
            ),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
        )

        self.context_conv = nn.Conv2d(
            mid_channels, context_channels, kernel_size=1, stride=1, padding=0
        )

        self.bn = nn.BatchNorm1d(27)
        self.depth_mlp = Mlp(
            in_features=27,
            hidden_features=mid_channels,
            out_features=mid_channels,
        )
        self.depth_se = SELayer(
            channels=mid_channels
        )
        self.context_mlp = Mlp(
            in_features=27,
            hidden_features=mid_channels,
            out_features=mid_channels,
        )
        self.context_se = SELayer(
            channels=mid_channels
        )
        depth_conv_input_channels = mid_channels + depth_channels
        downsample = nn.Conv2d(
            depth_conv_input_channels, mid_channels, 1, 1, 0
        )
        cost_volumn_net = []
        for _ in range(2):
            cost_volumn_net.extend(
                [
                    nn.Conv2d(
                        depth_channels,
                        depth_channels,
                        kernel_size=3,
                        stride=2,
                        padding=1,
                    ),
                    nn.BatchNorm2d(depth_channels),
                ]
            )
        self.cost_volumn_net = nn.Sequential(*cost_volumn_net)
        self.bias = bias

        depth_conv_list = [
            BasicBlock(
                depth_conv_input_channels, mid_channels, downsample=downsample
            ),
            BasicBlock(mid_channels, mid_channels),
            BasicBlock(mid_channels, mid_channels),
        ]
        depth_conv_list.append(ASPP(mid_channels, aspp_mid_channels))
        depth_conv_list.append(
            nn.Conv2d(
                mid_channels,
                depth_channels,
                kernel_size=1,
                stride=1,
                padding=0,
            )
        )
        self.depth_conv = nn.Sequential(*depth_conv_list)
        self.with_cp = with_cp

    def gen_grid(self, metas, B, N, D, H, W, hi, wi):
        frustum = metas['frustum']
        view_shape = (B, N, 1, 1, 1)
        x = frustum[..., 0].view(1, 1, D, H, W) - metas['post_trans'][
            ..., 0
        ].view(*view_shape)
        y = frustum[..., 1].view(1, 1, D, H, W) - metas['post_trans'][
            ..., 1
        ].view(*view_shape)
        z = frustum[..., 2].view(1, 1, D, H, W) - metas['post_trans'][
            ..., 2
        ].view(*view_shape)

        inv_post_rots = torch.inverse(metas['post_rots'])
        inv = inv_post_rots.view(B, N, 1, 1, 1, 3, 3)
        x_aug = inv[..., 0, 0] * x + inv[..., 0, 1] * y + inv[..., 0, 2] * z
        y_aug = inv[..., 1, 0] * x + inv[..., 1, 1] * y + inv[..., 1, 2] * z
        z_aug = inv[..., 2, 0] * x + inv[..., 2, 1] * y + inv[..., 2, 2] * z
        cam_x = x_aug * z_aug
        cam_y = y_aug * z_aug
        cam_z = z_aug

        rots = metas['k2s_sensor'][:, :, :3, :3].contiguous()
        trans = metas['k2s_sensor'][:, :, :3, 3].contiguous()
        combine = rots.matmul(torch.inverse(metas['intrins']))
        proj = metas['intrins'].matmul(combine)
        proj_trans = metas['intrins'].matmul(trans.unsqueeze(-1)).squeeze(-1)
        proj = proj.view(B, N, 1, 1, 1, 3, 3)
        proj_trans = proj_trans.view(B, N, 1, 1, 1, 3)

        out_x = (
            proj[..., 0, 0] * cam_x
            + proj[..., 0, 1] * cam_y
            + proj[..., 0, 2] * cam_z
            + proj_trans[..., 0]
        )
        out_y = (
            proj[..., 1, 0] * cam_x
            + proj[..., 1, 1] * cam_y
            + proj[..., 1, 2] * cam_z
            + proj_trans[..., 1]
        )
        out_z = (
            proj[..., 2, 0] * cam_x
            + proj[..., 2, 1] * cam_y
            + proj[..., 2, 2] * cam_z
            + proj_trans[..., 2]
        )

        combine_z = combine[:, :, 2, :].view(B, N, 1, 1, 1, 3)
        neg_mask = (
            combine_z[..., 0] * cam_x
            + combine_z[..., 1] * cam_y
            + combine_z[..., 2] * cam_z
            + trans[:, :, 2].view(*view_shape)
        ) < 1e-3

        px = out_x / out_z
        py = out_y / out_z
        post_rots = metas['post_rots'][..., :2, :2].view(B, N, 1, 1, 1, 2, 2)
        aug_x = (
            post_rots[..., 0, 0] * px
            + post_rots[..., 0, 1] * py
            + metas['post_trans'][..., 0].view(*view_shape)
        )
        aug_y = (
            post_rots[..., 1, 0] * px
            + post_rots[..., 1, 1] * py
            + metas['post_trans'][..., 1].view(*view_shape)
        )

        px = aug_x / (wi - 1.0) * 2.0 - 1.0
        py = aug_y / (hi - 1.0) * 2.0 - 1.0
        px[neg_mask] = -2
        py[neg_mask] = -2
        grid = torch.stack([px, py], dim=-1)
        return grid.view(B * N, D * H, W, 2)

    def calculate_cost_volumn(self, metas):
        """Return stereo depth probabilities [B*N, D, H, W].

        metas contains camera calibration, augmentation, key-to-reference
        camera transforms, a [u, v, depth] frustum, and previous/current features.
        """
        prev, curr = metas[
            'cv_feat_list'
        ]
        _, c, hf, wf = curr.shape
        if self.bias != 0 and c < 4:
            raise ValueError(
                'Stereo features require at least four channels for the invalid-point mask.'
            )
        hi, wi = hf * 4, wf * 4
        B, N, _ = metas['post_trans'].shape
        D, H, W, _ = metas['frustum'].shape
        grid = self.gen_grid(metas, B, N, D, H, W, hi, wi).to(
            curr.dtype
        )

        prev = prev.view(
            B * N, -1, H, W
        )
        curr = curr.view(
            B * N, -1, H, W
        )
        cost_volumn = triton_fused_cost_volume(
            prev, curr, grid, B, N, D, H, W, self.bias
        )
        return (-cost_volumn).softmax(dim=1)

    def forward(self, x, mlp_input, stereo_metas):
        """Return depth logits followed by context features [B*N, D+C, H, W].

        mlp_input contains 27 camera parameters per view. Stereo metadata
        follows calculate_cost_volumn.
        """
        mlp_input = self.bn(
            mlp_input.reshape(-1, mlp_input.shape[-1])
        )
        x = self.reduce_conv(x)

        context_se = self.context_mlp(mlp_input)[..., None, None]
        context = self.context_se(x, context_se)
        context = self.context_conv(context)

        depth_se = self.depth_mlp(mlp_input)[..., None, None]
        depth = self.depth_se(x, depth_se)

        with torch.no_grad():
            cost_volumn = self.calculate_cost_volumn(stereo_metas)
        cost_volumn = self.cost_volumn_net(cost_volumn)
        depth = torch.cat([depth, cost_volumn], dim=1)
        if self.with_cp:
            depth = checkpoint(self.depth_conv, depth)
        else:
            depth = self.depth_conv(
                depth
            )
        return torch.cat([depth, context], dim=1)
