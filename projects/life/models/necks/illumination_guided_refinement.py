import torch
import torch.nn as nn

from mmdet3d.models.builder import NECKS


@NECKS.register_module()
class IlluminationGuidedRefinement(nn.Module):
    """Refine the key BEV with illumination-weighted, aligned history features."""

    def __init__(
        self,
        num_frames=2,
        channels=80,
        hidden_channels=64,
        key_frame_index=-1,
        max_residual=0.5,
        norm_groups=8,
        eps=1e-4,
    ):
        super().__init__()
        if num_frames < 2:
            raise ValueError(
                'IlluminationGuidedRefinement requires at least two frames.'
            )
        if hidden_channels % norm_groups != 0:
            raise ValueError(
                'hidden_channels must be divisible by norm_groups.'
            )
        if max_residual < 0:
            raise ValueError('max_residual must be non-negative.')

        self.num_frames = num_frames
        self.channels = channels
        self.key_frame_index = key_frame_index
        self.max_residual = max_residual
        self.eps = eps

        gate_in_channels = channels * 2 + 4
        self.gate_net = nn.Sequential(
            nn.Conv2d(
                gate_in_channels, hidden_channels, 3, padding=1, bias=False
            ),
            nn.GroupNorm(norm_groups, hidden_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, 1, 1),
        )
        self.residual_net = nn.Sequential(
            nn.Conv2d(channels, hidden_channels, 3, padding=1, bias=False),
            nn.GroupNorm(norm_groups, hidden_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, channels, 1),
        )

        nn.init.constant_(self.gate_net[-1].weight, 0.0)
        nn.init.constant_(self.gate_net[-1].bias, 0.0)
        nn.init.constant_(self.residual_net[-1].weight, 0.0)
        nn.init.constant_(self.residual_net[-1].bias, 0.0)

    def _key_index(self, num_frames):
        key_index = self.key_frame_index
        if key_index < 0:
            key_index += num_frames
        if key_index < 0 or key_index >= num_frames:
            raise AssertionError(
                f'key_frame_index {self.key_frame_index} is invalid for '
                f'{num_frames} frames.'
            )
        return key_index

    def _validate_inputs(self, bev_feat_list, illu_bev_weight_list):
        if len(bev_feat_list) != self.num_frames:
            raise AssertionError(
                f'Expected {self.num_frames} BEV frames, got {len(bev_feat_list)}.'
            )
        if len(illu_bev_weight_list) != self.num_frames:
            raise AssertionError(
                f'Expected {self.num_frames} illumination BEV maps, got '
                f'{len(illu_bev_weight_list)}.'
            )

        ref_shape = bev_feat_list[0].shape
        if len(ref_shape) != 4:
            raise AssertionError(
                f'BEV features must be 4D tensors, got shape {ref_shape}.'
            )
        batch, channels, height, width = ref_shape
        if channels != self.channels:
            raise AssertionError(
                f'Expected {self.channels} channels per BEV frame, got {channels}.'
            )

        ref_device = bev_feat_list[0].device
        for feat in bev_feat_list[1:]:
            if feat.shape != ref_shape:
                raise AssertionError(
                    f'All BEV frames must share shape {ref_shape}, got {feat.shape}.'
                )
            if feat.device != ref_device:
                raise AssertionError(
                    'All BEV frames must be on the same device.'
                )

        illu_list = []
        for illu in illu_bev_weight_list:
            if illu is None:
                raise AssertionError('Illumination BEV map is missing.')
            if len(illu.shape) != 4:
                raise AssertionError(
                    f'Illumination BEV maps must be 4D tensors, got {illu.shape}.'
                )
            if (
                illu.shape[0] != batch
                or illu.shape[1] != 1
                or illu.shape[2] != height
                or illu.shape[3] != width
            ):
                raise AssertionError(
                    'Illumination BEV maps must have shape [B, 1, H, W] '
                    f'matching BEV features, got {illu.shape}.'
                )
            if illu.device != ref_device:
                raise AssertionError(
                    'Illumination BEV maps must be on the same device as BEV features.'
                )
            illu_list.append(
                illu.to(dtype=bev_feat_list[0].dtype).clamp(0.0, 1.0)
            )
        return illu_list

    def forward(self, bev_feat_list, illu_bev_weight_list):
        illu_bev_weight_list = self._validate_inputs(
            bev_feat_list, illu_bev_weight_list
        )
        key_index = self._key_index(len(bev_feat_list))
        history_indices = [
            idx for idx in range(len(bev_feat_list)) if idx != key_index
        ]

        key_feat = bev_feat_list[key_index]
        current_illu = illu_bev_weight_list[key_index]

        history_feats = torch.stack(
            [bev_feat_list[idx] for idx in history_indices], dim=1
        )
        history_illus = torch.stack(
            [illu_bev_weight_list[idx] for idx in history_indices], dim=1
        )

        history_illu = history_illus.max(dim=1).values
        history_weights = history_illus / (
            history_illus.sum(dim=1, keepdim=True) + self.eps
        )
        if history_weights.shape != history_illus.shape:
            raise AssertionError(
                'History illumination weights must preserve the frame axis.'
            )
        history_feat = (history_feats * history_weights).sum(dim=1)

        recover_prior = (1.0 - current_illu) * history_illu
        illu_delta = history_illu - current_illu
        gate_input = torch.cat(
            [
                key_feat,
                history_feat,
                current_illu,
                history_illu,
                recover_prior,
                illu_delta,
            ],
            dim=1,
        )
        gate = torch.sigmoid(self.gate_net(gate_input))
        residual = self.residual_net(history_feat - key_feat)

        enhanced_key = (
            key_feat + self.max_residual * gate * recover_prior * residual
        )
        outputs = list(bev_feat_list)
        outputs[key_index] = enhanced_key
        return outputs
