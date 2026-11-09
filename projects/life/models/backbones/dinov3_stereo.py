import os
import sys

import torch
import torch.nn as nn
from mmdet.models.backbones.resnet import ResNet
from mmdet3d.models import BACKBONES


@BACKBONES.register_module()
class DINOv3StereoBackbone(nn.Module):
    """DINOv3 semantic backbone with a frozen ResNet stereo stem.

    The output tuple contains the features used by LIFE:
    output[0] is the stride-4 stereo feature and output[1] is the stride-16
    semantic feature consumed by img_neck.
    """

    def __init__(
        self,
        dino_repo,
        dino_weights,
        dino_arch='dinov3_vits16plus',
        dino_layer=11,
        chunk_size=8,
        freeze_dino=True,
        freeze_stereo=True,
    ):
        super().__init__()
        self.dino_repo = os.path.abspath(os.path.expanduser(dino_repo))
        self.dino_weights = os.path.abspath(os.path.expanduser(dino_weights))
        self.dino_arch = dino_arch
        self.dino_layer = int(dino_layer)
        self.chunk_size = int(chunk_size) if chunk_size else 0
        self.freeze_dino = bool(freeze_dino)
        self.freeze_stereo = bool(freeze_stereo)
        self.stereo_branch = ResNet(
            depth=50,
            num_stages=1,
            strides=(1,),
            dilations=(1,),
            out_indices=(0,),
            frozen_stages=-1,
            norm_cfg=dict(type='BN', requires_grad=True),
            norm_eval=False,
            with_cp=False,
            style='pytorch',
            pretrained=None,
        )
        self._is_init = False

        self.dino = self._build_dino()
        self._apply_freeze_policy()

    def _build_dino(self):
        if not os.path.isdir(self.dino_repo):
            raise FileNotFoundError(f'dino_repo not found: {self.dino_repo}')
        if self.dino_repo not in sys.path:
            sys.path.insert(0, self.dino_repo)

        from dinov3.hub import backbones as dinov3_backbones

        if self.dino_arch != 'dinov3_vits16plus':
            raise ValueError(
                'DINOv3StereoBackbone currently supports only '
                f'dinov3_vits16plus, got {self.dino_arch!r}'
            )
        builder = getattr(dinov3_backbones, self.dino_arch)
        return builder(pretrained=False)

    def _freeze_module(self, module):
        for param in module.parameters():
            param.requires_grad_(False)
        module.eval()

    def _apply_freeze_policy(self):
        if self.freeze_dino:
            self._freeze_module(self.dino)
        if self.freeze_stereo:
            self._freeze_module(self.stereo_branch)

    def init_weights(self):
        if self._is_init:
            return
        state_dict = torch.load(
            self.dino_weights, map_location='cpu', weights_only=True
        )
        self.dino.load_state_dict(state_dict, strict=True)
        self.stereo_branch.init_weights()
        self._apply_freeze_policy()
        self._is_init = True

    def train(self, mode=True):
        super().train(mode)
        if self.freeze_dino:
            self.dino.eval()
        if self.freeze_stereo:
            self.stereo_branch.eval()
        return self

    def forward_stereo(self, x):
        feat = self.stereo_branch(x)
        if isinstance(feat, (list, tuple)):
            feat = feat[0]
        return feat

    def _forward_dino_chunk(self, x):
        return self.dino.get_intermediate_layers(
            x,
            n=[self.dino_layer],
            reshape=True,
            return_class_token=False,
            return_extra_tokens=False,
            norm=True,
        )[0].to(dtype=x.dtype)

    def _forward_dino(self, x):
        with torch.set_grad_enabled(
            not self.freeze_dino and torch.is_grad_enabled()
        ):
            if self.chunk_size <= 0 or x.size(0) <= self.chunk_size:
                return self._forward_dino_chunk(x)
            return torch.cat(
                [
                    self._forward_dino_chunk(chunk)
                    for chunk in torch.split(x, self.chunk_size, dim=0)
                ],
                dim=0,
            )

    def forward(self, x):
        stereo_feat = self.forward_stereo(x)
        semantic_feat = self._forward_dino(x)
        return (stereo_feat, semantic_feat)
