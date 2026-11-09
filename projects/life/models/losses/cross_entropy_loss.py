# Copyright (c) OpenMMLab. All rights reserved.
import torch.nn as nn
import torch.nn.functional as F
from mmdet.models.builder import LOSSES
from mmdet.models.losses.utils import weight_reduce_loss


@LOSSES.register_module(force=True)
class CrossEntropyLoss(nn.Module):
    """Class-weighted softmax cross entropy over camera-visible voxels."""

    def __init__(self, class_weight=None, ignore_index=255, loss_weight=1.0):
        super().__init__()
        self.class_weight = class_weight
        self.ignore_index = ignore_index
        self.loss_weight = loss_weight

    def forward(self, cls_score, label, weight, avg_factor):
        class_weight = cls_score.new_tensor(self.class_weight)
        loss = F.cross_entropy(
            cls_score,
            label,
            weight=class_weight,
            reduction='none',
            ignore_index=self.ignore_index,
        )
        return self.loss_weight * weight_reduce_loss(
            loss,
            weight=weight.float(),
            reduction='mean',
            avg_factor=avg_factor,
        )
