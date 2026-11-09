# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
from mmdet3d.models.builder import NECKS


@NECKS.register_module()
class IlluminationSelector(nn.Module):
    """Select enhanced images when their mean illumination is below threshold."""

    def __init__(self, thre_light=0.4953):
        super(IlluminationSelector, self).__init__()
        self.thre_light = thre_light

    def forward(self, img_denorm, img_deillu, ill_first):
        batch_size = img_deillu.shape[0]

        corrected_images = []
        for i in range(batch_size):
            img_origin = img_denorm[i]
            img_enhance = img_deillu[i]

            ill_fac = ill_first[i].mean()

            if ill_fac >= self.thre_light:
                img_corrected = img_origin
            else:
                img_corrected = img_enhance

            corrected_images.append(img_corrected)

        corrected_images = torch.stack(corrected_images, dim=0)
        return corrected_images
