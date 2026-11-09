# Copyright (c) OpenMMLab. All rights reserved.
# Modified from Megvii BEVDepth.
import math
import os
from copy import deepcopy

import torch
from mmcv.runner import load_state_dict
from mmcv.runner.dist_utils import master_only
from mmcv.runner.hooks import HOOKS, Hook

from .utils import is_parallel

__all__ = ['ModelEMA']


class ModelEMA:
    """Track moving averages of floating-point parameters and buffers.

    Initialize after device placement and parallel wrapping.
    Adapted from https://github.com/rwightman/pytorch-image-models.
    """

    def __init__(self, model, decay=0.9999, updates=0):
        self.ema_model = deepcopy(model).eval()
        self.ema = self.ema_model.module.module if is_parallel(
            self.ema_model.module) else self.ema_model.module
        self.updates = updates
        # Ramp up the decay to reduce bias during early updates.
        self.decay = lambda x: decay * (1 - math.exp(-x / 2000))
        for p in self.ema.parameters():
            p.requires_grad_(False)

    def update(self, model):
        with torch.no_grad():
            self.updates += 1
            d = self.decay(self.updates)

            msd = model.module.state_dict() if is_parallel(
                model) else model.state_dict()
            for k, v in self.ema.state_dict().items():
                if v.dtype.is_floating_point:
                    v *= d
                    v += (1.0 - d) * msd[k].detach()


@HOOKS.register_module()
class MEGVIIEMAHook(Hook):
    """Update EMA each iteration and save EMA checkpoints each epoch.

    Adapted from
    https://github.com/Megvii-BaseDetection/BEVDepth/blob/main/callbacks/ema.py.
    """

    def __init__(self,
                 init_updates=0,
                 decay=0.9990,
                 resume=None):
        super().__init__()
        self.init_updates = init_updates
        self.resume = resume
        self.decay = decay

    def before_run(self, runner):
        from torch.nn.modules.batchnorm import SyncBatchNorm

        bn_model_list = list()
        bn_model_dist_group_list = list()
        for model_ref in runner.model.modules():
            if isinstance(model_ref, SyncBatchNorm):
                bn_model_list.append(model_ref)
                bn_model_dist_group_list.append(model_ref.process_group)
                model_ref.process_group = None
        runner.ema_model = ModelEMA(runner.model, self.decay)

        for bn_model, dist_group in zip(bn_model_list,
                                        bn_model_dist_group_list):
            bn_model.process_group = dist_group
        runner.ema_model.updates = self.init_updates

        if self.resume is not None:
            runner.logger.info(f'resume ema checkpoint from {self.resume}')
            cpt = torch.load(self.resume, map_location='cpu')
            load_state_dict(runner.ema_model.ema, cpt['state_dict'])
            runner.ema_model.updates = cpt['updates']

    def after_train_iter(self, runner):
        runner.ema_model.update(runner.model.module)

    def after_train_epoch(self, runner):
        self.save_checkpoint(runner)

    @master_only
    def save_checkpoint(self, runner):
        state_dict = runner.ema_model.ema.state_dict()
        ema_checkpoint = {
            'epoch': runner.epoch,
            'state_dict': state_dict,
            'updates': runner.ema_model.updates
        }
        save_path = f'epoch_{runner.epoch+1}_ema.pth'
        save_path = os.path.join(runner.work_dir, save_path)
        torch.save(ema_checkpoint, save_path)
        runner.logger.info(f'Saving ema checkpoint at {save_path}')
