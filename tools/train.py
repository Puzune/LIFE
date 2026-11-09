# Copyright (c) OpenMMLab. All rights reserved.
"""Train the LIFE occupancy model."""
import argparse
import os
from os import path as osp
import time

from _runtime import set_cuda_device, setup_import_path

setup_import_path()


def parse_args():
    from mmcv import DictAction

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('config', help='training configuration')
    parser.add_argument('--work-dir', help='directory for logs and checkpoints')
    parser.add_argument('--resume-from', help='checkpoint from which to resume')
    parser.add_argument('--validate', action='store_true', help='evaluate during training')
    parser.add_argument('--gpu-id', type=int, default=0, help='visible CUDA device for one process')
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument('--deterministic', action='store_true', help='use deterministic cuDNN settings')
    parser.add_argument('--cfg-options', nargs='+', action=DictAction,
                        help='configuration overrides, e.g. data.samples_per_gpu=1')
    parser.add_argument('--launcher', choices=['none', 'pytorch'], default='none')
    parser.add_argument('--local-rank', '--local_rank', dest='local_rank', type=int, default=0)
    return parser.parse_args()


def main():
    args = parse_args()
    import mmcv
    import torch
    import torch.distributed as dist
    from mmcv import Config
    from mmcv.runner import get_dist_info, init_dist
    from mmcv.utils import import_modules_from_strings
    from mmdet import __version__ as mmdet_version
    from mmdet.apis import set_random_seed
    from mmdet.utils import setup_multi_processes
    from mmdet3d import __version__ as mmdet3d_version
    from mmdet3d.apis import train_model
    from mmdet3d.datasets import build_dataset
    from mmdet3d.models import build_model
    from mmdet3d.utils import collect_env, get_root_logger
    from mmseg import __version__ as mmseg_version
    from torch2_compat import patch_mmcv_scatter_for_torch2

    os.environ.setdefault('LOCAL_RANK', str(args.local_rank))
    set_cuda_device(args)
    patch_mmcv_scatter_for_torch2()
    cfg = Config.fromfile(args.config)
    if args.cfg_options:
        cfg.merge_from_dict(args.cfg_options)
    setup_multi_processes(cfg)
    if cfg.get('custom_imports'):
        import_modules_from_strings(**cfg.custom_imports)
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True

    cfg.work_dir = args.work_dir or cfg.get('work_dir') or osp.join('outputs', osp.splitext(osp.basename(args.config))[0])
    if args.resume_from:
        cfg.resume_from = args.resume_from
    distributed = args.launcher != 'none'
    if distributed:
        init_dist(args.launcher, **cfg.dist_params)
        _, world_size = get_dist_info()
        cfg.gpu_ids = range(world_size)
    else:
        cfg.gpu_ids = [args.gpu_id]

    mmcv.mkdir_or_exist(osp.abspath(cfg.work_dir))
    cfg.dump(osp.join(cfg.work_dir, osp.basename(args.config)))
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    logger = get_root_logger(log_file=osp.join(cfg.work_dir, f'{timestamp}.log'),
                             log_level=cfg.log_level, name='mmdet')
    env_info = '\n'.join(f'{key}: {value}' for key, value in collect_env().items())
    logger.info('Environment info:\n%s', env_info)
    logger.info('Distributed training: %s', distributed)
    logger.info('Config:\n%s', cfg.pretty_text)
    set_random_seed(args.seed, deterministic=args.deterministic)
    cfg.seed = args.seed
    meta = dict(env_info=env_info, seed=args.seed, config=cfg.pretty_text,
                exp_name=osp.basename(args.config))
    model = build_model(cfg.model, train_cfg=cfg.get('train_cfg'), test_cfg=cfg.get('test_cfg'))
    if not (cfg.get('resume_from') or cfg.get('load_from')):
        model.init_weights()
    datasets = [build_dataset(cfg.data.train)]
    if cfg.checkpoint_config is not None:
        cfg.checkpoint_config.meta = dict(
            mmdet_version=mmdet_version, mmseg_version=mmseg_version,
            mmdet3d_version=mmdet3d_version, config=cfg.pretty_text,
            CLASSES=datasets[0].CLASSES,
            PALETTE=getattr(datasets[0], 'PALETTE', None))
    model.CLASSES = datasets[0].CLASSES
    train_model(model, datasets, cfg, distributed=distributed,
                validate=args.validate, timestamp=timestamp, meta=meta)
    if distributed and dist.is_initialized():
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
