# Copyright (c) OpenMMLab. All rights reserved.
"""Evaluate a LIFE checkpoint on semantic occupancy prediction."""
import argparse
import os

from _runtime import set_cuda_device, setup_import_path

setup_import_path()


def parse_args():
    from mmcv import DictAction

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('config', help='evaluation configuration')
    parser.add_argument('checkpoint', help='LIFE checkpoint')
    parser.add_argument('--out', help='optional pickle file for predictions')
    parser.add_argument('--eval', nargs='+', default=['mIoU'], choices=['mIoU', 'ray-iou'])
    parser.add_argument('--gpu-id', type=int, default=0, help='visible CUDA device for one process')
    parser.add_argument('--gpu-collect', action='store_true', help='collect distributed predictions on GPU')
    parser.add_argument('--tmpdir', help='temporary directory for distributed prediction collection')
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument('--deterministic', action='store_true', help='use deterministic cuDNN settings')
    parser.add_argument('--cfg-options', nargs='+', action=DictAction,
                        help='configuration overrides, e.g. data.test.ann_file=...')
    parser.add_argument('--launcher', choices=['none', 'pytorch'], default='none')
    parser.add_argument('--local-rank', '--local_rank', dest='local_rank', type=int, default=0)
    args = parser.parse_args()
    if args.out and not args.out.endswith(('.pkl', '.pickle')):
        parser.error('--out must end in .pkl or .pickle')
    return args


def main():
    args = parse_args()
    import mmcv
    import torch
    from mmcv import Config
    from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
    from mmcv.runner import get_dist_info, init_dist, load_checkpoint, wrap_fp16_model
    from mmcv.utils import import_modules_from_strings
    from mmdet.apis import multi_gpu_test, set_random_seed
    from mmdet.utils import compat_cfg, setup_multi_processes
    from mmdet3d.apis import single_gpu_test
    from mmdet3d.datasets import build_dataloader, build_dataset
    from mmdet3d.models import build_model
    from torch2_compat import patch_mmcv_scatter_for_torch2

    os.environ.setdefault('LOCAL_RANK', str(args.local_rank))
    set_cuda_device(args)
    patch_mmcv_scatter_for_torch2()
    cfg = Config.fromfile(args.config)
    if args.cfg_options:
        cfg.merge_from_dict(args.cfg_options)
    cfg = compat_cfg(cfg)
    setup_multi_processes(cfg)
    if cfg.get('custom_imports'):
        import_modules_from_strings(**cfg.custom_imports)
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True

    distributed = args.launcher != 'none'
    if distributed:
        init_dist(args.launcher, **cfg.dist_params)
    cfg.gpu_ids = [args.gpu_id]
    cfg.data.test.test_mode = True
    loader_cfg = dict(samples_per_gpu=1, workers_per_gpu=cfg.data.get('workers_per_gpu', 2),
                      dist=distributed, shuffle=False)
    loader_cfg.update(cfg.data.get('test_dataloader', {}))
    set_random_seed(args.seed, deterministic=args.deterministic)
    dataset = build_dataset(cfg.data.test)
    data_loader = build_dataloader(dataset, **loader_cfg)
    cfg.model.train_cfg = None
    model = build_model(cfg.model, test_cfg=cfg.get('test_cfg'))
    if cfg.get('fp16'):
        wrap_fp16_model(model)
    checkpoint = load_checkpoint(
        model, args.checkpoint, map_location='cpu', strict=True)
    model.CLASSES = checkpoint.get('meta', {}).get('CLASSES', dataset.CLASSES)
    if not distributed:
        model = MMDataParallel(model.cuda(args.gpu_id), device_ids=cfg.gpu_ids)
        outputs = single_gpu_test(model, data_loader)
    else:
        model = MMDistributedDataParallel(model.cuda(), device_ids=[torch.cuda.current_device()],
                                          broadcast_buffers=False)
        outputs = multi_gpu_test(model, data_loader, args.tmpdir, args.gpu_collect)
    rank, _ = get_dist_info()
    if rank == 0:
        if args.out:
            mmcv.dump(outputs, args.out)
        print(dataset.evaluate(outputs, metric=args.eval))
    if distributed and torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()


if __name__ == '__main__':
    main()
