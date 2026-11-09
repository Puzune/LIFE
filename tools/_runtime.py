"""Shared path and device setup for LIFE command-line tools."""
import os
import sys
from pathlib import Path


def setup_import_path():
    """Prefer the checked-out MMDetection3D dependency and project plugin."""
    root = Path(__file__).resolve().parents[1]
    for path in (root / 'third_party' / 'mmdetection3d', root):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))


def set_cuda_device(args):
    """Respect visible devices and bind torchrun workers to their local rank."""
    import torch

    device = (int(os.environ.get('LOCAL_RANK', args.local_rank))
              if args.launcher == 'pytorch' else args.gpu_id)
    count = torch.cuda.device_count()
    if not count:
        raise RuntimeError('LIFE training and evaluation require a CUDA device.')
    if not 0 <= device < count:
        raise ValueError(f'CUDA device {device} is outside the {count} visible devices.')
    torch.cuda.set_device(device)
    return device
