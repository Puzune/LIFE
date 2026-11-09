import torch


def patch_mmcv_scatter_for_torch2():
    """Keep MMCV 1.x DataContainer scatter working with newer PyTorch."""
    import mmcv.parallel._functions as mmcv_functions
    from mmcv.parallel.distributed import MMDistributedDataParallel

    original_get_stream = mmcv_functions._get_stream
    if not getattr(original_get_stream, '_life_torch2_compatible', False):
        def get_stream(device):
            if isinstance(device, int):
                device = torch.device('cuda', device)
            return original_get_stream(device)

        get_stream._life_torch2_compatible = True
        mmcv_functions._get_stream = get_stream

    if getattr(MMDistributedDataParallel._run_ddp_forward,
               '_life_torch2_compatible', False):
        return

    def run_ddp_forward(self, *inputs, **kwargs):
        if self.device_ids:
            inputs, kwargs = self.to_kwargs(inputs, kwargs, self.device_ids[0])
            return self.module(*inputs[0], **kwargs[0])
        return self.module(*inputs, **kwargs)

    run_ddp_forward._life_torch2_compatible = True
    MMDistributedDataParallel._run_ddp_forward = run_ddp_forward
