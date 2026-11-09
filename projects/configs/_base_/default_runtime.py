log_config = dict(
    interval=50,
    hooks=[dict(type='TextLoggerHook'), dict(type='TensorboardLoggerHook')],
)
dist_params = dict(backend='nccl')
log_level = 'INFO'
workflow = [('train', 1)]
seed = 0
opencv_num_threads = 0
mp_start_method = 'fork'
resume_from = None
