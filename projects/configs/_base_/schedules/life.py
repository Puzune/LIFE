optimizer = dict(type='AdamW', lr=0.0002, weight_decay=0.01)
optimizer_config = dict(grad_clip=dict(max_norm=5, norm_type=2))
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=200,
    warmup_ratio=0.001,
    step=[16],
)
runner = dict(type='EpochBasedRunner', max_epochs=20)
custom_hooks = [
    dict(type='MEGVIIEMAHook', init_updates=10560, priority='NORMAL'),
]
checkpoint_config = dict(interval=1, max_keep_ckpts=20)
evaluation = dict(interval=1, start=20, metric=['mIoU'])
