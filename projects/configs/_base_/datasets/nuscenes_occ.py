data_config = dict(
    cams=[
        'CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_BACK_LEFT',
        'CAM_BACK', 'CAM_BACK_RIGHT',
    ],
    Ncams=6,
    input_size=(256, 704),
    resize=(-0.06, 0.11),
    rot=(-5.4, 5.4),
    flip=True,
    crop_h=(0.0, 0.0),
    resize_test=0.0,
)
grid_config = dict(
    x=[-40, 40, 0.4],
    y=[-40, 40, 0.4],
    z=[-1, 5.4, 6.4],
    depth=[1.0, 45.0, 0.5],
)
multi_adj_frame_id_cfg = (1, 2, 1)
bda_aug_conf = dict(
    rot_lim=(-0.0, 0.0),
    scale_lim=(1.0, 1.0),
    flip_dx_ratio=0.5,
    flip_dy_ratio=0.5,
)

train_pipeline = [
    dict(
        type='PrepareImageInputsWithIllumination',
        is_train=True,
        data_config=data_config,
        sequential=True,
    ),
    dict(
        type='BEVAugmentation',
        bda_aug_conf=bda_aug_conf,
        is_train=True,
    ),
    dict(type='LoadOccGTFromFile'),
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=5,
        use_dim=5,
        file_client_args=dict(backend='disk'),
    ),
    dict(type='PointToMultiViewDepth', downsample=1, grid_config=grid_config),
    dict(
        type='Collect3D',
        keys=[
            'img_inputs', 'gt_depth', 'voxel_semantics',
            'mask_camera', 'img_gray',
        ],
    ),
]
test_pipeline = [
    dict(
        type='PrepareImageInputsWithIllumination',
        data_config=data_config,
        sequential=True,
    ),
    dict(
        type='BEVAugmentation',
        bda_aug_conf=bda_aug_conf,
        is_train=False,
    ),
    dict(
        type='MultiScaleFlipAug3D',
        img_scale=(1333, 800),
        pts_scale_ratio=1,
        flip=False,
        transforms=[
            dict(type='Collect3D', keys=['img_inputs', 'img_gray']),
        ],
    ),
]

input_modality = dict(
    use_lidar=False,
    use_camera=True,
    use_radar=False,
    use_map=False,
    use_external=False,
)
dataset_type = 'NuScenesOccupancyDataset'
data_root = 'data/nuscenes/'
shared_dataset = dict(
    type=dataset_type,
    data_root=data_root,
    modality=input_modality,
    stereo=True,
    filter_empty_gt=False,
    multi_adj_frame_id_cfg=multi_adj_frame_id_cfg,
    box_type_3d='LiDAR',
)
data = dict(
    samples_per_gpu=8,
    workers_per_gpu=7,
    pin_memory=True,
    train=dict(
        **shared_dataset,
        ann_file=data_root + 'nuscenes_infos_train.pkl',
        pipeline=train_pipeline,
        test_mode=False,
    ),
    val=dict(
        **shared_dataset,
        ann_file=data_root + 'nuscenes_infos_val.pkl',
        pipeline=test_pipeline,
        test_mode=True,
    ),
    test=dict(
        **shared_dataset,
        ann_file=data_root + 'nuscenes_infos_val.pkl',
        pipeline=test_pipeline,
        test_mode=True,
    ),
)
