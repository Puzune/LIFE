_base_ = [
    '_base_/models/life.py',
    '_base_/datasets/nuscenes_occ.py',
    '_base_/schedules/life.py',
    '_base_/default_runtime.py',
]

custom_imports = dict(imports=['projects.life'], allow_failed_imports=False)

model = dict(
    img_view_transformer=dict(
        grid_config={{_base_.grid_config}},
        input_size={{_base_.data_config.input_size}},
        backward_projection=dict(
            transformer=dict(
                encoder=dict(data_config={{_base_.data_config}}))),
    ),
)

evaluation = dict(pipeline={{_base_.test_pipeline}})
load_from = None
work_dir = 'outputs/life'
