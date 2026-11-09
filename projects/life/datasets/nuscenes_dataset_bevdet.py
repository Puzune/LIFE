# Copyright (c) OpenMMLab. All rights reserved.
import mmcv

from mmdet3d.datasets import DATASETS
from mmdet3d.datasets.custom_3d import Custom3DDataset


@DATASETS.register_module()
class NuScenesDatasetBEVDet(Custom3DDataset):
    """Temporal nuScenes camera data with calibrated ego-motion metadata."""

    CLASSES = ('others', 'barrier', 'bicycle', 'bus', 'car',
               'construction_vehicle', 'motorcycle', 'pedestrian',
               'traffic_cone', 'trailer', 'truck', 'driveable_surface',
               'other_flat', 'sidewalk', 'terrain', 'manmade', 'vegetation',
               'free')

    def __init__(self,
                 ann_file,
                 pipeline=None,
                 data_root=None,
                 classes=None,
                 load_interval=1,
                 modality=None,
                 box_type_3d='LiDAR',
                 filter_empty_gt=False,
                 test_mode=False,
                 multi_adj_frame_id_cfg=None,
                 stereo=False):
        self.load_interval = load_interval
        super().__init__(
            data_root=data_root,
            ann_file=ann_file,
            pipeline=pipeline,
            classes=classes,
            modality=modality,
            box_type_3d=box_type_3d,
            filter_empty_gt=filter_empty_gt,
            test_mode=test_mode)
        self.multi_adj_frame_id_cfg = multi_adj_frame_id_cfg
        self.stereo = stereo

    def load_annotations(self, ann_file):
        """Load metadata in timestamp order, optionally subsampling frames."""
        data = mmcv.load(ann_file, file_format='pkl')
        data_infos = list(sorted(data['infos'], key=lambda e: e['timestamp']))
        return data_infos[::self.load_interval]

    def get_data_info(self, index):
        info = self.data_infos[index]
        return dict(
            sample_idx=info['token'],
            pts_filename=info['lidar_path'],
            timestamp=info['timestamp'] / 1e6,
            curr=info,
            adjacent=self.get_adj_info(info, index),
        )

    def get_adj_info(self, info, index):
        adj_ids = list(range(*self.multi_adj_frame_id_cfg))
        if self.stereo:
            assert self.multi_adj_frame_id_cfg[0] == 1
            assert self.multi_adj_frame_id_cfg[2] == 1
            adj_ids.append(self.multi_adj_frame_id_cfg[1])
        adjacent = []
        for offset in adj_ids:
            previous = self.data_infos[max(index - offset, 0)]
            adjacent.append(previous if previous['scene_token'] == info['scene_token'] else info)
        return adjacent
