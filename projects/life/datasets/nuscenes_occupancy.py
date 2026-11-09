# Copyright (c) OpenMMLab. All rights reserved.
import os
import numpy as np
from tqdm import tqdm

from mmdet3d.datasets import DATASETS
from .nuscenes_dataset_bevdet import NuScenesDatasetBEVDet
from ..core.evaluation.occ_metrics import Metric_mIoU
from .ego_pose_dataset import EgoPoseDataset
from ..core.evaluation.ray_metrics import main as calc_rayiou
from torch.utils.data import DataLoader


@DATASETS.register_module()
class NuScenesOccupancyDataset(NuScenesDatasetBEVDet):
    def get_data_info(self, index):
        input_dict = super(NuScenesOccupancyDataset, self).get_data_info(index)
        input_dict['occ_gt_path'] = self.data_infos[index]['occ_path']
        return input_dict

    def evaluate(self, occ_results, metric='mIoU', **eval_kwargs):
        if isinstance(metric, str):
            metrics = [metric]
        else:
            metrics = list(metric)
        unsupported_metrics = set(metrics) - {'mIoU', 'ray-iou'}
        if unsupported_metrics:
            raise ValueError(
                f'Unsupported occupancy metrics: {sorted(unsupported_metrics)}')

        evaluate_miou = 'mIoU' in metrics
        evaluate_rayiou = 'ray-iou' in metrics
        eval_results = {}

        if evaluate_miou:
            self.occ_eval_metrics = Metric_mIoU(num_classes=18)

            print('\nStarting mIoU Evaluation...')
            for index, occ_pred in enumerate(tqdm(occ_results)):
                info = self.data_infos[index]
                occ_gt = np.load(os.path.join(info['occ_path'], 'labels.npz'))
                gt_semantics = occ_gt['semantics']
                mask_camera = occ_gt['mask_camera'].astype(bool)
                self.occ_eval_metrics.add_batch(
                    occ_pred,
                    gt_semantics,
                    mask_camera
                )

            miou_results = self.occ_eval_metrics.count_miou()
            eval_results['mIoU'] = miou_results['mean_iou']

        if evaluate_rayiou:
            occ_gts = []
            occ_preds = []
            lidar_origins = []
            
            print('\nStarting RayIoU Evaluation...')

            data_loader = DataLoader(
                EgoPoseDataset(self.data_infos),
                batch_size=1,
                shuffle=False,
                num_workers=8
            )

            for data_id, (_, output_origin) in enumerate(data_loader):
                info = self.data_infos[data_id]
                occ_gt = np.load(os.path.join(info['occ_path'], 'labels.npz'))
                gt_semantics = occ_gt['semantics']
                occ_pred = occ_results[data_id]

                lidar_origins.append(output_origin)
                occ_gts.append(gt_semantics)
                occ_preds.append(occ_pred)

            eval_results.update(calc_rayiou(occ_preds, occ_gts, lidar_origins))

        return eval_results
