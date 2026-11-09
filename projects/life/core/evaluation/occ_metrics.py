import numpy as np


class Metric_mIoU():
    def __init__(self, num_classes=18):
        self.class_names = ['others','barrier', 'bicycle', 'bus', 'car', 'construction_vehicle',
                            'motorcycle', 'pedestrian', 'traffic_cone', 'trailer', 'truck',
                            'driveable_surface', 'other_flat', 'sidewalk',
                            'terrain', 'manmade', 'vegetation','free']
        self.num_classes = num_classes

        self.hist = np.zeros((self.num_classes, self.num_classes))
        self.cnt = 0

    def hist_info(self, n_cl, pred, gt):
        """Build a confusion matrix with ground-truth rows and prediction columns."""
        assert pred.shape == gt.shape
        k = (gt >= 0) & (gt < n_cl)
        return np.bincount(
            n_cl * gt[k].astype(int) + pred[k].astype(int), minlength=n_cl ** 2
        ).reshape(n_cl, n_cl)

    def per_class_iu(self, hist):

        with np.errstate(divide='ignore', invalid='ignore'):
            return np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist))

    def add_batch(self, semantics_pred, semantics_gt, mask_camera):
        """Accumulate camera-visible voxels with valid ground-truth labels."""
        self.cnt += 1
        self.hist += self.hist_info(
            self.num_classes, semantics_pred[mask_camera].flatten(),
            semantics_gt[mask_camera].flatten())

    def count_miou(self):
        mIoU = self.per_class_iu(self.hist)
        present_classes = self.hist.sum(axis=1)[:self.num_classes-1] > 0
        mean_iou = float(np.nanmean(mIoU[:self.num_classes-1][present_classes]))
        print(f'===> per class IoU of {self.cnt} samples:')
        for ind_class in range(self.num_classes-1):
            print(f'===> {self.class_names[ind_class]} - IoU = ' + str(round(mIoU[ind_class] * 100, 2)))

        print(f'===> mIoU of {self.cnt} samples: ' + str(round(mean_iou * 100, 2)))

        eval_res = dict()
        eval_res['mIoU'] = mIoU
        eval_res['mean_iou'] = mean_iou
        return eval_res
