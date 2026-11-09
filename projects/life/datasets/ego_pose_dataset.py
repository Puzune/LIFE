import torch
import numpy as np
from pyquaternion import Quaternion
from torch.utils.data import Dataset


def trans_matrix(T, R):
    tm = np.eye(4)
    tm[:3, :3] = R.rotation_matrix
    tm[:3, 3] = T
    return tm


class EgoPoseDataset(Dataset):
    def __init__(self, data_infos):
        super(EgoPoseDataset, self).__init__()

        self.data_infos = data_infos
        self.scene_frames = {}

        for info in data_infos:
            scene_token = info['scene_token']
            if scene_token not in self.scene_frames:
                self.scene_frames[scene_token] = []
            self.scene_frames[scene_token].append(info)

    def __len__(self):
        return len(self.data_infos)

    def get_ego_from_lidar(self, info):
        ego_from_lidar = trans_matrix(
            np.array(info['lidar2ego_translation']), 
            Quaternion(info['lidar2ego_rotation']))
        return ego_from_lidar

    def get_global_pose(self, info, inverse=False):
        global_from_ego = trans_matrix(
            np.array(info['ego2global_translation']), 
            Quaternion(info['ego2global_rotation']))
        ego_from_lidar = trans_matrix(
            np.array(info['lidar2ego_translation']), 
            Quaternion(info['lidar2ego_rotation']))
        pose = global_from_ego.dot(ego_from_lidar)
        if inverse:
            pose = np.linalg.inv(pose)
        return pose

    def __getitem__(self, idx):
        info = self.data_infos[idx]

        ref_sample_token = info['token']
        ref_lidar_from_global = self.get_global_pose(info, inverse=True)
        ref_ego_from_lidar = self.get_ego_from_lidar(info)

        scene_token = info['scene_token']
        scene_frame = self.scene_frames[scene_token]
        ref_index = scene_frame.index(info)

        output_origin_list = []
        for curr_index in range(len(scene_frame)):
            if curr_index == ref_index:
                origin_tf = np.array([0.0, 0.0, 0.0], dtype=np.float32)
            else:
                # Transform LiDAR origins into the reference LiDAR frame via global poses.
                global_from_curr = self.get_global_pose(scene_frame[curr_index], inverse=False)
                ref_from_curr = ref_lidar_from_global.dot(global_from_curr)
                origin_tf = np.array(ref_from_curr[:3, 3], dtype=np.float32)

            origin_tf_pad = np.ones([4])
            origin_tf_pad[:3] = origin_tf
            origin_tf = np.dot(ref_ego_from_lidar[:3], origin_tf_pad.T).T

            if np.abs(origin_tf[0]) < 39 and np.abs(origin_tf[1]) < 39:
                output_origin_list.append(origin_tf)
        
        if len(output_origin_list) > 8:
            select_idx = np.round(np.linspace(0, len(output_origin_list) - 1, 8)).astype(np.int64)
            output_origin_list = [output_origin_list[i] for i in select_idx]

        output_origin_tensor = torch.from_numpy(np.stack(output_origin_list))

        return (ref_sample_token, output_origin_tensor)
