# Copyright (c) OpenMMLab. All rights reserved.
from os import path as osp

import mmcv
import numpy as np


def create_nuscenes_infos(root_path,
                          info_prefix,
                          version='v1.0-trainval'):
    """Write nuScenes train/validation metadata to pickle files."""
    from nuscenes.nuscenes import NuScenes
    nusc = NuScenes(version=version, dataroot=root_path, verbose=True)
    from nuscenes.utils import splits
    if version == 'v1.0-trainval':
        train_scenes = splits.train
        val_scenes = splits.val
    elif version == 'v1.0-mini':
        train_scenes = splits.mini_train
        val_scenes = splits.mini_val
    else:
        raise ValueError(f'Unsupported nuScenes version: {version}')

    available_scenes = {
        scene['name']: scene['token'] for scene in get_available_scenes(nusc)
    }
    train_scenes = {
        available_scenes[name] for name in train_scenes if name in available_scenes
    }
    val_scenes = {
        available_scenes[name] for name in val_scenes if name in available_scenes
    }

    print('train scene: {}, val scene: {}'.format(
        len(train_scenes), len(val_scenes)))
    train_nusc_infos, val_nusc_infos = _fill_trainval_infos(
        nusc, train_scenes, val_scenes)

    metadata = dict(version=version)
    print('train sample: {}, val sample: {}'.format(
        len(train_nusc_infos), len(val_nusc_infos)))
    data = dict(infos=train_nusc_infos, metadata=metadata)
    info_path = osp.join(root_path, '{}_infos_train.pkl'.format(info_prefix))
    mmcv.dump(data, info_path)
    data['infos'] = val_nusc_infos
    info_val_path = osp.join(root_path, '{}_infos_val.pkl'.format(info_prefix))
    mmcv.dump(data, info_val_path)


def get_available_scenes(nusc):
    """Return scenes whose first LiDAR sample is available on disk."""
    available_scenes = []
    print('total scene num: {}'.format(len(nusc.scene)))
    for scene in nusc.scene:
        sample_rec = nusc.get('sample', scene['first_sample_token'])
        lidar_path = str(nusc.get_sample_data_path(sample_rec['data']['LIDAR_TOP']))
        if not osp.isfile(lidar_path):
            continue
        available_scenes.append(scene)
    print('exist scene num: {}'.format(len(available_scenes)))
    return available_scenes


def _fill_trainval_infos(nusc, train_scenes, val_scenes):
    """Collect LiDAR and camera metadata, split by training scene tokens."""
    train_nusc_infos = []
    val_nusc_infos = []
    available_scene_tokens = train_scenes | val_scenes

    for sample in mmcv.track_iter_progress(nusc.sample):
        if sample['scene_token'] not in available_scene_tokens:
            continue
        lidar_token = sample['data']['LIDAR_TOP']
        sd_rec = nusc.get('sample_data', sample['data']['LIDAR_TOP'])
        cs_record = nusc.get('calibrated_sensor',
                             sd_rec['calibrated_sensor_token'])
        pose_record = nusc.get('ego_pose', sd_rec['ego_pose_token'])
        lidar_path = nusc.get_sample_data_path(lidar_token)

        mmcv.check_file_exist(lidar_path)

        info = {
            'lidar_path': lidar_path,
            'token': sample['token'],
            'cams': dict(),
            'lidar2ego_translation': cs_record['translation'],
            'lidar2ego_rotation': cs_record['rotation'],
            'ego2global_translation': pose_record['translation'],
            'ego2global_rotation': pose_record['rotation'],
            'timestamp': sample['timestamp'],
        }

        camera_types = [
            'CAM_FRONT',
            'CAM_FRONT_RIGHT',
            'CAM_FRONT_LEFT',
            'CAM_BACK',
            'CAM_BACK_LEFT',
            'CAM_BACK_RIGHT',
        ]
        for cam in camera_types:
            cam_token = sample['data'][cam]
            info['cams'][cam] = obtain_camera_info(nusc, cam_token)

        if sample['scene_token'] in train_scenes:
            train_nusc_infos.append(info)
        else:
            val_nusc_infos.append(info)

    return train_nusc_infos, val_nusc_infos


def obtain_camera_info(nusc, sensor_token):
    """Read the camera image, intrinsics and capture-time ego/global poses."""
    sd_rec = nusc.get('sample_data', sensor_token)
    cs_record = nusc.get('calibrated_sensor', sd_rec['calibrated_sensor_token'])
    pose_record = nusc.get('ego_pose', sd_rec['ego_pose_token'])
    data_path = str(nusc.get_sample_data_path(sd_rec['token']))
    return {
        'data_path': data_path,
        'cam_intrinsic': np.array(cs_record['camera_intrinsic']),
        'sensor2ego_translation': cs_record['translation'],
        'sensor2ego_rotation': cs_record['rotation'],
        'ego2global_translation': pose_record['translation'],
        'ego2global_rotation': pose_record['rotation'],
    }
