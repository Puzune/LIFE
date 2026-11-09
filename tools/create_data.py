# Copyright (c) OpenMMLab. All rights reserved.
"""Create nuScenes metadata with temporal and Occ3D occupancy annotations."""
import argparse
from pathlib import Path
import pickle

from _runtime import setup_import_path

setup_import_path()


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root-path', default='data/nuscenes', help='nuScenes data directory')
    parser.add_argument('--occ-root', help='Occ3D gts directory; defaults to ROOT_PATH/gts')
    parser.add_argument('--version', choices=['v1.0-trainval', 'v1.0-mini'], default='v1.0-trainval')
    parser.add_argument('--extra-tag', default='nuscenes', help='prefix for generated info files')
    parser.add_argument('--night-splits', action='store_true', help='also write nighttime train/validation metadata')
    return parser.parse_args()


def add_annotations(root_path, occ_root, version, info_prefix, night_splits=False):
    from nuscenes import NuScenes

    nusc = NuScenes(version=version, dataroot=str(root_path), verbose=True)
    for split in ('train', 'val'):
        info_path = root_path / f'{info_prefix}_infos_{split}.pkl'
        with info_path.open('rb') as stream:
            dataset = pickle.load(stream)
        night_infos = []
        for index, info in enumerate(dataset['infos']):
            sample = nusc.get('sample', info['token'])
            info['scene_token'] = sample['scene_token']
            scene = nusc.get('scene', sample['scene_token'])
            info['occ_path'] = str(occ_root / scene['name'] / info['token'])
            if 'night' in scene['description'].lower():
                night_infos.append(info)
            if index % 1000 == 0:
                print(f'{split}: {index}/{len(dataset["infos"])}')
        with info_path.open('wb') as stream:
            pickle.dump(dataset, stream, protocol=pickle.HIGHEST_PROTOCOL)
        if night_splits:
            night_path = root_path / f'{info_prefix}_infos_{split}_onlynight.pkl'
            night_dataset = dict(dataset, infos=night_infos)
            with night_path.open('wb') as stream:
                pickle.dump(night_dataset, stream, protocol=pickle.HIGHEST_PROTOCOL)
            print(f'{night_path}: {len(night_infos)} samples')


def main():
    args = parse_args()
    from tools.data_converter.nuscenes_converter import create_nuscenes_infos

    root_path = Path(args.root_path)
    occ_root = Path(args.occ_root) if args.occ_root else root_path / 'gts'
    create_nuscenes_infos(str(root_path), args.extra_tag, version=args.version)
    add_annotations(root_path, occ_root, args.version, args.extra_tag, args.night_splits)


if __name__ == '__main__':
    main()
