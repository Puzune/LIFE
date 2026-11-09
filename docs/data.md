# Data Preparation

Download [nuScenes v1.0-trainval](https://www.nuscenes.org/nuscenes) and
[Occ3D-nuScenes](https://github.com/Tsinghua-MARS-Lab/Occ3D) occupancy labels.
Arrange the files as shown below.

```text
data/nuscenes/
├── v1.0-trainval/
├── maps/
├── samples/
└── gts/
    └── scene-XXXX/
        └── SAMPLE_TOKEN/
            └── labels.npz
```

Generate the metadata.

```bash
python tools/create_data.py --root-path data/nuscenes --version v1.0-trainval --night-splits
```

This creates `nuscenes_infos_train.pkl`, `nuscenes_infos_val.pkl`, and their
`*_onlynight.pkl` counterparts under `data/nuscenes/`. Night subsets use scenes
whose descriptions contain `night`.
