# Training and Evaluation

## Training

```bash
# Day-and-night training
bash tools/dist_train.sh projects/configs/life.py 4

# Night-only training
bash tools/dist_train.sh projects/configs/life.py 4 \
  --work-dir outputs/life_night \
  --cfg-options data.train.ann_file=data/nuscenes/nuscenes_infos_train_onlynight.pkl \
  data.val.ann_file=data/nuscenes/nuscenes_infos_val_onlynight.pkl
```

## Evaluation

```bash
# Full validation
bash tools/dist_test.sh projects/configs/life.py ckpts/LIFE_FULL.pth 1 --eval mIoU

# Night validation
bash tools/dist_test.sh projects/configs/life.py ckpts/LIFE_FULL.pth 1 --eval mIoU \
  --cfg-options data.test.ann_file=data/nuscenes/nuscenes_infos_val_onlynight.pkl

# Night-only model
bash tools/dist_test.sh projects/configs/life.py ckpts/LIFE_NIGHT.pth 1 --eval mIoU \
  --cfg-options data.test.ann_file=data/nuscenes/nuscenes_infos_val_onlynight.pkl
```

Use EMA weights for evaluation. For RayIoU, use `--eval ray-iou`.
