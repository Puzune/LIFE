# Installation

**1. Install [PyTorch and torchvision](https://pytorch.org/get-started/locally/) in a Python 3.10 environment.**

**2. Install dependencies.**

```bash
pip install -r requirements.txt
pip install mmcv-full==1.7.0 mmdet==2.28.2 mmsegmentation==0.30.0
```

**3. Build LIFE.**

```bash
mkdir -p third_party
git clone --branch v1.0.0rc4 --depth 1 \
  https://github.com/open-mmlab/mmdetection3d.git third_party/mmdetection3d
pip install --no-deps --no-build-isolation -e third_party/mmdetection3d

git clone https://github.com/facebookresearch/dinov3.git third_party/dinov3
git -C third_party/dinov3 checkout 54694f7627fd815f62a5dcc82944ffa6153bbb76
pip install --no-deps --no-build-isolation -e projects
```

Download the [DINOv3 ViT-S+/16 weights](https://ai.meta.com/resources/models-and-libraries/dinov3-downloads/)
to `ckpts/dinov3_vits16plus_pretrain_lvd1689m-4057cbaa.pth`.
