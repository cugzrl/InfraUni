"""ImageNet ResNet-50 and an FPN over all six nuScenes cameras.

The released TransFusion fusion checkpoint is not on disk, so the ResNet stays
at ImageNet initialization and the FPN is trained. Freezing a random FPN would
drop the camera features.
"""

from __future__ import annotations

import torch
from torch import nn

from mmdet.models.backbones.resnet import ResNet
from mmdet.models.necks.fpn import FPN

from ..data.constants import RESNET50_WEIGHTS


class CameraBackbone(nn.Module):
    def __init__(self, pretrained=RESNET50_WEIGHTS):
        super().__init__()
        self.backbone = ResNet(
            depth=50,
            num_stages=4,
            out_indices=(0, 1, 2, 3),
            frozen_stages=1,
            norm_cfg=dict(type="BN", requires_grad=False),
            norm_eval=True,
            style="pytorch",
            init_cfg=dict(type="Pretrained", checkpoint=pretrained),
        )
        self.neck = FPN(in_channels=[256, 512, 1024, 2048], out_channels=256, num_outs=5)
        for param in self.backbone.parameters():
            param.requires_grad = False

    def train(self, mode=True):
        super().train(mode)
        self.backbone.eval()
        return self

    def forward(self, images):
        # images: [B, V, 3, H, W]
        batch, views, channels, height, width = images.shape
        flat = images.reshape(batch * views, channels, height, width)
        feats = self.backbone(flat)
        return self.neck(feats)
