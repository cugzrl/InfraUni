"""TransFusion LiDAR tower: hard voxels, Sparse Encoder, SECOND, SECONDFPN."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from mmdet3d.models.backbones.second import SECOND
from mmdet3d.models.backbones.sparse_encoder import SparseEncoder
from mmdet3d.models.necks.second import SECONDFPN
from mmdet3d.ops import Voxelization

from ..data.constants import OUT_SIZE_FACTOR, PC_RANGE, VOXEL_SIZE


def sparse_shape_for(voxel_size, pc_range=PC_RANGE):
    """BEVFusion spconv wants [X, Y, Z+1]. Coords come out as x, y, z."""
    span = [pc_range[i + 3] - pc_range[i] for i in range(3)]
    grid = [int(round(span[i] / voxel_size[i])) for i in range(3)]
    return [grid[0], grid[1], grid[2] + 1], grid


class LidarBackbone(nn.Module):
    def __init__(self, voxel_size=None, max_voxels=(120000, 160000)):
        super().__init__()
        self.voxel_size = list(voxel_size or VOXEL_SIZE)
        self.sparse_shape, self.grid_size = sparse_shape_for(self.voxel_size)
        self.voxelize = Voxelization(
            voxel_size=self.voxel_size,
            point_cloud_range=PC_RANGE,
            max_num_points=10,
            max_voxels=max_voxels,
            deterministic=False,
        )
        self.sparse = SparseEncoder(
            in_channels=5,
            sparse_shape=self.sparse_shape,
            output_channels=128,
            order=("conv", "norm", "act"),
            encoder_channels=((16, 16, 32), (32, 32, 64), (64, 64, 128), (128, 128)),
            encoder_paddings=((0, 0, 1), (0, 0, 1), (0, 0, [1, 1, 0]), (0, 0)),
            block_type="basicblock",
        )
        self.second = SECOND(
            in_channels=256,
            out_channels=[128, 256],
            layer_nums=[5, 5],
            layer_strides=[1, 2],
            norm_cfg=dict(type="BN", eps=0.001, momentum=0.01),
            conv_cfg=dict(type="Conv2d", bias=False),
        )
        self.neck = SECONDFPN(
            in_channels=[128, 256],
            out_channels=[256, 256],
            upsample_strides=[1, 2],
            norm_cfg=dict(type="BN", eps=0.001, momentum=0.01),
            upsample_cfg=dict(type="deconv", bias=False),
            use_conv_for_no_stride=True,
        )
        self.out_size_factor = OUT_SIZE_FACTOR

    @torch.no_grad()
    def _voxelize(self, points):
        features, coords, sizes = [], [], []
        for index, cloud in enumerate(points):
            voxels, coors, num_points = self.voxelize(cloud)
            features.append(voxels.sum(dim=1) / num_points.type_as(voxels).view(-1, 1))
            coords.append(F.pad(coors, (1, 0), mode="constant", value=index))
            sizes.append(num_points)
        return torch.cat(features, 0).contiguous(), torch.cat(coords, 0)

    def forward(self, points):
        feats, coords = self._voxelize(points)
        batch = int(coords[-1, 0].item()) + 1
        x = self.sparse(feats, coords, batch)
        scales = self.second(x)
        bev = self.neck(scales)[0]
        return bev, scales
