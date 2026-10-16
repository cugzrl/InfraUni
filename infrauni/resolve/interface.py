"""Boundary between OpenPCDet TransFusion-L and the DeTra decoder.

The detection baseline keeps the official head. A later LiDAR-only joint model
reads the tensors below and does not load an MMDetection3D checkpoint into this
head, or an OpenPCDet checkpoint into the nuScenes camera model.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class LidarCandidates:
    """One frame of initial queries in the rotated RESOLVE lidar frame.

    ``bev`` is ``spatial_features_2d`` from the official backbone, ``[B, C, H, W]``.
    ``query`` is the decoder feature ``[B, C, N]``. The official ``predict``
    computes it and then returns only the box heads, so a joint adapter has to
    keep that tensor. ``boxes`` is ``[B, N, 9]`` as x, y, z, l, w, h, yaw, vx, vy.
    """

    bev: torch.Tensor
    query: torch.Tensor
    boxes: torch.Tensor
    heatmap_score: torch.Tensor
    point_cloud_range: tuple
    voxel_size: tuple


FORECAST_SECONDS = 3.0
FORECAST_STEPS = 6
FORECAST_HZ = 2.0
