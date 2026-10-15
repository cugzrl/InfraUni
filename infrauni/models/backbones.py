"""LiDAR pillar BEV encoder and a light camera encoder."""

from __future__ import annotations

import torch
import torch.nn as nn

from infrauni.models.layers import ConvBNReLU


class PillarEncoder(nn.Module):
    """PointPillars-style BEV encoder without a sparse-conv dependency.

    This stands in for the LiDAR backbone used by TransFusion. The tensor
    layout is (B, C, Y, X), so row index follows y and column index follows x.
    """

    def __init__(self, pc_range, voxel_size, out_ch=64):
        super().__init__()
        self.pc_range = [float(v) for v in pc_range]
        self.vx = float(voxel_size[0])
        self.vy = float(voxel_size[1])
        self.grid_x = int(round((self.pc_range[3] - self.pc_range[0]) / self.vx))
        self.grid_y = int(round((self.pc_range[4] - self.pc_range[1]) / self.vy))
        self.mlp = nn.Sequential(
            nn.Linear(6, out_ch),
            nn.ReLU(inplace=True),
            nn.Linear(out_ch, out_ch),
            nn.ReLU(inplace=True),
        )
        self.out_ch = out_ch

    def forward(self, points, point_mask):
        batch, _, _ = points.shape
        device = points.device
        x = points[..., 0]
        y = points[..., 1]
        z = points[..., 2]
        xmin, ymin, zmin, xmax, ymax, zmax = self.pc_range
        ix = torch.floor((x - xmin) / self.vx).long()
        iy = torch.floor((y - ymin) / self.vy).long()
        valid = (
            point_mask
            & (ix >= 0)
            & (ix < self.grid_x)
            & (iy >= 0)
            & (iy < self.grid_y)
            & (z >= zmin)
            & (z <= zmax)
        )
        cx = (ix.clamp(0, self.grid_x - 1).float() + 0.5) * self.vx + xmin
        cy = (iy.clamp(0, self.grid_y - 1).float() + 0.5) * self.vy + ymin
        feat_in = torch.stack(
            [
                x / 51.2,
                y / 51.2,
                z / 4.0,
                points[..., 3],
                (x - cx) / self.vx,
                (y - cy) / self.vy,
            ],
            dim=-1,
        )
        feat = self.mlp(feat_in) * valid.unsqueeze(-1).to(feat_in.dtype)
        canvas = feat.new_zeros(batch, self.out_ch, self.grid_y * self.grid_x)
        flat = (iy.clamp(0, self.grid_y - 1) * self.grid_x + ix.clamp(0, self.grid_x - 1))
        flat = torch.where(valid, flat, torch.zeros_like(flat))
        index = flat[:, None, :].expand(batch, self.out_ch, flat.shape[1])
        src = feat.permute(0, 2, 1).contiguous()
        canvas.scatter_reduce_(2, index, src, reduce="amax", include_self=True)
        return canvas.view(batch, self.out_ch, self.grid_y, self.grid_x)


class BevNeck(nn.Module):
    """Full-resolution map for the heatmap, coarse tokens for attention."""

    def __init__(self, in_ch, d_model):
        super().__init__()
        self.stem = ConvBNReLU(in_ch, d_model, 3, 1)
        self.down = nn.Sequential(
            ConvBNReLU(d_model, d_model, 3, 2),
            ConvBNReLU(d_model, d_model, 3, 2),
        )

    def forward(self, bev):
        fine = self.stem(bev)
        coarse = self.down(fine)
        return fine, coarse


class ImageEncoder(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.net = nn.Sequential(
            ConvBNReLU(3, 32, 7, 2),
            ConvBNReLU(32, 64, 3, 2),
            ConvBNReLU(64, d_model, 3, 2),
            ConvBNReLU(d_model, d_model, 3, 2),
        )

    def forward(self, image):
        return self.net(image)
