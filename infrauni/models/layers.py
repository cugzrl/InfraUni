"""Small blocks shared by the joint network."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBNReLU(nn.Module):
    def __init__(self, in_ch, out_ch, kernel, stride):
        super().__init__()
        padding = kernel // 2
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel, stride=stride, padding=padding, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class AttnFFN(nn.Module):
    """Pre-norm self-attention or cross-attention plus an FFN."""

    def __init__(self, d_model, nhead, dropout, cross: bool):
        super().__init__()
        self.cross = cross
        self.attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
        )
        self.drop = nn.Dropout(dropout)

    def forward(self, query, context=None):
        h = self.norm1(query)
        kv = context if self.cross else h
        attn_out, _ = self.attn(h, kv, kv, need_weights=False)
        query = query + self.drop(attn_out)
        query = query + self.ffn(self.norm2(query))
        return query


def sample_feature_map(feat, xy, pc_range):
    """Bilinear-sample a BEV map at xy locations.

    feat: (B, C, H, W), xy: (B, ..., 2) in the LiDAR frame.
    Returns features with the same leading shape as xy and channel last.
    """
    batch, channels, _, _ = feat.shape
    xmin, ymin, xmax, ymax = pc_range[0], pc_range[1], pc_range[3], pc_range[4]
    u = (xy[..., 0] - xmin) / (xmax - xmin) * 2.0 - 1.0
    v = (xy[..., 1] - ymin) / (ymax - ymin) * 2.0 - 1.0
    grid = torch.stack([u, v], dim=-1)
    lead = grid.shape[1:-1]
    count = 1
    for size in lead:
        count *= int(size)
    grid = grid.reshape(batch, count, 1, 2)
    sampled = F.grid_sample(
        feat, grid, mode="bilinear", padding_mode="zeros", align_corners=True
    )
    sampled = sampled.squeeze(-1).permute(0, 2, 1).reshape(batch, *lead, channels)
    return sampled


def project_lidar_to_image(xyz, lidar2cam, K):
    """Project (B, N, 3) LiDAR points to pixel coordinates.

    Returns uv (B, N, 2) and a valid mask (points in front of the camera).
    """
    ones = torch.ones(*xyz.shape[:-1], 1, device=xyz.device, dtype=xyz.dtype)
    homo = torch.cat([xyz, ones], dim=-1)
    cam = torch.einsum("bij,bnj->bni", lidar2cam, homo)
    depth = cam[..., 2]
    valid = depth > 0.1
    pix = torch.einsum("bij,bnj->bni", K, cam[..., :3])
    uv = pix[..., :2] / depth.unsqueeze(-1).clamp(min=1e-4)
    return uv, valid


def decode_boxes(raw, pc_range):
    """Map unconstrained head outputs to x,y,z,w,l,h,yaw."""
    xmin, ymin, xmax, ymax = pc_range[0], pc_range[1], pc_range[3], pc_range[4]
    xy = raw[..., 0:2].sigmoid()
    x = xy[..., 0] * (xmax - xmin) + xmin
    y = xy[..., 1] * (ymax - ymin) + ymin
    z = raw[..., 2].tanh() * 4.0
    size = F.softplus(raw[..., 3:6]) + 0.05
    yaw = torch.atan2(raw[..., 6], raw[..., 7])
    return torch.stack([x, y, z, size[..., 0], size[..., 1], size[..., 2], yaw], dim=-1)


class BoxHead(nn.Module):
    def __init__(self, d_model, num_classes):
        super().__init__()
        self.cls = nn.Linear(d_model, num_classes)
        self.box = nn.Linear(d_model, 8)
        nn.init.constant_(self.cls.bias, -2.0)
        nn.init.constant_(self.box.bias[3:6], 0.5)

    def forward(self, query, pc_range):
        return self.cls(query), decode_boxes(self.box(query), pc_range)
