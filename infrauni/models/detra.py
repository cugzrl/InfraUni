"""DeTra-style trajectory refinement on TransFusion object queries.

Each object query carries F motion modes and T = 1 + future steps. Time 0 is
the detection center. Later times are waypoints offset from that center, so a
forecast loss moves the box that will be reported.

Blocks follow DeTra: object / time / mode attention, position-guided sampling,
then a stop-grad pose update. LiDAR sampling uses deformable attention on the
two SECOND scales. Camera sampling projects the waypoint into the six views.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from mmcv.ops.multi_scale_deform_attn import MultiScaleDeformableAttention
from torch import nn

from ..data.constants import NUM_FUTURE, NUM_MODES, PC_RANGE


class FactorizedAttention(nn.Module):
    def __init__(self, dim, heads=4):
        super().__init__()
        self.obj = nn.MultiheadAttention(dim, heads, dropout=0.1, batch_first=True)
        self.time = nn.MultiheadAttention(dim, heads, dropout=0.1, batch_first=True)
        self.mode = nn.MultiheadAttention(dim, heads, dropout=0.1, batch_first=True)
        self.norm_obj = nn.LayerNorm(dim)
        self.norm_time = nn.LayerNorm(dim)
        self.norm_mode = nn.LayerNorm(dim)
        self.norm_ffn = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(nn.Linear(dim, dim * 4), nn.ReLU(inplace=True), nn.Linear(dim * 4, dim))

    def forward(self, query):
        batch, agents, modes, steps, dim = query.shape
        x = query.permute(0, 2, 3, 1, 4).reshape(batch * modes * steps, agents, dim)
        x = x + self.obj(self.norm_obj(x), self.norm_obj(x), self.norm_obj(x), need_weights=False)[0]
        x = x.reshape(batch, modes, steps, agents, dim).permute(0, 3, 1, 2, 4)
        x = x.reshape(batch * agents * modes, steps, dim)
        x = x + self.time(self.norm_time(x), self.norm_time(x), self.norm_time(x), need_weights=False)[0]
        x = x.reshape(batch, agents, modes, steps, dim).permute(0, 1, 3, 2, 4)
        x = x.reshape(batch * agents * steps, modes, dim)
        x = x + self.mode(self.norm_mode(x), self.norm_mode(x), self.norm_mode(x), need_weights=False)[0]
        x = x.reshape(batch, agents, steps, modes, dim).permute(0, 1, 3, 2, 4).contiguous()
        return x + self.ffn(self.norm_ffn(x))


class RefineBlock(nn.Module):
    def __init__(self, dim, lidar_channels, future_steps):
        super().__init__()
        self.future_steps = future_steps
        self.attn = FactorizedAttention(dim)
        self.lidar_proj = nn.ModuleList(nn.Conv2d(channels, dim, 1) for channels in lidar_channels)
        self.deform = MultiScaleDeformableAttention(
            embed_dims=dim,
            num_heads=4,
            num_levels=len(lidar_channels),
            num_points=4,
            im2col_step=1,
            batch_first=True,
        )
        self.cam_proj = nn.ModuleList(nn.Conv2d(256, dim, 1) for _ in range(2))
        self.sample_fuse = nn.Sequential(nn.Linear(dim * 2, dim), nn.ReLU(inplace=True), nn.Linear(dim, dim))
        self.center_head = nn.Sequential(nn.Linear(dim, dim), nn.ReLU(inplace=True), nn.Linear(dim, 2))
        self.offset_head = nn.Sequential(nn.Linear(dim, dim), nn.ReLU(inplace=True), nn.Linear(dim, 2))
        self.mode_head = nn.Linear(dim, 1)

    def forward(self, query, pose, lidar_scales, cam_feats, metas, height):
        query = self.attn(query)
        lidar_feat = self._sample_lidar(query, pose, lidar_scales)
        cam_feat = self._sample_camera(pose, height, cam_feats, metas)
        query = query + self.sample_fuse(torch.cat([lidar_feat, cam_feat], dim=-1)).view_as(query)

        base = pose.detach()
        center_in = query[:, :, :, 0].mean(dim=2)
        center_delta = self.center_head(center_in)
        center = base[:, :, 0, 0, :] + center_delta
        offsets = self.offset_head(query[:, :, :, 1:])
        future = center[:, :, None, None, :] + offsets
        center_steps = center[:, :, None, None, :].expand_as(base[:, :, :, :1, :])
        pose = torch.cat([center_steps, future], dim=3)
        logits = self.mode_head(query[:, :, :, 1:].mean(dim=3)).squeeze(-1)
        return query, pose, logits, center

    def _sample_lidar(self, query, pose, lidar_scales):
        batch, agents, modes, steps, dim = query.shape
        projected = []
        shapes = []
        for level, feat in enumerate(lidar_scales):
            projected.append(self.lidar_proj[level](feat))
            shapes.append(feat.shape[-2:])
        value = torch.cat([item.flatten(2).transpose(1, 2) for item in projected], dim=1)
        spatial = query.new_tensor(shapes, dtype=torch.long)
        starts = torch.cat([spatial.new_zeros(1), spatial.prod(-1).cumsum(0)[:-1]])
        # DeTra samples LiDAR around the current pose. Share it across modes.
        reference = self._normalize_xy(pose[:, :, 0, 0, :])
        reference = reference[:, :, None, :].expand(batch, agents, len(shapes), 2)
        sampled = self.deform(
            query=query[:, :, 0, 0],
            value=value,
            reference_points=reference,
            spatial_shapes=spatial,
            level_start_index=starts,
        )
        return sampled[:, :, None, None, :].expand(batch, agents, modes, steps, dim)

    def _sample_camera(self, pose, height, cam_feats, metas):
        batch, agents, modes, steps, _ = pose.shape
        z = height[:, :, None, None, None].expand(batch, agents, modes, steps, 1)
        points = torch.cat([pose, z], dim=-1).reshape(batch, agents * modes * steps, 3)
        dim = self.cam_proj[0].out_channels
        sampled = pose.new_zeros(batch, points.shape[1], dim)
        for b in range(batch):
            sampled[b] = self._sample_camera_one(points[b], cam_feats, metas[b], b)
        return sampled.view(batch, agents, modes, steps, dim)

    def _sample_camera_one(self, points, cam_feats, meta, batch_index):
        from mmdet3d.models.fusion_layers import apply_3d_transformation

        ones = torch.ones(points.shape[0], 1, device=points.device, dtype=points.dtype)
        restored = apply_3d_transformation(points, "LIDAR", meta, reverse=True)
        homo = torch.cat([restored, ones], dim=-1)
        lidar2img = points.new_tensor(meta["lidar2img"])
        scale = points.new_tensor(meta["scale_factor"][:2])
        pad_h, pad_w = meta["input_shape"][:2]
        acc = points.new_zeros(points.shape[0], self.cam_proj[0].out_channels)
        weight = points.new_zeros(points.shape[0], 1)
        for view in range(lidar2img.shape[0]):
            uvz = homo @ lidar2img[view].t()
            depth = uvz[:, 2]
            uv = uvz[:, :2] / depth.clamp(min=1e-5).unsqueeze(-1) * scale
            valid = (depth > 0) & (uv[:, 0] >= 0) & (uv[:, 0] < pad_w) & (uv[:, 1] >= 0) & (uv[:, 1] < pad_h)
            if int(valid.sum()) == 0:
                continue
            view_feat = 0
            for level in range(2):
                feat = self.cam_proj[level](cam_feats[level][batch_index * 6 + view].unsqueeze(0))
                stride = 4 * (2 ** level)
                gx = uv[:, 0] / stride / feat.shape[-1] * 2 - 1
                gy = uv[:, 1] / stride / feat.shape[-2] * 2 - 1
                grid = torch.stack([gx, gy], dim=-1).view(1, 1, -1, 2)
                view_feat = view_feat + F.grid_sample(feat, grid, align_corners=False).squeeze(2).squeeze(0).t()
            acc = acc + (view_feat / 2) * valid.float().unsqueeze(-1)
            weight = weight + valid.float().unsqueeze(-1)
        return acc / weight.clamp(min=1.0)

    @staticmethod
    def _normalize_xy(xy):
        x0, y0, _, x1, y1, _ = PC_RANGE
        norm = torch.stack([(xy[..., 0] - x0) / (x1 - x0), (xy[..., 1] - y0) / (y1 - y0)], dim=-1)
        return norm.clamp(0.0, 1.0)


class DetraDecoder(nn.Module):
    def __init__(self, dim=128, blocks=2, modes=NUM_MODES, future_steps=NUM_FUTURE, lidar_channels=(128, 256)):
        super().__init__()
        self.modes = modes
        self.future_steps = future_steps
        self.steps = future_steps + 1
        self.mode_embed = nn.Parameter(torch.randn(modes, dim) * 0.02)
        self.time_embed = nn.Parameter(torch.randn(self.steps, dim) * 0.02)
        self.blocks = nn.ModuleList(
            RefineBlock(dim, lidar_channels, future_steps) for _ in range(blocks)
        )
        self.executed_blocks = 0

    def forward(self, query, boxes, lidar_scales, cam_feats, metas):
        # query [B, C, N], boxes [N, >=7] of the single batch item.
        batch, channels, agents = query.shape
        if boxes.dim() == 2:
            boxes = boxes.unsqueeze(0)
        xy = boxes[..., :2]
        height = boxes[..., 2] + boxes[..., 5] * 0.5
        pose = xy[:, :, None, None, :].expand(batch, agents, self.modes, self.steps, 2).contiguous()
        token = query.permute(0, 2, 1)
        token = (
            token[:, :, None, None, :]
            + self.mode_embed[None, None, :, None, :]
            + self.time_embed[None, None, None, :, :]
        )
        poses, logits, centers = [], [], []
        self.executed_blocks = 0
        for block in self.blocks:
            token, pose, mode_logit, center = block(token, pose, lidar_scales, cam_feats, metas, height)
            poses.append(pose)
            logits.append(mode_logit)
            centers.append(center)
            self.executed_blocks += 1
        return {"poses": poses, "logits": logits, "centers": centers}
