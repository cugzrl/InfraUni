"""TransFusion-style queries with a DeTra-style trajectory refiner.

Data flow
    1. Pillar BEV + camera CNN.
    2. Heatmap top-k initializes object queries (TransFusion proposal).
    3. Queries cross-attend to LiDAR BEV and then to the image.
    4. Each detection query is expanded over modes and future times,
       anchored on the detected xy, and refined (DeTra).
    5. Pooled future-query features are written back into the detection
       query so motion can update the box (bidirectional path).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from infrauni.models.backbones import BevNeck, ImageEncoder, PillarEncoder
from infrauni.models.layers import AttnFFN, BoxHead, project_lidar_to_image, sample_feature_map


class InfraUniBase(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        d_model = int(cfg["d_model"])
        nhead = int(cfg["nhead"])
        dropout = float(cfg["dropout"])
        self.num_classes = int(cfg["num_classes"])
        self.num_queries = int(cfg["num_queries"])
        self.num_modes = int(cfg["num_modes"])
        self.num_future = int(cfg["num_future"])
        self.pc_range = [float(v) for v in cfg["pc_range"]]
        self.traj_radius = float(cfg.get("traj_radius", 60.0))

        self.pillar = PillarEncoder(self.pc_range, cfg["voxel_size"], out_ch=64)
        self.bev_neck = BevNeck(64, d_model)
        self.heatmap = nn.Conv2d(d_model, self.num_classes, kernel_size=1)
        nn.init.constant_(self.heatmap.bias, -2.19)

        self.image_encoder = ImageEncoder(d_model)
        self.query_embed = nn.Embedding(self.num_queries, d_model)
        self.q_pos = nn.Linear(2, d_model)
        self.bev_pos = nn.Linear(2, d_model)

        self.lidar_decoder = AttnFFN(d_model, nhead, dropout, cross=True)
        self.lidar_self = AttnFFN(d_model, nhead, dropout, cross=False)
        self.image_decoder = AttnFFN(d_model, nhead, dropout, cross=True)
        self.image_inject = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.ReLU(inplace=True),
            nn.Linear(d_model, d_model),
        )

        self.head_lidar = BoxHead(d_model, self.num_classes)
        self.head_cam = BoxHead(d_model, self.num_classes)
        self.head_motion = BoxHead(d_model, self.num_classes)

        self.mode_embed = nn.Parameter(torch.randn(1, 1, self.num_modes, 1, d_model) * 0.02)
        self.time_embed = nn.Parameter(torch.randn(1, 1, 1, self.num_future, d_model) * 0.02)
        self.time_attn = AttnFFN(d_model, nhead, dropout, cross=False)
        self.mode_attn = AttnFFN(d_model, nhead, dropout, cross=False)
        self.traj_bev_fuse = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.ReLU(inplace=True),
            nn.Linear(d_model, d_model),
        )
        self.delta_head = nn.Linear(d_model, 2)
        self.mode_head = nn.Linear(d_model, 1)

        self.motion_gate = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.ReLU(inplace=True),
            nn.Linear(d_model, d_model),
            nn.Sigmoid(),
        )
        self.motion_proj = nn.Linear(d_model, d_model)
        self.motion_norm = nn.LayerNorm(d_model)
        # Predicted waypoints, not only hidden features, update the detection query.
        self.motion_geom = nn.Sequential(
            nn.Linear(self.num_modes * self.num_future * 2, d_model),
            nn.ReLU(inplace=True),
            nn.Linear(d_model, d_model),
        )
        nn.init.normal_(self.motion_geom[-1].weight, std=1e-3)
        nn.init.zeros_(self.motion_geom[-1].bias)

    def _init_queries(self, fine_bev, heatmap):
        batch, _, height, width = heatmap.shape
        score = heatmap.detach().sigmoid().amax(dim=1)
        val, idx = score.view(batch, -1).topk(self.num_queries, dim=1)
        ys = torch.div(idx, width, rounding_mode="floor")
        xs = idx - ys * width
        gather_index = idx[:, None, :].expand(batch, fine_bev.shape[1], self.num_queries)
        content = torch.gather(fine_bev.flatten(2), 2, gather_index).permute(0, 2, 1)
        query = content + self.query_embed.weight[None, :, :]

        xmin, ymin, xmax, ymax = (
            self.pc_range[0],
            self.pc_range[1],
            self.pc_range[3],
            self.pc_range[4],
        )
        vx = (xmax - xmin) / width
        vy = (ymax - ymin) / height
        ref_x = (xs.float() + 0.5) * vx + xmin
        ref_y = (ys.float() + 0.5) * vy + ymin
        ref_xy = torch.stack([ref_x, ref_y], dim=-1)
        xy_norm = torch.stack(
            [
                (ref_x - xmin) / (xmax - xmin) * 2 - 1,
                (ref_y - ymin) / (ymax - ymin) * 2 - 1,
            ],
            dim=-1,
        )
        query = query + self.q_pos(xy_norm)
        return query, ref_xy, val

    def _bev_tokens(self, coarse):
        batch, channels, height, width = coarse.shape
        tokens = coarse.flatten(2).permute(0, 2, 1)
        ys = torch.linspace(-1, 1, height, device=coarse.device)
        xs = torch.linspace(-1, 1, width, device=coarse.device)
        grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
        pos = torch.stack([grid_x, grid_y], dim=-1).reshape(1, height * width, 2)
        return tokens + self.bev_pos(pos)

    def _fuse_image(self, query, boxes, img_feat, lidar2cam, K, image_hw):
        xyz = boxes[..., :3].detach()
        uv, valid = project_lidar_to_image(xyz, lidar2cam, K)
        height, width = image_hw
        u = uv[..., 0] / max(width - 1, 1) * 2 - 1
        v = uv[..., 1] / max(height - 1, 1) * 2 - 1
        grid = torch.stack([u, v], dim=-1).unsqueeze(2)
        sampled = F.grid_sample(
            img_feat, grid, mode="bilinear", padding_mode="zeros", align_corners=True
        )
        sampled = sampled.squeeze(-1).permute(0, 2, 1)
        inside = (
            valid
            & (uv[..., 0] >= 0)
            & (uv[..., 0] <= width - 1)
            & (uv[..., 1] >= 0)
            & (uv[..., 1] <= height - 1)
        )
        sampled = sampled * inside.unsqueeze(-1).to(sampled.dtype)
        query = query + self.image_inject(torch.cat([query, sampled], dim=-1))
        img_tokens = img_feat.flatten(2).permute(0, 2, 1)
        query = self.image_decoder(query, img_tokens)
        return query

    def _forecast(self, det_query, det_xy, fine_bev):
        batch, num_q, channels = det_query.shape
        query = det_query[:, :, None, None, :] + self.mode_embed + self.time_embed
        time_in = query.reshape(batch * num_q * self.num_modes, self.num_future, channels)
        time_out = self.time_attn(time_in).reshape(
            batch, num_q, self.num_modes, self.num_future, channels
        )
        mode_in = time_out.permute(0, 1, 3, 2, 4).reshape(
            batch * num_q * self.num_future, self.num_modes, channels
        )
        mode_out = self.mode_attn(mode_in).reshape(
            batch, num_q, self.num_future, self.num_modes, channels
        )
        query = mode_out.permute(0, 1, 3, 2, 4).contiguous()

        anchor = det_xy.detach()[:, :, None, None, :]
        guess = anchor + self.traj_radius * torch.tanh(self.delta_head(query))
        sampled = sample_feature_map(fine_bev, guess.detach(), self.pc_range)
        query = query + self.traj_bev_fuse(torch.cat([query, sampled], dim=-1))
        delta = self.traj_radius * torch.tanh(self.delta_head(query))
        traj = anchor + delta
        mode_logits = self.mode_head(query.mean(dim=3)).squeeze(-1)
        return traj, mode_logits, query

    def _motion_feedback(self, det_query, traj_query, traj, det_xy):
        hidden = traj_query.mean(dim=(2, 3))
        offset = traj - det_xy.detach()[:, :, None, None, :]
        geom = self.motion_geom(offset.reshape(det_query.shape[0], det_query.shape[1], -1))
        motion = hidden + geom
        gate = self.motion_gate(torch.cat([det_query, motion], dim=-1))
        updated = det_query + gate * self.motion_proj(motion)
        return self.motion_norm(updated)

    def forward(self, batch):
        fine, coarse = self.bev_neck(self.pillar(batch["points"], batch["point_mask"]))
        heatmap = self.heatmap(fine)
        query, ref_xy, query_score = self._init_queries(fine, heatmap)
        bev_tokens = self._bev_tokens(coarse)

        query = self.lidar_self(query)
        query = self.lidar_decoder(query, bev_tokens)
        cls_lidar, box_lidar = self.head_lidar(query, self.pc_range)

        img_feat = self.image_encoder(batch["image"])
        query = self._fuse_image(
            query,
            box_lidar,
            img_feat,
            batch["lidar2cam"],
            batch["K"],
            batch["image"].shape[-2:],
        )
        cls_cam, box_cam = self.head_cam(query, self.pc_range)

        det_xy = box_cam[..., :2]
        traj, mode_logits, traj_query = self._forecast(query, det_xy, fine)
        query = self._motion_feedback(query, traj_query, traj, det_xy)
        cls_motion, box_motion = self.head_motion(query, self.pc_range)

        return {
            "heatmap": heatmap,
            "ref_xy": ref_xy,
            "query_score": query_score,
            "cls": [cls_lidar, cls_cam, cls_motion],
            "box": [box_lidar, box_cam, box_motion],
            "traj": traj,
            "mode_logits": mode_logits,
        }
