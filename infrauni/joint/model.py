"""TransFusion-LC proposals drive an unmodified DeTra refinement stack.

Version B detaches the detector query and predicts trajectories with an MLP.
Version C injects that query into MC-DeTra. PoseUpdate then writes the current
box from the same query that produced the future waypoints.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from detra_repro.config import DataConfig, ModelConfig
from detra_repro.models.refinement import DeTraMini


class QueryDeTra(DeTraMini):
    """DeTraMini plus one object embedding taken from the detector query."""

    def forward(self, lidar_features, initial_poses, initial_boxes=None, object_embed=None):
        lidar_features = self._as_multiscale(lidar_features)
        batch, count, _ = initial_poses.shape
        modes, times, dim = self.cfg.num_modes, self.cfg.num_times, self.cfg.hidden_dim
        tokens = (self.mode_embed + self.time_embed).expand(batch, count, modes, times, dim).contiguous()
        if object_embed is not None:
            tokens = tokens + object_embed[:, :, None, None, :]
        poses = initial_poses[:, :, None, None, :].expand(batch, count, modes, times, 3).contiguous()
        box = initial_boxes.clone() if initial_boxes is not None else self._default_box(initial_poses)
        aux = []
        for block in self.blocks:
            anchor_pose = poses.detach() if self.cfg.detach_pose_refinement else poses
            anchor_box = box.detach() if self.cfg.detach_pose_refinement else box
            tokens = block(tokens, anchor_pose, lidar_features)
            poses, box, pred = self.pose_update(tokens, anchor_pose, anchor_box)
            aux.append(pred)
        pred = dict(aux[-1])
        pred["aux_outputs"] = aux
        pred["boxes"] = box
        return pred


class IndependentForecast(nn.Module):
    """One-shot multi-modal offsets added to a detached detection center."""

    def __init__(self, in_dim, modes, future):
        super().__init__()
        self.modes = modes
        self.future = future
        self.net = nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, modes * future * 2),
        )
        self.mode_head = nn.Linear(in_dim, modes)

    def forward(self, query, center):
        batch, _, count = query.shape
        flat = query.transpose(1, 2)
        delta = self.net(flat).view(batch, count, self.modes, self.future, 2)
        return center[:, :, None, None, :] + delta, self.mode_head(flat)


def _with_current(center, future):
    """Prepend the current center so trajectories are [B, N, modes, 1+T, 2]."""

    current = center[:, :, None, None, :].expand(-1, -1, future.shape[2], 1, 2)
    return torch.cat([current, future], dim=3)


class JointTransFusionDeTra(nn.Module):
    """Official detector plus either an independent head or DeTra refinement."""

    def __init__(self, detector, mode="joint", blocks=2):
        super().__init__()
        if mode not in {"pred_head", "joint", "cv"}:
            raise ValueError("mode must be pred_head, joint, or cv")
        self.detector = detector
        self.mode = mode
        self.modes = 6
        self.future = 12
        self.data_cfg = DataConfig(
            roi_m=(-54.0, -54.0, 54.0, 54.0),
            voxel_size_m=0.075,
            future_steps=self.future,
        )
        hidden = 128
        # SECONDFPN concatenates its two 256-d maps.
        self.bev_proj = nn.Conv2d(512, hidden, kernel_size=1)
        self.query_proj = nn.Linear(256, hidden)
        if mode == "pred_head":
            self.forecast = IndependentForecast(256, self.modes, self.future)
            self.detra = None
        elif mode == "cv":
            self.forecast = None
            self.detra = None
        else:
            self.forecast = None
            cfg = ModelConfig(
                num_queries=200,
                num_modes=self.modes,
                num_times=self.future + 1,
                hidden_dim=hidden,
                num_refinement_blocks=blocks,
                num_classes=10,
                attention_type="deformable",
                lidar_attention_anchor="current",
                detach_pose_refinement=True,
                forecast_base_mode="current",
            )
            self.detra = QueryDeTra(cfg, self.data_cfg)
        self._captured = {}
        detector.pts_bbox_head.prediction_heads[-1].register_forward_hook(self._capture_query)

    def _capture_query(self, module, inputs, output):
        self._captured["query"] = inputs[0]

    def freeze_detector(self):
        self.detector.eval()
        for param in self.detector.parameters():
            param.requires_grad_(False)

    def unfreeze_fusion(self):
        """Train the image-query decoder and the fusion prediction head."""

        head = self.detector.pts_bbox_head
        modules = [head.decoder[1], head.prediction_heads[-1]]
        for name in ("heatmap_head_img", "shared_conv_img", "fc"):
            if hasattr(head, name):
                modules.append(getattr(head, name))
        for module in modules:
            module.train()
            for param in module.parameters():
                param.requires_grad_(True)

    def train(self, mode=True):
        super().train(mode)
        # Frozen weights stay in eval. Unfrozen fusion modules are set afterwards.
        self.detector.eval()
        return self

    def _bev(self, pts_feats):
        feat = pts_feats[0] if isinstance(pts_feats, (list, tuple)) else pts_feats
        if feat.shape[1] != self.bev_proj.in_channels:
            raise RuntimeError("unexpected BEV channels %s" % (tuple(feat.shape),))
        level8 = self.bev_proj(feat)
        return {
            "stride8": level8,
            "stride4": F.interpolate(level8, scale_factor=2, mode="bilinear", align_corners=True),
            "stride16": F.avg_pool2d(level8, kernel_size=2, stride=2),
        }

    def _decode(self, layer):
        # Official decode edits its inputs. Clone first.
        coder = self.detector.pts_bbox_head.bbox_coder
        center = layer["center"].detach().clone()
        height = layer["height"].detach().clone()
        dim = layer["dim"].detach().clone()
        rot = layer["rot"].detach().clone()
        vel = layer["vel"].detach().clone() if "vel" in layer else None
        heat = layer["heatmap"].detach().sigmoid()
        return coder.decode(heat, rot, dim, center, height, vel, filter=False)

    def forward(self, points, img, img_metas):
        self._captured = {}
        detector = self.detector
        img_feats, pts_feats = detector.extract_feat(points, img=img, img_metas=img_metas)
        outs = detector.pts_bbox_head(pts_feats, img_feats, img_metas)
        query = self._captured["query"]
        layer = outs[0][0] if isinstance(outs[0], (list, tuple)) else outs[0]
        decoded = self._decode(layer)
        proposal_boxes = torch.stack([item["bboxes"] for item in decoded], dim=0)
        center = proposal_boxes[..., :2]
        detra_out = None
        head = detector.pts_bbox_head
        # Same gated score as TransFusionHead.get_bboxes, without dropping the query index.
        gated = layer["heatmap"].detach().sigmoid() * layer["query_heatmap_score"].detach()
        gated = gated * F.one_hot(head.query_labels, num_classes=gated.shape[1]).permute(0, 2, 1).float()
        proposal_scores = gated.max(dim=1).values
        proposal_labels = head.query_labels
        if self.mode == "pred_head":
            future, mode_logits = self.forecast(query.detach(), center.detach())
            boxes = proposal_boxes
            trajectories = _with_current(center.detach(), future)
        elif self.mode == "cv":
            # nuScenes keyframes are 2 Hz, so each step is 0.5 s.
            steps = torch.arange(1, self.future + 1, device=center.device, dtype=center.dtype)
            steps = steps.view(1, 1, 1, -1, 1) * 0.5
            velocity = proposal_boxes[..., 7:9]
            future = center[:, :, None, None, :] + velocity[:, :, None, None, :] * steps
            future = future.expand(-1, -1, self.modes, -1, -1)
            boxes = proposal_boxes
            trajectories = _with_current(center, future)
            mode_logits = center.new_zeros(center.shape[0], center.shape[1], self.modes)
        else:
            poses = torch.stack(
                [proposal_boxes[..., 0], proposal_boxes[..., 1], proposal_boxes[..., 6]],
                dim=-1,
            )
            detra_out = self.detra(
                self._bev(pts_feats),
                poses,
                proposal_boxes[..., :7],
                object_embed=self.query_proj(query.transpose(1, 2)),
            )
            boxes = proposal_boxes.clone()
            boxes[..., :7] = detra_out["boxes"]
            trajectories = _with_current(boxes[..., :2], detra_out["future_xy"])
            mode_logits = detra_out["mode_logits"]
        return {
            "boxes": boxes,
            "proposal_boxes": proposal_boxes,
            "scores": proposal_scores,
            "labels": proposal_labels,
            "trajectories": trajectories,
            "mode_logits": mode_logits,
            "query_index": torch.arange(boxes.shape[1], device=boxes.device),
            "query": query,
            "detra": detra_out,
            "outs": outs,
        }
