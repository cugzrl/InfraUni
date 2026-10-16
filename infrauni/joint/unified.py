"""TransFusion proposals initialize one DeTra decoder.

The detector still builds the lidar-camera query. Its box, class, and score are
the candidate, not the published detection. DeTra's last refinement block is
the only box, class, objectness, and trajectory that evaluation reads.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from detra_repro.config import DataConfig, ModelConfig

from .model import QueryDeTra, _with_current


class UnifiedTransFusionDeTra(nn.Module):
    """Frozen or partially trained TransFusion-LC in front of DeTra."""

    def __init__(self, detector, blocks=2, detach_anchor=True, causal_time=False, forecast_weight=1.0, match_mode="class_gate"):
        super().__init__()
        self.detector = detector
        self.detach_anchor = detach_anchor
        self.causal_time = causal_time
        self.forecast_weight = float(forecast_weight)
        self.match_mode = match_mode
        self.modes = 6
        self.future = 12
        hidden = 128
        self.bev_proj = nn.Conv2d(512, hidden, kernel_size=1)
        self.query_proj = nn.Linear(256, hidden)
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
        self.detra = QueryDeTra(
            cfg,
            DataConfig(
                roi_m=(-54.0, -54.0, 54.0, 54.0),
                voxel_size_m=0.075,
                future_steps=self.future,
            ),
        )
        if causal_time:
            for block in self.detra.blocks:
                block.causal_time = True

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
        self.detector.eval()
        return self

    def _bev(self, pts_feats):
        feat = pts_feats[0] if isinstance(pts_feats, (list, tuple)) else pts_feats
        level8 = self.bev_proj(feat)
        return {
            "stride8": level8,
            "stride4": F.interpolate(level8, scale_factor=2, mode="bilinear", align_corners=True),
            "stride16": F.avg_pool2d(level8, kernel_size=2, stride=2),
        }

    def _proposal_boxes(self, layer):
        """Clone before decode. Official decode writes into its arguments."""

        coder = self.detector.pts_bbox_head.bbox_coder
        center = layer["center"].clone()
        height = layer["height"].clone()
        dim = layer["dim"].clone()
        rot = layer["rot"].clone()
        vel = layer["vel"].clone() if "vel" in layer else None
        heat = layer["heatmap"].sigmoid()
        decoded = coder.decode(heat, rot, dim, center, height, vel, filter=False)
        return torch.stack([item["bboxes"] for item in decoded], dim=0)

    def forward(self, points, img, img_metas):
        detector = self.detector
        img_feats, pts_feats = detector.extract_feat(points, img=img, img_metas=img_metas)
        outs = detector.pts_bbox_head(pts_feats, img_feats, img_metas)
        layer = outs[0][0] if isinstance(outs[0], (list, tuple)) else outs[0]
        if "fused_query" not in layer:
            raise RuntimeError("TransFusion head did not export fused_query")
        query = layer["fused_query"]
        proposal_boxes = self._proposal_boxes(layer)
        head = detector.pts_bbox_head
        proposal_labels = head.query_labels
        gated = layer["heatmap"].detach().sigmoid() * layer["query_heatmap_score"].detach()
        gated = gated * F.one_hot(proposal_labels, num_classes=gated.shape[1]).permute(0, 2, 1).float()
        proposal_scores = gated.max(dim=1).values
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
        class_prob = detra_out["class_logits"].softmax(dim=-1)
        labels = class_prob.argmax(dim=-1)
        scores = detra_out["object_logits"].sigmoid() * class_prob.max(dim=-1).values
        boxes = torch.cat([detra_out["boxes"], detra_out["pred_velocity"]], dim=-1)
        future = detra_out["future_xy"]
        origin = detra_out["poses"][:, :, :, 0, :2]
        if self.detach_anchor:
            future = future - origin.unsqueeze(3) + origin.detach().unsqueeze(3)
        trajectories = _with_current(boxes[..., :2].detach() if self.detach_anchor else boxes[..., :2], future)
        return {
            "boxes": boxes,
            "proposal_boxes": proposal_boxes,
            "proposal_scores": proposal_scores,
            "proposal_labels": proposal_labels,
            "scores": scores,
            "labels": labels,
            "trajectories": trajectories,
            "mode_logits": detra_out["mode_logits"],
            "query_index": torch.arange(boxes.shape[1], device=boxes.device),
            "query": query,
            "detra": detra_out,
            "outs": outs,
            "box_mode": "full",
            "detach_anchor": self.detach_anchor,
            "forecast_weight": self.forecast_weight,
            "match_mode": self.match_mode,
            "unified": True,
            "object_logits": detra_out["object_logits"],
            "class_logits": detra_out["class_logits"],
        }
