"""TransFusion detector with an optional DeTra trajectory decoder.

Modes:
    det        detection only
    pred_head  an MLP forecast head on detection queries
    joint      DeTra refinement; time 0 of every mode is the reported center
"""

from __future__ import annotations

import torch
from torch import nn

from ..data.constants import NUM_FUTURE, NUM_MODES, PC_RANGE, POST_CENTER_RANGE
from .camera_backbone import CameraBackbone
from .detra import DetraDecoder
from .forecast_loss import forecast_losses, pred_head_loss
from .lidar_backbone import LidarBackbone
from .transfusion import build_transfusion_head, decode_proposals, detection_loss, fused_object_query


class TrajectoryHead(nn.Module):
    """One-shot forecast from a detection query. The box center stays detached."""

    def __init__(self, dim=128, modes=NUM_MODES, future_steps=NUM_FUTURE):
        super().__init__()
        self.modes = modes
        self.future_steps = future_steps
        self.offset = nn.Sequential(nn.Linear(dim, dim), nn.ReLU(inplace=True), nn.Linear(dim, modes * future_steps * 2))
        self.mode = nn.Linear(dim, modes)

    def forward(self, query, center):
        token = query.permute(0, 2, 1)
        batch, agents, _ = token.shape
        delta = self.offset(token).view(batch, agents, self.modes, self.future_steps, 2)
        anchor = center.detach()[:, :, None, None, :]
        current = anchor.expand(batch, agents, self.modes, 1, 2)
        future = anchor + delta
        return torch.cat([current, future], dim=3), self.mode(token)


class InfraUniDetector(nn.Module):
    def __init__(self, mode="det", num_proposals=200, voxel_size=None, refine_blocks=2):
        super().__init__()
        if mode not in ("det", "pred_head", "joint"):
            raise ValueError(mode)
        self.mode = mode
        self.lidar = LidarBackbone(voxel_size=voxel_size)
        self.camera = CameraBackbone()
        self.head = build_transfusion_head(
            num_proposals=num_proposals,
            voxel_size=self.lidar.voxel_size,
            grid_size=self.lidar.grid_size,
        )
        self.pred_head = TrajectoryHead() if mode == "pred_head" else None
        self.detra = DetraDecoder(blocks=refine_blocks) if mode == "joint" else None

    def train(self, mode=True):
        super().train(mode)
        self.camera.train(mode)
        return self

    def forward(self, batch):
        points = [cloud.to(self._device()) for cloud in batch["points"]]
        images = batch["img"].to(self._device())
        metas = batch["img_metas"]
        for meta in metas:
            meta["input_shape"] = (images.shape[-2], images.shape[-1])
        bev, lidar_scales = self.lidar(points)
        cam_feats = self.camera(images)
        preds = self.head([bev], cam_feats, metas)
        query = fused_object_query(self.head)
        boxes, scores, labels = decode_proposals(self.head, preds)
        output = {
            "preds": preds,
            "query": query,
            "boxes": boxes,
            "scores": scores,
            "labels": labels,
        }
        if self.mode == "pred_head":
            pose, logits = self.pred_head(query, boxes[:, :2].unsqueeze(0))
            output.update(poses=[pose], logits=[logits], centers=[boxes[:, :2].unsqueeze(0)])
        elif self.mode == "joint":
            refined = self.detra(query, boxes.unsqueeze(0), lidar_scales, cam_feats, metas)
            output.update(refined)
            output["boxes"] = self._apply_center(boxes, refined["centers"][-1][0])
        return output

    def loss(self, batch, output):
        gt_boxes = batch["gt_bboxes_3d"]
        gt_labels = [label.to(self._device()) for label in batch["gt_labels_3d"]]
        det = detection_loss(self.head, gt_boxes, gt_labels, output["preds"])
        losses = {key: value for key, value in det.items()}
        if self.mode == "det":
            return losses
        if self.mode == "pred_head":
            extra = pred_head_loss(
                output["poses"][0],
                output["logits"][0],
                gt_boxes[0],
                gt_labels[0],
                batch["gt_fut"][0],
                batch["gt_fut_mask"][0],
                output["labels"],
            )
        else:
            extra = forecast_losses(
                output["poses"],
                output["logits"],
                output["centers"],
                gt_boxes[0],
                gt_labels[0],
                batch["gt_fut"][0],
                batch["gt_fut_mask"][0],
                output["labels"],
            )
        losses.update(extra)
        return losses

    def predict(self, batch):
        output = self.forward(batch)
        boxes, scores, labels, traj = self._filter(output)
        return {
            "boxes": boxes,
            "scores": scores,
            "labels": labels,
            "trajectories": traj,
            "token": batch["token"][0],
            "refine_blocks": 0 if self.detra is None else self.detra.executed_blocks,
        }

    def _apply_center(self, boxes, center):
        updated = boxes.clone()
        updated[:, :2] = center
        return updated

    def _filter(self, output):
        boxes = output["boxes"]
        scores = output["scores"]
        labels = output["labels"]
        lo, hi = POST_CENTER_RANGE[:3], POST_CENTER_RANGE[3:]
        center = boxes[:, :3]
        keep = ((center >= center.new_tensor(lo)) & (center <= center.new_tensor(hi))).all(-1)
        if "poses" in output:
            traj = output["poses"][-1][0]
            if self.mode == "joint":
                # Every mode's current waypoint is the refined detection center.
                traj = traj.clone()
                traj[:, :, 0, :] = boxes[:, None, :2]
        else:
            traj = boxes[:, :2][:, None, None, :].expand(boxes.shape[0], NUM_MODES, NUM_FUTURE + 1, 2).contiguous()
        return boxes[keep], scores[keep], labels[keep], traj[keep]

    def _device(self):
        return next(self.head.parameters()).device


def trainable_parameters(model):
    return [param for param in model.parameters() if param.requires_grad]


def in_range_point(xy):
    return (
        (xy[:, 0] > PC_RANGE[0])
        & (xy[:, 0] < PC_RANGE[3])
        & (xy[:, 1] > PC_RANGE[1])
        & (xy[:, 1] < PC_RANGE[4])
    )
