"""Detection, heatmap and winner-takes-all trajectory losses."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment


def centernet_focal(logits, target):
    pred = logits.sigmoid().clamp(1e-4, 1.0 - 1e-4)
    pos_mask = target.eq(1).float()
    neg_mask = target.lt(1).float()
    pos_loss = -torch.log(pred) * (1 - pred).pow(2) * pos_mask
    neg_loss = -torch.log(1 - pred) * pred.pow(2) * (1 - target).pow(4) * neg_mask
    return (pos_loss.sum() + neg_loss.sum()) / pos_mask.sum().clamp(min=1.0)


def sigmoid_focal(logits, targets, alpha=0.25, gamma=2.0):
    prob = logits.sigmoid()
    ce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    p_t = prob * targets + (1.0 - prob) * (1.0 - targets)
    alpha_t = alpha * targets + (1.0 - alpha) * (1.0 - targets)
    loss = alpha_t * (1.0 - p_t).pow(gamma) * ce
    return loss.sum() / targets.sum().clamp(min=1.0)


def render_heatmap(boxes, labels, gt_mask, spatial_hw, pc_range, num_classes, sigma=1.5):
    """Gaussian peaks on the pillar grid. Peak cells are set to 1."""
    batch, _, _ = boxes.shape
    height, width = spatial_hw
    heat = boxes.new_zeros(batch, num_classes, height, width)
    xmin, ymin, xmax, ymax = pc_range[0], pc_range[1], pc_range[3], pc_range[4]
    ys = torch.arange(height, device=boxes.device, dtype=boxes.dtype)
    xs = torch.arange(width, device=boxes.device, dtype=boxes.dtype)
    gy, gx = torch.meshgrid(ys, xs, indexing="ij")
    for b in range(batch):
        valid = gt_mask[b].nonzero(as_tuple=False).squeeze(1)
        for m in valid.tolist():
            px = (boxes[b, m, 0] - xmin) / (xmax - xmin) * width
            py = (boxes[b, m, 1] - ymin) / (ymax - ymin) * height
            if px < 0 or py < 0 or px >= width or py >= height:
                continue
            dist2 = (gx + 0.5 - px) ** 2 + (gy + 0.5 - py) ** 2
            blob = torch.exp(-dist2 / (2.0 * sigma * sigma))
            cls_id = int(labels[b, m].item())
            heat[b, cls_id] = torch.maximum(heat[b, cls_id], blob)
            heat[b, cls_id, int(py.item()), int(px.item())] = 1.0
    return heat


def match_detections(cls_logits, boxes, gt_labels, gt_boxes, gt_mask):
    """Hungarian match on the camera-refined boxes. Indices are detached."""
    prob = cls_logits.detach().sigmoid()
    pred_xy = boxes.detach()[..., :2]
    matches = []
    batch = cls_logits.shape[0]
    for b in range(batch):
        gt_idx = gt_mask[b].nonzero(as_tuple=False).squeeze(1)
        if gt_idx.numel() == 0:
            empty = gt_idx.new_zeros((0,), dtype=torch.long)
            matches.append((empty, empty))
            continue
        cost_cls = -prob[b][:, gt_labels[b, gt_idx]]
        cost_box = torch.cdist(pred_xy[b], gt_boxes[b, gt_idx, :2])
        cost = (5.0 * cost_cls + cost_box).cpu().numpy()
        rows, cols = linear_sum_assignment(cost)
        pred_ind = torch.as_tensor(rows, device=cls_logits.device, dtype=torch.long)
        gt_ind = gt_idx[torch.as_tensor(cols, device=cls_logits.device, dtype=torch.long)]
        matches.append((pred_ind, gt_ind))
    return matches


def _box_l1(pred, gt):
    xy = (pred[:2] - gt[:2]).abs().mean()
    z = (pred[2] - gt[2]).abs()
    size = (pred[3:6] - gt[3:6]).abs().mean()
    yaw = (torch.sin(pred[6]) - torch.sin(gt[6])).abs() + (
        torch.cos(pred[6]) - torch.cos(gt[6])
    ).abs()
    return xy + 0.5 * z + 0.5 * size + 0.5 * yaw


class JointLoss(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.pc_range = [float(v) for v in cfg["pc_range"]]
        self.num_classes = int(cfg["num_classes"])
        weights = cfg["loss"]
        self.w_hm = float(weights["heatmap"])
        self.w_cls = float(weights["cls"])
        self.w_box = float(weights["box"])
        self.w_traj = float(weights["traj"])
        self.w_mode = float(weights["mode"])
        self.stage_weights = [float(v) for v in weights["stage_weights"]]

    def forward(self, out, batch):
        heat_gt = render_heatmap(
            batch["boxes"],
            batch["labels"],
            batch["gt_mask"],
            out["heatmap"].shape[-2:],
            self.pc_range,
            self.num_classes,
        )
        loss_hm = centernet_focal(out["heatmap"], heat_gt)
        # Match with the camera stage; supervise every detection stage
        # and the trajectory head with that assignment.
        matches = match_detections(
            out["cls"][1], out["box"][1], batch["labels"], batch["boxes"], batch["gt_mask"]
        )

        loss_cls = out["heatmap"].new_zeros(())
        loss_box = out["heatmap"].new_zeros(())
        matched = 0
        for stage, (cls_logits, boxes) in enumerate(zip(out["cls"], out["box"])):
            target = cls_logits.new_zeros(cls_logits.shape)
            stage_box = cls_logits.new_zeros(())
            stage_count = 0
            for b, (pred_ind, gt_ind) in enumerate(matches):
                if pred_ind.numel() == 0:
                    continue
                target[b, pred_ind, batch["labels"][b, gt_ind]] = 1.0
                for qi, gi in zip(pred_ind.tolist(), gt_ind.tolist()):
                    stage_box = stage_box + _box_l1(boxes[b, qi], batch["boxes"][b, gi])
                    stage_count += 1
            loss_cls = loss_cls + self.stage_weights[stage] * sigmoid_focal(cls_logits, target)
            if stage_count > 0:
                loss_box = loss_box + self.stage_weights[stage] * stage_box / stage_count
                matched = max(matched, stage_count)

        loss_traj = out["heatmap"].new_zeros(())
        loss_mode = out["heatmap"].new_zeros(())
        traj_count = 0
        traj = out["traj"]
        mode_logits = out["mode_logits"]
        for b, (pred_ind, gt_ind) in enumerate(matches):
            for qi, gi in zip(pred_ind.tolist(), gt_ind.tolist()):
                step_mask = batch["traj_mask"][b, gi]
                if float(step_mask.sum()) < 1:
                    continue
                pred = traj[b, qi]
                gt = batch["traj"][b, gi]
                err = (pred - gt.unsqueeze(0)).norm(dim=-1)
                err = (err * step_mask.unsqueeze(0)).sum(-1) / step_mask.sum()
                winner = int(torch.argmin(err).item())
                loss_traj = loss_traj + err[winner]
                target_mode = torch.tensor([winner], device=mode_logits.device)
                loss_mode = loss_mode + F.cross_entropy(
                    mode_logits[b, qi].unsqueeze(0), target_mode
                )
                traj_count += 1
        if traj_count > 0:
            loss_traj = loss_traj / traj_count
            loss_mode = loss_mode / traj_count

        total = (
            self.w_hm * loss_hm
            + self.w_cls * loss_cls
            + self.w_box * loss_box
            + self.w_traj * loss_traj
            + self.w_mode * loss_mode
        )
        stats = {
            "loss": float(total.detach()),
            "hm": float(loss_hm.detach()),
            "cls": float(loss_cls.detach()),
            "box": float(loss_box.detach()),
            "traj": float(loss_traj.detach()),
            "mode": float(loss_mode.detach()),
            "matched": matched,
            "traj_count": traj_count,
        }
        parts = {
            "det": self.w_hm * loss_hm + self.w_cls * loss_cls + self.w_box * loss_box,
            "forecast": self.w_traj * loss_traj + self.w_mode * loss_mode,
        }
        return total, stats, parts
