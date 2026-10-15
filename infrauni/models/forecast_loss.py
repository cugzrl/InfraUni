"""Detection matching and DeTra winner-take-all forecast loss."""

from __future__ import annotations

import torch
from scipy.optimize import linear_sum_assignment


def match_agents(pred_xy, pred_labels, gt_xy, gt_labels):
    """Class-aware Hungarian matching on current BEV centers."""
    if pred_xy.numel() == 0 or gt_xy.numel() == 0:
        empty = pred_xy.new_zeros((0,), dtype=torch.long)
        return empty, empty
    cost = torch.cdist(pred_xy.detach(), gt_xy.detach(), p=2)
    mismatch = pred_labels[:, None] != gt_labels[None, :]
    cost = cost.masked_fill(mismatch, 1e3)
    rows, cols = linear_sum_assignment(cost.cpu().numpy())
    rows = torch.as_tensor(rows, device=pred_xy.device)
    cols = torch.as_tensor(cols, device=pred_xy.device)
    keep = cost[rows, cols] < 1e3
    return rows[keep], cols[keep]


def forecast_losses(poses, logits, centers, gt_boxes, gt_labels, gt_future, gt_mask, pred_labels):
    """Auxiliary loss at every refinement block.

    ``poses`` time 0 is the refined detection center. Future steps are offsets
    from that center, so this loss updates the reported box.
    """
    device = poses[0].device
    total_reg = poses[0].new_zeros(())
    total_cls = poses[0].new_zeros(())
    total_center = poses[0].new_zeros(())
    matched = 0
    gt_xy = gt_boxes.gravity_center[:, :2].to(device)
    gt_labels = gt_labels.to(device)
    gt_future = gt_future.to(device)
    gt_mask = gt_mask.to(device)
    for pose, logit, center in zip(poses, logits, centers):
        pred_xy = center[0]
        rows, cols = match_agents(pred_xy, pred_labels, gt_xy, gt_labels)
        matched += int(rows.numel())
        if rows.numel() == 0:
            # Keep the graph alive when a frame has no match.
            total_reg = total_reg + pose.sum() * 0
            total_cls = total_cls + logit.sum() * 0
            total_center = total_center + center.sum() * 0
            continue
        total_center = total_center + (center[0, rows] - gt_xy[cols]).abs().mean()
        future = pose[0, rows, :, 1:]
        target = gt_future[cols][:, None, :, :]
        mask = gt_mask[cols][:, None, :, None]
        error = (future - target).norm(dim=-1) * mask.squeeze(-1)
        denom = mask.squeeze(-1).sum(dim=-1).clamp(min=1.0)
        ade = error.sum(dim=-1) / denom
        best = ade.argmin(dim=-1)
        gather = best[:, None, None, None].expand(-1, 1, future.shape[2], 2)
        chosen = future.gather(1, gather).squeeze(1)
        reg = ((chosen - gt_future[cols]).abs() * gt_mask[cols][:, :, None]).sum()
        reg = reg / gt_mask[cols].sum().clamp(min=1.0)
        total_reg = total_reg + reg
        total_cls = total_cls + torch.nn.functional.cross_entropy(logit[0, rows], best)
    scale = float(len(poses))
    return {
        "loss_center": total_center / scale,
        "loss_traj": total_reg / scale,
        "loss_mode": total_cls / scale,
        "matched": poses[0].new_tensor(matched / scale),
    }


def pred_head_loss(pose, logits, gt_boxes, gt_labels, gt_future, gt_mask, pred_labels):
    """Forecast loss for the MLP head. The current center is detached upstream."""
    device = pose.device
    gt_xy = gt_boxes.gravity_center[:, :2].to(device)
    rows, cols = match_agents(pose[0, :, 0, 0], pred_labels, gt_xy, gt_labels.to(device))
    if rows.numel() == 0:
        zero = pose.sum() * 0
        return {"loss_traj": zero, "loss_mode": zero, "matched": pose.new_zeros(())}
    future = pose[0, rows, :, 1:]
    target = gt_future.to(device)[cols]
    mask = gt_mask.to(device)[cols]
    error = (future - target[:, None]).norm(dim=-1) * mask[:, None]
    ade = error.sum(-1) / mask.sum(-1).clamp(min=1.0)[:, None]
    best = ade.argmin(-1)
    chosen = future.gather(1, best[:, None, None, None].expand(-1, 1, future.shape[2], 2)).squeeze(1)
    reg = ((chosen - target).abs() * mask[:, :, None]).sum() / mask.sum().clamp(min=1.0)
    cls = torch.nn.functional.cross_entropy(logits[0, rows], best)
    return {"loss_traj": reg, "loss_mode": cls, "matched": pose.new_tensor(float(rows.numel()))}
