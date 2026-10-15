"""BEV and camera overlays for one frame."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch


def _box_corners(box):
    x, y, _, width, length, _, yaw = box[:7]
    corners = np.array(
        [[length / 2, width / 2], [length / 2, -width / 2], [-length / 2, -width / 2], [-length / 2, width / 2]]
    )
    rotation = np.array([[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]])
    return corners @ rotation.T + np.array([x, y])


def save_frame(path, points, gt_boxes, pred_boxes, trajectories, scores, score_min=0.1):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    points = points.detach().cpu().numpy() if torch.is_tensor(points) else np.asarray(points)
    gt_boxes = gt_boxes.tensor.detach().cpu().numpy() if hasattr(gt_boxes, "tensor") else np.asarray(gt_boxes)
    pred_boxes = pred_boxes.detach().cpu().numpy() if torch.is_tensor(pred_boxes) else np.asarray(pred_boxes)
    trajectories = np.asarray(trajectories)
    scores = np.asarray(scores)
    keep = scores >= score_min
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(points[::4, 0], points[::4, 1], s=0.2, c="0.6", linewidths=0)
    for box in gt_boxes:
        corners = _box_corners(box)
        ax.plot(np.r_[corners[:, 0], corners[0, 0]], np.r_[corners[:, 1], corners[0, 1]], color="lime", lw=1)
    for box, traj in zip(pred_boxes[keep], trajectories[keep]):
        corners = _box_corners(box)
        ax.plot(np.r_[corners[:, 0], corners[0, 0]], np.r_[corners[:, 1], corners[0, 1]], color="tomato", lw=1)
        best = traj[0, 1:]
        ax.plot(best[:, 0], best[:, 1], color="deepskyblue", lw=1)
        ax.scatter([box[0]], [box[1]], c="red", s=8)
        ax.scatter([traj[0, 0, 0]], [traj[0, 0, 1]], c="cyan", s=8)
    ax.set_xlim(-54, 54)
    ax.set_ylim(-54, 54)
    ax.set_aspect("equal")
    ax.set_title(path.stem)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
