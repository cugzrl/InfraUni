"""BEV figure for a single frame: points, boxes and future trajectories."""

from __future__ import annotations

import numpy as np


def _corners(box):
    x, y, _, width, length, _, yaw = box
    c, s = np.cos(yaw), np.sin(yaw)
    local = np.array(
        [
            [length / 2, width / 2],
            [length / 2, -width / 2],
            [-length / 2, -width / 2],
            [-length / 2, width / 2],
            [length / 2, width / 2],
        ]
    )
    rot = np.array([[c, -s], [s, c]])
    return local @ rot.T + np.array([x, y])


def save_bev(path, points, gt, pred, pc_range, title):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 7))
    if points is not None and len(points):
        step = max(len(points) // 8000, 1)
        pts = points[::step]
        ax.scatter(pts[:, 0], pts[:, 1], s=0.2, c="0.6", linewidths=0)
    for box in gt["box"]:
        poly = _corners(box)
        ax.plot(poly[:, 0], poly[:, 1], color="lime", linewidth=1.0)
    if gt.get("traj") is not None:
        for xy, traj, mask in zip(gt["xy"], gt["traj"], gt["traj_mask"]):
            valid = mask > 0.5
            if valid.sum() < 1:
                continue
            path_xy = np.concatenate([xy[None, :], traj[valid]], axis=0)
            ax.plot(path_xy[:, 0], path_xy[:, 1], color="green", linewidth=0.8)
    ax.plot([], [], color="lime", label="GT box")
    ax.plot([], [], color="green", label="GT future")
    ax.plot([], [], color="red", label="pred box")
    ax.plot([], [], color="orange", label="pred future (mode 0)")
    if len(pred["score"]):
        rank = pred["query_score"] if "query_score" in pred else pred["score"]
        order = np.argsort(-rank)[:20]
        for box in pred["box"][order]:
            poly = _corners(box)
            ax.plot(poly[:, 0], poly[:, 1], color="red", linewidth=0.8, alpha=0.85)
        if pred.get("traj") is not None:
            for xy, traj in zip(pred["xy"][order], pred["traj"][order]):
                path_xy = np.concatenate([xy[None, :], traj[0]], axis=0)
                ax.plot(path_xy[:, 0], path_xy[:, 1], color="orange", linewidth=0.7, alpha=0.85)
    ax.legend(loc="upper right", fontsize=8)
    ax.set_xlim(pc_range[0], pc_range[3])
    ax.set_ylim(pc_range[1], pc_range[4])
    ax.set_aspect("equal")
    ax.set_title(title)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
