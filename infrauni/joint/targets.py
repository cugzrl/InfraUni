"""Match detector queries to precomputed future tracks. No future labels enter the model."""

from __future__ import annotations

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment


def align_future(pred_xy, track, dist_thresh=2.0):
    """Align one frame of predictions with future supervision.

    Args:
        pred_xy: ``[N, 2]`` current predicted centers.
        track: dict with ``current``, ``future``, ``mask``, ``labels``.

    Returns:
        ``future [1,N,T,2]``, ``mask [1,N,T]``, ``object_targets [1,N]``,
        and the matched ground-truth row for each query (``-1`` if unmatched).
    """

    device = pred_xy.device
    count = pred_xy.shape[0]
    future_steps = track["future"].shape[1] if len(track["future"]) else 12
    future = torch.zeros(1, count, future_steps, 2, device=device)
    mask = torch.zeros(1, count, future_steps, device=device, dtype=torch.bool)
    targets = torch.zeros(1, count, device=device)
    matched = np.full(count, -1, dtype=np.int64)
    if len(track["current"]) == 0 or count == 0:
        return future, mask, targets, matched
    cost = torch.cdist(pred_xy[None].float(), torch.as_tensor(track["current"], device=device)[None].float())[0]
    cost_np = cost.detach().cpu().numpy()
    rows, cols = linear_sum_assignment(cost_np)
    for row, col in zip(rows, cols):
        if cost_np[row, col] > dist_thresh:
            continue
        future[0, row] = torch.as_tensor(track["future"][col], device=device)
        mask[0, row] = torch.as_tensor(track["mask"][col], device=device)
        targets[0, row] = 1
        matched[row] = col
    return future, mask, targets, matched
