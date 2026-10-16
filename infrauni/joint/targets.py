"""Class-aware matching of queries to current boxes and future tracks.

Ground-truth futures are supervision only. Inference never reads them.
"""

from __future__ import annotations

import torch


def match_queries(pred_xy, pred_labels, gt_xy, gt_labels, dist_thresh=2.0, class_penalty=1.0e3):
    """Hungarian match on BEV center, with a hard class gate.

    Returns ``[N]`` ground-truth rows, or ``-1`` when unmatched. Two nearby
    objects of different classes are not paired.
    """

    count = pred_xy.shape[0]
    matched = torch.full((count,), -1, dtype=torch.long, device=pred_xy.device)
    if count == 0 or gt_xy.shape[0] == 0:
        return matched
    distance = torch.cdist(pred_xy.float(), gt_xy.float())
    distance = torch.nan_to_num(distance, nan=1.0e6, posinf=1.0e6, neginf=1.0e6)
    same = pred_labels[:, None] == gt_labels[None, :]
    cost = distance + (~same).float() * class_penalty
    # linear_sum_assignment is CPU-only.
    rows, cols = _assign(cost.detach().cpu())
    for row, col in zip(rows, cols):
        if (not bool(same[row, col])) or float(distance[row, col]) > dist_thresh:
            continue
        matched[row] = col
    return matched


def match_cost(
    pred_xy,
    pred_boxes,
    class_logits,
    object_logits,
    gt_boxes,
    gt_labels,
    dist_thresh=2.0,
):
    """Hungarian match on distance, box shape, and DeTra class probability.

    The class term uses the decoder distribution, not the TransFusion label.
    A nearby query can receive a ground-truth object whose candidate class
    was wrong. Pairs farther than ``dist_thresh`` stay unmatched, and each
    ground-truth object is used at most once.
    """

    count = pred_xy.shape[0]
    matched = torch.full((count,), -1, dtype=torch.long, device=pred_xy.device)
    if count == 0 or gt_boxes.shape[0] == 0:
        return matched
    distance = torch.cdist(pred_xy.float(), gt_boxes[:, :2].float())
    distance = torch.nan_to_num(distance, nan=1.0e6, posinf=1.0e6, neginf=1.0e6)
    log_size = pred_boxes[:, 3:6].clamp_min(1e-2).log()
    gt_log_size = gt_boxes[:, 3:6].clamp_min(1e-2).log()
    size = (log_size[:, None, :] - gt_log_size[None, :, :]).abs().sum(dim=-1)
    heading = torch.atan2(
        torch.sin(pred_boxes[:, None, 6] - gt_boxes[None, :, 6]),
        torch.cos(pred_boxes[:, None, 6] - gt_boxes[None, :, 6]),
    ).abs()
    class_cost = -class_logits.log_softmax(dim=-1)[:, gt_labels]
    object_cost = -object_logits.sigmoid()[:, None]
    cost = distance + 0.25 * size + 0.5 * heading + class_cost + object_cost
    rows, cols = _assign(cost.detach().cpu())
    for row, col in zip(rows, cols):
        if float(distance[row, col]) > dist_thresh:
            continue
        matched[row] = col
    return matched


def _assign(cost):
    from scipy.optimize import linear_sum_assignment

    rows, cols = linear_sum_assignment(cost.numpy())
    return rows, cols


def attach_future(matched, gt_xy, gt_labels, track, link_thresh=0.3):
    """Copy future waypoints onto matched queries.

    ``track`` rows are linked to the batch ground truth by class and center,
    so a range filter in the dataloader cannot swap identities.
    """

    count = matched.shape[0]
    steps = track["future"].shape[1] if len(track["future"]) else 12
    device = matched.device
    future = torch.zeros(count, steps, 2, device=device)
    mask = torch.zeros(count, steps, dtype=torch.bool, device=device)
    if len(track["current"]) == 0:
        return future, mask
    track_xy = torch.as_tensor(track["current"], device=device, dtype=torch.float32)
    track_label = torch.as_tensor(track["labels"], device=device)
    track_future = torch.as_tensor(track["future"], device=device, dtype=torch.float32)
    track_mask = torch.as_tensor(track["mask"], device=device) > 0
    for query_id, gt_id in enumerate(matched.tolist()):
        if gt_id < 0:
            continue
        same = track_label == gt_labels[gt_id]
        if not bool(same.any()):
            continue
        dist = torch.norm(track_xy[same, :] - gt_xy[gt_id].float(), dim=-1)
        local = int(dist.argmin())
        if float(dist[local]) > link_thresh:
            continue
        row = torch.arange(track_xy.shape[0], device=device)[same][local]
        future[query_id] = track_future[row]
        mask[query_id] = track_mask[row]
    return future, mask
