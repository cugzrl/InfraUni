"""Keep detection boxes and trajectories on the same query index."""

from __future__ import annotations

import torch


def filter_query_indices(scores, boxes, score_thresh=0.0, center_range=None, nms_radius=None):
    """Return query indices after score, range, and optional radius suppression.

    ``scores`` is ``[N]`` and ``boxes`` is ``[N, >=3]``. The result indexes the
    original query axis. It is never a nearest-neighbor lookup.
    """

    if scores.ndim != 1 or boxes.shape[0] != scores.shape[0]:
        raise ValueError("scores and boxes must share the query axis")
    keep = torch.ones(scores.shape[0], dtype=torch.bool, device=scores.device)
    if score_thresh is not None:
        keep = keep & (scores >= score_thresh)
    if center_range is not None:
        limit = torch.as_tensor(center_range, device=boxes.device, dtype=boxes.dtype)
        center = boxes[:, :3]
        keep = keep & (center >= limit[:3]).all(dim=-1) & (center <= limit[3:]).all(dim=-1)
    index = torch.arange(scores.shape[0], device=scores.device)[keep]
    if nms_radius is None or index.numel() == 0:
        return index
    order = torch.argsort(scores[index], descending=True)
    chosen = []
    for query_id in index[order].tolist():
        if all(
            torch.norm(boxes[query_id, :2] - boxes[kept, :2]).item() > nms_radius
            for kept in chosen
        ):
            chosen.append(query_id)
    return torch.tensor(chosen, dtype=torch.long, device=scores.device)


def take_queries(index, **named):
    """Slice each ``[N, ...]`` tensor with the same query index."""

    return {name: value[index] for name, value in named.items()}
