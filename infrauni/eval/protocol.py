"""Joint detection-forecast metrics used by the paper tables.

Detection mAP and NDS are NOT computed here. They come from the nuScenes
devkit. This file scores the same predictions for forecasting:

* matching is class-wise bipartite matching on the current BEV center
* every current ground-truth box can absorb a prediction, including short tracks
* a prediction matched to a track that does not reach 6 s is ignored, not a false positive
* minADE / minFDE / MR use matches whose 6 s endpoint exists
* a match is an EPA hit only when minFDE at 6 s is within 2 m
* false positives are predictions matched to no current box
* EPA = (hits - 0.5 * false_positives) / ground_truths with a 6 s future
* the headline mean uses moving classes only

Forecast queries are the detector's own boxes. Ground-truth boxes are never
used to spawn queries at inference.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment

from ..data.constants import CLASS_NAMES, EPA_ALPHA, EPA_FDE, FORECAST_CLASSES, MATCH_DIST


def _match(pred_xy, gt_xy):
    if len(pred_xy) == 0 or len(gt_xy) == 0:
        return np.zeros((0,), dtype=np.int64), np.zeros((0,), dtype=np.int64)
    cost = np.linalg.norm(pred_xy[:, None, :] - gt_xy[None, :, :], axis=-1)
    rows, cols = linear_sum_assignment(cost)
    keep = cost[rows, cols] <= MATCH_DIST
    return rows[keep], cols[keep]


def accumulate_forecast(predictions, ground_truth):
    """Accumulate hits over a split.

    Each prediction item needs ``labels``, ``scores``, ``trajectories``
    with shape [K, modes, 1 + future, 2]. Each ground-truth item needs
    ``labels``, ``future`` [M, future, 2] and ``future_mask`` [M, future].
    """
    per_class = {
        name: {
            "hit": 0.0, "fp": 0.0, "gt": 0.0, "ade": 0.0, "fde": 0.0,
            "mr": 0.0, "n": 0.0, "pred": 0.0, "ignored": 0.0,
        }
        for name in CLASS_NAMES
    }
    for pred, gt in zip(predictions, ground_truth):
        traj = np.asarray(pred["trajectories"], dtype=np.float64)
        future_steps = traj.shape[2] - 1
        gt_mask = np.asarray(gt["future_mask"])
        if gt_mask.size == 0:
            gt_mask = np.zeros((0, future_steps))
        for class_id, name in enumerate(CLASS_NAMES):
            # Match every current box. A short track can absorb a detection so it
            # is not counted as a false positive. Forecast scores use the 6 s end.
            gt_keep = np.asarray(gt["labels"]) == class_id
            class_mask = gt_mask[gt_keep] if len(gt_mask) else np.zeros((0, future_steps))
            valid_end = class_mask[:, -1] > 0 if len(class_mask) else np.zeros((0,), dtype=bool)
            gt_xy = np.asarray(gt["current"])[gt_keep]
            gt_future = np.asarray(gt["future"])[gt_keep]
            pred_keep = np.asarray(pred["labels"]) == class_id
            pred_xy = traj[pred_keep, 0, 0, :]
            pred_future = traj[pred_keep, :, 1:, :]
            bucket = per_class[name]
            bucket["gt"] += int(valid_end.sum()) if len(valid_end) else 0
            bucket["pred"] += len(pred_xy)
            rows, cols = _match(pred_xy, gt_xy)
            bucket["fp"] += len(pred_xy) - len(rows)
            for row, col in zip(rows, cols):
                if not bool(valid_end[col]):
                    bucket["ignored"] += 1
                    continue
                error = np.linalg.norm(pred_future[row] - gt_future[col][None], axis=-1)
                valid = class_mask[col].astype(np.float64)
                ade = (error * valid).sum(axis=-1) / max(valid.sum(), 1.0)
                fde = error[:, future_steps - 1].min()
                bucket["ade"] += float(ade.min())
                bucket["fde"] += float(fde)
                bucket["mr"] += float(fde > EPA_FDE)
                bucket["n"] += 1
                if fde <= EPA_FDE:
                    bucket["hit"] += 1
    return per_class


def summarize_forecast(per_class):
    rows = {}
    hits = fps = gts = ades = fdes = mrs = count = 0.0
    for name, bucket in per_class.items():
        epa = (bucket["hit"] - EPA_ALPHA * bucket["fp"]) / bucket["gt"] if bucket["gt"] else 0.0
        denom = max(bucket["n"], 1.0)
        rows[name] = {
            "EPA": epa,
            "minADE": bucket["ade"] / denom if bucket["n"] else None,
            "minFDE": bucket["fde"] / denom if bucket["n"] else None,
            "MR": bucket["mr"] / denom if bucket["n"] else None,
            "gt": bucket["gt"],
            "matched": bucket["n"],
            "fp": bucket["fp"],
            "pred": bucket["pred"],
            "ignored": bucket["ignored"],
        }
        if name not in FORECAST_CLASSES or bucket["gt"] == 0:
            continue
        hits += bucket["hit"]
        fps += bucket["fp"]
        gts += bucket["gt"]
        ades += bucket["ade"]
        fdes += bucket["fde"]
        mrs += bucket["mr"]
        count += bucket["n"]
    present = [rows[name]["EPA"] for name in FORECAST_CLASSES if rows[name]["gt"] > 0]
    summary = {
        "EPA": float(np.mean(present)) if present else 0.0,
        "minADE": ades / count if count else None,
        "minFDE": fdes / count if count else None,
        "MR": mrs / count if count else None,
        "matched": count,
        "gt": gts,
        "fp": fps,
        "ignored": float(sum(bucket["ignored"] for bucket in per_class.values())),
        "per_class": rows,
    }
    return summary


def self_check():
    """Hand-computed cases for the EPA definition."""
    pred = {
        "labels": np.array([0, 0, 0]),
        "trajectories": np.zeros((3, 1, 3, 2)),
    }
    pred["trajectories"][0, 0, :, :] = (0, 0)
    pred["trajectories"][1, 0, 0] = (10, 0)
    pred["trajectories"][1, 0, 1:] = (10, 0)
    pred["trajectories"][2, 0, 0] = (3, 0)
    pred["trajectories"][2, 0, 1:] = (30, 0)
    gt = {
        "labels": np.array([0, 0]),
        "current": np.array([[0.0, 0.0], [3.0, 0.0]]),
        "future": np.zeros((2, 2, 2)),
        "future_mask": np.ones((2, 2)),
    }
    stats = accumulate_forecast([pred], [gt])
    car = stats["car"]
    # Two matches. The first is a hit. The second FDE is 30 m. One extra FP.
    assert car["hit"] == 1 and car["n"] == 2 and car["fp"] == 1 and car["gt"] == 2, car
    summary = summarize_forecast(stats)
    expected = (1 - 0.5 * 1) / 2
    assert abs(summary["EPA"] - expected) < 1e-6, summary["EPA"]
