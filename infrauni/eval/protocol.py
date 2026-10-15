"""Joint detection-forecast metrics used by the paper tables.

Detection mAP and NDS are NOT computed here. They come from the nuScenes
devkit. This file scores the same predictions for forecasting:

* matching is class-wise bipartite matching on the current BEV center
* a pair closer than 2 m is a detection match
* minADE / minFDE / MR are averaged over those matches
* a match is an EPA hit only when minFDE at 6 s is within 2 m
* unmatched predictions are false positives
* EPA = (hits - 0.5 * false_positives) / ground_truths

Forecast queries are the detector's own boxes. Ground-truth boxes are never
used to spawn queries at inference.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment

from ..data.constants import CLASS_NAMES, EPA_ALPHA, EPA_FDE, MATCH_DIST


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
        name: {"hit": 0.0, "fp": 0.0, "gt": 0.0, "ade": 0.0, "fde": 0.0, "mr": 0.0, "n": 0.0}
        for name in CLASS_NAMES
    }
    for pred, gt in zip(predictions, ground_truth):
        traj = np.asarray(pred["trajectories"], dtype=np.float64)
        future_steps = traj.shape[2] - 1
        for class_id, name in enumerate(CLASS_NAMES):
            gt_keep = np.asarray(gt["labels"]) == class_id
            valid = np.asarray(gt["future_mask"])[:, -1] > 0 if len(gt["future_mask"]) else np.zeros((0,), dtype=bool)
            gt_keep = gt_keep & valid
            gt_xy = np.asarray(gt["current"])[gt_keep]
            gt_future = np.asarray(gt["future"])[gt_keep]
            pred_keep = np.asarray(pred["labels"]) == class_id
            pred_xy = traj[pred_keep, 0, 0, :]
            pred_future = traj[pred_keep, :, 1:, :]
            bucket = per_class[name]
            bucket["gt"] += len(gt_xy)
            rows, cols = _match(pred_xy, gt_xy)
            bucket["fp"] += len(pred_xy) - len(rows)
            for row, col in zip(rows, cols):
                error = np.linalg.norm(pred_future[row] - gt_future[col][None], axis=-1)
                ade = error.mean(axis=-1).min()
                fde = error[:, future_steps - 1].min()
                bucket["ade"] += ade
                bucket["fde"] += fde
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
        }
        if bucket["gt"] == 0:
            continue
        hits += bucket["hit"]
        fps += bucket["fp"]
        gts += bucket["gt"]
        ades += bucket["ade"]
        fdes += bucket["fde"]
        mrs += bucket["mr"]
        count += bucket["n"]
    present = [item["EPA"] for item in rows.values() if item["gt"] > 0]
    summary = {
        "EPA": float(np.mean(present)) if present else 0.0,
        "minADE": ades / count if count else None,
        "minFDE": fdes / count if count else None,
        "MR": mrs / count if count else None,
        "matched": count,
        "gt": gts,
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
