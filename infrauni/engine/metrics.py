"""Lightweight detection and forecasting metrics for the mini split.

These are not the official nuScenes numbers. Detection uses class-aware
center-distance matching. Forecasting reports minADE / minFDE on true positives.
"""

from __future__ import annotations

import numpy as np


def accumulate(pred, gt, dist_thresh, score_thresh):
    """pred/gt are lists of dicts with xy, label, score and optional traj."""
    tp = fp = fn = 0
    center_err = []
    ades, fdes = [], []
    for pred_i, gt_i in zip(pred, gt):
        keep = pred_i["score"] >= score_thresh
        order = np.argsort(-pred_i["score"][keep])
        pred_xy = pred_i["xy"][keep][order]
        pred_lab = pred_i["label"][keep][order]
        pred_traj = pred_i["traj"][keep][order] if pred_i.get("traj") is not None else None
        gt_xy = gt_i["xy"]
        gt_lab = gt_i["label"]
        pairs = []
        used = set()
        for pi in range(len(pred_xy)):
            best, best_d = None, dist_thresh
            for gi in range(len(gt_xy)):
                if gi in used or int(pred_lab[pi]) != int(gt_lab[gi]):
                    continue
                dist = float(np.linalg.norm(pred_xy[pi] - gt_xy[gi]))
                if dist < best_d:
                    best, best_d = gi, dist
            if best is None:
                fp += 1
                continue
            used.add(best)
            tp += 1
            center_err.append(best_d)
            pairs.append((pi, best))
        fn += int(len(gt_xy) - len(used))
        if pred_traj is None or gt_i.get("traj") is None:
            continue
        for pi, gi in pairs:
            mask = gt_i["traj_mask"][gi] > 0.5
            if mask.sum() < 1:
                continue
            this_traj = pred_traj[pi]
            mode_err = []
            final_err = []
            for mode in range(this_traj.shape[0]):
                step_err = np.linalg.norm(
                    this_traj[mode, mask] - gt_i["traj"][gi][mask], axis=-1
                )
                mode_err.append(float(step_err.mean()))
                final_err.append(float(step_err[-1]))
            ades.append(min(mode_err))
            fdes.append(min(final_err))
    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    f1 = 2 * prec * rec / max(prec + rec, 1e-8)
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "center_err": float(np.mean(center_err)) if center_err else None,
        "minADE": float(np.mean(ades)) if ades else None,
        "minFDE": float(np.mean(fdes)) if fdes else None,
        "n_traj": len(ades),
    }


def predictions_from_output(out, batch, score_thresh):
    """Decode one batch into numpy dicts. Uses the motion-refined boxes."""
    boxes = out["box"][-1].detach().cpu().numpy()
    cls = out["cls"][-1].detach().sigmoid().cpu().numpy()
    traj = out["traj"].detach().cpu().numpy()
    scores = cls.max(axis=-1)
    labels = cls.argmax(axis=-1)
    query_score = out["query_score"].detach().cpu().numpy()
    pred_list, gt_list = [], []
    gt_boxes = batch["boxes"].detach().cpu().numpy()
    gt_labels = batch["labels"].detach().cpu().numpy()
    gt_mask = batch["gt_mask"].detach().cpu().numpy()
    gt_traj = batch["traj"].detach().cpu().numpy()
    gt_traj_mask = batch["traj_mask"].detach().cpu().numpy()
    for b in range(boxes.shape[0]):
        keep = scores[b] >= score_thresh
        pred_list.append(
            {
                "xy": boxes[b, keep, :2],
                "xyz": boxes[b, keep, :3],
                "box": boxes[b, keep],
                "label": labels[b, keep],
                "score": scores[b, keep],
                "query_score": query_score[b, keep],
                "traj": traj[b, keep],
                "token": batch["tokens"][b],
            }
        )
        valid = gt_mask[b].astype(bool)
        gt_list.append(
            {
                "xy": gt_boxes[b, valid, :2],
                "box": gt_boxes[b, valid],
                "label": gt_labels[b, valid],
                "traj": gt_traj[b, valid],
                "traj_mask": gt_traj_mask[b, valid],
                "token": batch["tokens"][b],
            }
        )
    return pred_list, gt_list
