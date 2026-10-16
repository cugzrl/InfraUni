"""Detection and forecast losses for one TransFusion query set.

DeTra still stops pose gradients between refinement blocks. That is separate
from ``feedback``: when feedback is off, the forecast loss is not allowed to
update detection parameters or the shared refinement trunk.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from detra_repro.losses import sigmoid_focal_loss, wta_forecasting_loss

from .targets import attach_future, match_cost, match_queries

FORECAST_ONLY = ("future_gru", "future_head", "future_mode_head", "forecast.")


def _angle_delta(pred, target):
    return torch.atan2(torch.sin(pred - target), torch.cos(pred - target))


def detection_loss(boxes, class_logits, object_logits, matched, gt_boxes, gt_labels):
    """Full box regression plus class and objectness on matched queries.

    ``boxes`` is ``[N, 7+]`` in nuScenes lidar order ``x,y,z,w,l,h,yaw``.
    """

    targets = (matched >= 0).float()
    objectness = object_logits
    if objectness is None:
        cls = boxes.sum() * 0.0
        obj = cls
    else:
        obj = sigmoid_focal_loss(objectness, targets, reduction="mean")
        if class_logits is None or not bool(targets.any()):
            cls = objectness.sum() * 0.0
        else:
            cls = F.cross_entropy(class_logits[matched >= 0], gt_labels[matched[matched >= 0]])
    if not bool(targets.any()):
        zero = boxes.sum() * 0.0
        return {"loss_obj": obj, "loss_cls": cls, "loss_box": zero, "loss_det": obj + cls + zero}
    pred = boxes[matched >= 0, :7]
    gt = gt_boxes[matched[matched >= 0]]
    center = F.smooth_l1_loss(pred[:, :3], gt[:, :3])
    size = F.smooth_l1_loss(pred[:, 3:6].clamp_min(1e-2).log(), gt[:, 3:6].clamp_min(1e-2).log())
    yaw = F.smooth_l1_loss(_angle_delta(pred[:, 6], gt[:, 6]), torch.zeros_like(pred[:, 6]))
    box = center + size + yaw
    return {"loss_obj": obj, "loss_cls": cls, "loss_box": box, "loss_det": obj + cls + box}


def forecast_bundle(pred, future, mask, targets):
    """Winner-takes-all regression plus the mode-probability cross entropy."""

    if mask.dtype != torch.bool:
        mask = mask > 0
    # smooth_l1 keeps the winner-takes-all regression and the mode cross-entropy.
    # Laplace NLL is implemented in DeTra, but on this mini run it drove a box
    # coordinate non-finite in the second epoch.
    packed = wta_forecasting_loss(pred, future, mask, targets, loss_type="smooth_l1")
    return packed["loss_forecast"]


def _as_batch(future, mask, targets):
    return future[None], mask[None], targets[None]


def _reanchor(pred, origin_xy):
    """Express future waypoints as offsets from ``origin_xy`` ``[N, 2]``."""

    pred = dict(pred)
    current = pred["poses"][:, :, :, 0, :2].unsqueeze(3)
    offset = pred["future_xy"] - current
    pred["future_xy"] = origin_xy.detach()[None, :, None, None, :] + offset
    return pred


def _detach_forecast_origin(pred):
    """Keep forecast offsets, but stop them from dragging the current center."""

    pred = dict(pred)
    origin = pred["poses"][:, :, :, 0, :2].unsqueeze(3)
    pred["future_xy"] = pred["future_xy"] - origin + origin.detach()
    return pred


def _blend_box(refined, proposal, box_mode):
    """Apply the ablation's box update on top of the TransFusion proposal."""

    if box_mode == "none":
        return proposal[:, :7]
    box = proposal[:, :7].clone()
    if box_mode == "center":
        box[:, :2] = refined[:, :2]
        return box
    box[:, :7] = refined[:, :7]
    return box


def compute_joint_loss(output, gt_boxes, gt_labels, track, use_aux=True):
    """Match one sample and return the detection and forecast scalars.

    The assignment is computed once from the TransFusion proposal. Later
    refinement blocks keep that query-to-object pairing, so a box update cannot
    switch the future-trajectory target.
    """

    proposal = output["proposal_boxes"][0]
    labels = output["labels"][0]
    matched = match_queries(proposal[:, :2], labels, gt_boxes[:, :2], gt_labels)
    future, mask = attach_future(matched, gt_boxes[:, :2], gt_labels, track)
    targets = (matched >= 0).float()
    box_mode = output.get("box_mode", "full")
    detach_anchor = bool(output.get("detach_anchor", False))
    detra = output["detra"]
    if detra is None:
        pred = {
            "future_xy": output["trajectories"][:, :, :, 1:],
            "poses": output["trajectories"],
            "mode_logits": output["mode_logits"],
        }
        det = detection_loss(output["boxes"][0], None, None, matched, gt_boxes[:, :7], gt_labels)
        fore = forecast_bundle(pred, *_as_batch(future, mask, targets))
        return {
            "loss_det": det["loss_det"],
            "loss_box": det["loss_box"],
            "loss_forecast": fore,
            "matched": int(targets.sum()),
            "switch_rate": 0.0,
        }
    outputs = detra["aux_outputs"] if use_aux else [detra]
    det_total = proposal.sum() * 0.0
    fore_total = proposal.sum() * 0.0
    switched = compared = 0
    for block_pred in outputs:
        refined = _boxes_from_block(block_pred)
        with torch.no_grad():
            alt = match_queries(refined[:, :2], labels, gt_boxes[:, :2], gt_labels)
            both = (matched >= 0) & (alt >= 0)
            compared += int(both.sum())
            switched += int((matched[both] != alt[both]).sum()) if bool(both.any()) else 0
        if box_mode == "none":
            block_pred = _reanchor(block_pred, proposal[:, :2])
        elif detach_anchor:
            block_pred = _detach_forecast_origin(block_pred)
        class_logits = block_pred.get("class_logits")
        object_logits = block_pred.get("object_logits")
        if class_logits is not None and class_logits.ndim == 3:
            class_logits = class_logits[0]
        if object_logits is not None and object_logits.ndim == 2:
            object_logits = object_logits[0]
        if box_mode == "none":
            det_total = det_total + refined.sum() * 0.0
        else:
            det = detection_loss(
                _blend_box(refined, proposal, box_mode),
                class_logits,
                object_logits,
                matched,
                gt_boxes[:, :7],
                gt_labels,
            )
            det_total = det_total + det["loss_det"]
        fore_total = fore_total + forecast_bundle(
            block_pred, *_as_batch(future, mask, targets)
        )
    scale = float(len(outputs))
    return {
        "loss_det": det_total / scale,
        "loss_forecast": fore_total / scale,
        "matched": int(targets.sum()),
        "switch_rate": switched / compared if compared else 0.0,
    }


def _boxes_from_block(pred):
    """nuScenes ``w,l,h`` slots follow the proposal box, not DeTra's length name."""

    base = pred["base_boxes"]
    poses = pred["poses"]
    params = pred["box_params"]
    if base.ndim == 3:
        base, poses, params = base[0], poses[0], params[0]
    box = base.clone()
    box[:, 0] = poses[:, 0, 0, 0]
    box[:, 1] = poses[:, 0, 0, 1]
    box[:, 2] = base[:, 2] + params[:, 0]
    box[:, 3:6] = base[:, 3:6].clamp_min(1e-2) * params[:, 1:4].clamp(-4.0, 4.0).exp()
    box[:, 6] = base[:, 6] + params[:, 4]
    return box


def candidate_quality(proposal_xy, proposal_labels, final_xy, gt_xy, gt_labels, limit=2.0):
    """GT recall of the initial candidates, plus center error before and after DeTra."""

    if gt_xy.shape[0] == 0:
        return {"proposal_recall": 1.0, "proposal_center": 0.0, "final_center": 0.0}
    hits = 0
    proposal_err = []
    final_err = []
    for index in range(gt_xy.shape[0]):
        same = proposal_labels == gt_labels[index]
        if not bool(same.any()):
            continue
        distance = torch.linalg.norm(proposal_xy[same] - gt_xy[index], dim=-1)
        nearest = int(distance.argmin())
        chosen = torch.arange(proposal_xy.shape[0], device=proposal_xy.device)[same][nearest]
        if float(distance[nearest]) > limit:
            continue
        hits += 1
        proposal_err.append(distance[nearest])
        final_err.append(torch.linalg.norm(final_xy[chosen] - gt_xy[index], dim=-1))
    if not proposal_err:
        return {"proposal_recall": 0.0, "proposal_center": 0.0, "final_center": 0.0}
    return {
        "proposal_recall": hits / float(gt_xy.shape[0]),
        "proposal_center": float(torch.stack(proposal_err).mean()),
        "final_center": float(torch.stack(final_err).mean()),
    }


def compute_unified_loss(output, gt_boxes, gt_labels, track, gt_velocity=None):
    """Supervise DeTra's class, box, and forecast on one candidate assignment.

    The assignment uses the TransFusion candidate. Later blocks keep it.
    Forecast waypoints are offsets from a detached current center, so the
    forecast loss does not move that center. The detection loss still does.
    """

    proposal = output["proposal_boxes"][0]
    labels = output["proposal_labels"][0]
    gated = match_queries(proposal[:, :2], labels, gt_boxes[:, :2], gt_labels)
    if output.get("match_mode", "class_gate") == "cost":
        first = output["detra"]["aux_outputs"][0]
        class_logits = first["class_logits"]
        object_logits = first["object_logits"]
        if class_logits.ndim == 3:
            class_logits = class_logits[0]
        if object_logits.ndim == 2:
            object_logits = object_logits[0]
        matched = match_cost(
            proposal[:, :2],
            proposal[:, :7],
            class_logits,
            object_logits,
            gt_boxes[:, :7],
            gt_labels,
        )
    else:
        matched = gated
    future, mask = attach_future(matched, gt_boxes[:, :2], gt_labels, track)
    targets = (matched >= 0).float()
    outputs = output["detra"]["aux_outputs"]
    det_total = proposal.sum() * 0.0
    fore_total = proposal.sum() * 0.0
    for block_pred in outputs:
        refined = _boxes_from_block(block_pred)
        pred = _detach_forecast_origin(block_pred) if output.get("detach_anchor", True) else block_pred
        class_logits = block_pred["class_logits"]
        object_logits = block_pred["object_logits"]
        if class_logits.ndim == 3:
            class_logits = class_logits[0]
        if object_logits.ndim == 2:
            object_logits = object_logits[0]
        det = detection_loss(refined, class_logits, object_logits, matched, gt_boxes[:, :7], gt_labels)
        if gt_velocity is not None and "pred_velocity" in block_pred and bool(targets.any()):
            velocity = block_pred["pred_velocity"]
            if velocity.ndim == 3:
                velocity = velocity[0]
            assigned = matched >= 0
            gt_v = gt_velocity[matched[assigned]]
            finite = torch.isfinite(gt_v).all(dim=-1)
            if bool(finite.any()):
                det["loss_det"] = det["loss_det"] + F.smooth_l1_loss(
                    velocity[assigned][finite], gt_v[finite]
                )
        det_total = det_total + det["loss_det"]
        fore_total = fore_total + forecast_bundle(pred, *_as_batch(future, mask, targets))
    scale = float(len(outputs))
    weight = float(output.get("forecast_weight", 1.0))
    with torch.no_grad():
        quality = candidate_quality(
            proposal[:, :2],
            labels,
            output["boxes"][0, :, :2],
            gt_boxes[:, :2],
            gt_labels,
        )
    return {
        "loss_det": det_total / scale,
        "loss_forecast": fore_total / scale * weight,
        "matched": int(targets.sum()),
        "switch_rate": float((matched != gated).float().mean()) if matched.numel() else 0.0,
        "proposal_recall": quality["proposal_recall"],
        "proposal_center": quality["proposal_center"],
        "final_center": quality["final_center"],
    }


def apply_backward(model, losses, feedback):
    """Backprop detection and forecast losses.

    ``feedback=False`` restores every non-forecast parameter to the gradient
    it had from the detection loss alone. Forecast-only weights still train.
    """

    total_det = losses["loss_det"]
    total_fore = losses["loss_forecast"]
    if feedback:
        (total_det + total_fore).backward()
        return
    total_det.backward(retain_graph=True)
    saved = {}
    for name, param in model.named_parameters():
        if not param.requires_grad or any(token in name for token in FORECAST_ONLY):
            continue
        saved[name] = None if param.grad is None else param.grad.detach().clone()
    total_fore.backward()
    for name, param in model.named_parameters():
        if name in saved:
            param.grad = saved[name]
