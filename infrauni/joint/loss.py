"""Detection and forecast losses for one TransFusion query set.

DeTra still stops pose gradients between refinement blocks. That is separate
from ``feedback``: when feedback is off, the forecast loss is not allowed to
update detection parameters or the shared refinement trunk.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from detra_repro.losses import sigmoid_focal_loss, wta_forecasting_loss

from .targets import attach_future, match_queries

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


def compute_joint_loss(output, gt_boxes, gt_labels, track, use_aux=True):
    """Match one sample and return the detection and forecast scalars."""

    boxes = output["boxes"][0]
    labels = output["labels"][0]
    matched = match_queries(boxes[:, :2], labels, gt_boxes[:, :2], gt_labels)
    future, mask = attach_future(matched, gt_boxes[:, :2], gt_labels, track)
    targets = (matched >= 0).float()
    detra = output["detra"]
    if detra is None:
        pred = {
            "future_xy": output["trajectories"][:, :, :, 1:],
            "poses": output["trajectories"],
            "mode_logits": output["mode_logits"],
        }
        det = detection_loss(boxes, None, None, matched, gt_boxes[:, :7], gt_labels)
        fore = forecast_bundle(pred, *_as_batch(future, mask, targets))
        return {"loss_det": det["loss_det"], "loss_box": det["loss_box"], "loss_forecast": fore, "matched": int(targets.sum())}
    outputs = detra["aux_outputs"] if use_aux else [detra]
    det_total = boxes.sum() * 0.0
    fore_total = boxes.sum() * 0.0
    for block_pred in outputs:
        block_boxes = boxes.new_zeros(boxes.shape)
        # Each block stores its own refined box on the same query index.
        refined = _boxes_from_block(block_pred)
        block_boxes[:, :7] = refined
        class_logits = block_pred.get("class_logits")
        object_logits = block_pred.get("object_logits")
        if class_logits is not None and class_logits.ndim == 3:
            class_logits = class_logits[0]
        if object_logits is not None and object_logits.ndim == 2:
            object_logits = object_logits[0]
        det = detection_loss(
            block_boxes,
            class_logits,
            object_logits,
            match_queries(block_boxes[:, :2], labels, gt_boxes[:, :2], gt_labels),
            gt_boxes[:, :7],
            gt_labels,
        )
        block_future, block_mask = attach_future(
            match_queries(block_boxes[:, :2], labels, gt_boxes[:, :2], gt_labels),
            gt_boxes[:, :2],
            gt_labels,
            track,
        )
        block_targets = (block_mask.any(dim=-1)).float()
        fore_total = fore_total + forecast_bundle(
            block_pred, *_as_batch(block_future, block_mask, block_targets)
        )
        det_total = det_total + det["loss_det"]
    scale = float(len(outputs))
    return {
        "loss_det": det_total / scale,
        "loss_forecast": fore_total / scale,
        "matched": int(targets.sum()),
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
