"""Gradient alignment between detection and forecast on a few mini frames.

Compares three mechanisms on the same C2 weights:
the current joint backward, detaching the box center from the forecast loss,
and stopping future queries from reading into the current detection query.
"""

import os
import pickle
import sys

import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from tools.train_joint import (  # noqa: E402
    LC_CHECKPOINT,
    ROOT,
    _collate,
    EpochOrder,
)

import logging
logging.disable(logging.INFO)

from mmcv import Config
from mmcv.runner import load_checkpoint
from mmdet3d.datasets import build_dataset
from mmdet3d.models import build_detector
from torch.utils.data import DataLoader

from infrauni.joint.model import JointTransFusionDeTra


TOKENS = ("box_head", "time_attn", "decoder.1", "prediction_heads", "bev_proj", "query_proj")


def _flat(model, token):
    parts = [
        param.grad.detach().flatten()
        for name, param in model.named_parameters()
        if token in name and param.grad is not None
    ]
    if not parts:
        return None
    return torch.cat(parts)


def _cosine(left, right):
    if left is None or right is None or left.numel() != right.numel():
        return None
    return float(torch.dot(left, right) / (left.norm() * right.norm()).clamp_min(1e-12))


def _grads(model, batch, tracks):
    model.zero_grad(set_to_none=True)
    output = model(points=batch["points"], img=batch["img"], img_metas=batch["img_metas"])
    from infrauni.joint.loss import compute_joint_loss
    gt_boxes = batch["gt_bboxes_3d"][0].tensor.to(batch["img"].device)
    gt_labels = batch["gt_labels_3d"][0].to(gt_boxes.device).long()
    token = batch["img_metas"][0]["sample_idx"]
    losses = compute_joint_loss(output, gt_boxes[:, :7], gt_labels, tracks[token])
    losses["loss_det"].backward(retain_graph=True)
    det = {name: _flat(model, name) for name in TOKENS}
    model.zero_grad(set_to_none=True)
    losses["loss_forecast"].backward()
    fore = {name: _flat(model, name) for name in TOKENS}
    return {
        name: _cosine(det[name], fore[name])
        for name in TOKENS
    }, float(losses["switch_rate"]), float(losses["loss_det"].detach()), float(losses["loss_forecast"].detach())


def main():
    os.chdir(os.path.join(ROOT, "third_party", "TransFusion"))
    cfg = Config.fromfile(os.path.join(ROOT, "configs", "transfusion_mini_voxel_LC.py"))
    train_cfg = cfg.data.train["dataset"]
    dataset = build_dataset(train_cfg)
    detector = build_detector(cfg.model, train_cfg=cfg.get("train_cfg"), test_cfg=cfg.get("test_cfg"))
    load_checkpoint(detector, LC_CHECKPOINT, map_location="cpu")
    model = JointTransFusionDeTra(detector, mode="joint", blocks=2, box_mode="full")
    saved = torch.load(os.path.join(ROOT, "experiments", "ablation_c2_mini", "epoch_2.pt"), map_location="cpu")
    current = model.state_dict()
    model.load_state_dict(
        {key: value for key, value in saved["state_dict"].items() if key in current and current[key].shape == value.shape},
        strict=False,
    )
    model.freeze_detector()
    model.cuda()
    model.train()
    model.unfreeze_fusion()
    tracks = pickle.load(open(os.path.join(ROOT, "data", "nuscenes_mini_future12.pkl"), "rb"))
    order = list(range(4))
    loader = DataLoader(dataset, batch_size=1, sampler=EpochOrder(order), num_workers=0, collate_fn=_collate)
    settings = []
    batches = list(loader)
    for name, detach, causal in (("joint", False, False), ("detach_center", True, False), ("no_forward", False, True)):
        model.detach_anchor = detach
        for block in model.detra.blocks:
            block.causal_time = causal
        cosines = []
        for batch in batches:
            cosine, switch, det, fore = _grads(model, batch, tracks)
            cosines.append(cosine)
            print(name, "switch", round(switch, 3), "det", round(det, 3), "fore", round(fore, 3), cosine, flush=True)
        settings.append((name, cosines))
    print("done", flush=True)


if __name__ == "__main__":
    main()
