"""Short nuScenes-mini run of TransFusion-LC with a forecast head or DeTra."""

import argparse
import logging
import os
import sys

import torch

logging.getLogger("mmcv").setLevel(logging.ERROR)

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "third_party", "MC-DeTra"))
sys.path.insert(0, os.path.join(ROOT, "third_party", "TransFusion"))

from mmcv import Config
from mmcv.parallel import collate, scatter
from mmcv.runner import load_checkpoint
from mmdet3d.datasets import build_dataset
from mmdet3d.models import build_detector

from detra_repro.losses import wta_forecasting_loss
from infrauni.joint.model import JointTransFusionDeTra
from infrauni.joint.targets import align_future


def _batch(dataset, index):
    batch = collate([dataset[index]], samples_per_gpu=1)
    if torch.cuda.is_available():
        batch = scatter(batch, [0])[0]
    # mmcv collate wraps the single sample one level deeper than the detector expects.
    return {
        "points": batch["points"][0],
        "img": batch["img"][0],
        "img_metas": batch["img_metas"][0],
    }


def _move_detector(model):
    if torch.cuda.is_available():
        model.cuda()
    return model


def _forecast_pred(output):
    if output["detra"] is not None:
        return output["detra"]
    traj = output["trajectories"]
    return {
        "future_xy": traj[:, :, :, 1:],
        "poses": traj,
        "mode_logits": traj.new_zeros(traj.shape[0], traj.shape[1], traj.shape[2]),
    }


def _loss(output, track):
    future, mask, targets, matched = align_future(output["boxes"][0, :, :2], track)
    forecast = wta_forecasting_loss(_forecast_pred(output), future, mask, targets, loss_type="smooth_l1")
    center = output["boxes"][0, :, :2]
    rows = [index for index, gt_index in enumerate(matched) if gt_index >= 0]
    if rows:
        gt_rows = torch.as_tensor(track["current"][matched[rows]], device=center.device, dtype=center.dtype)
        det = torch.nn.functional.smooth_l1_loss(center[rows], gt_rows)
    else:
        det = center.sum() * 0
    return forecast["loss_forecast"] + det, forecast["loss_forecast"].detach(), det.detach(), int(targets.sum())


def _grad_report(model, batch, track):
    # The detector stays frozen during this probe's forward unless the mode
    # needs a connectivity check. Unfreezing the whole LC graph is done only
    # after a frozen forward has succeeded.
    model.zero_grad(set_to_none=True)
    output = model(points=batch["points"], img=batch["img"], img_metas=batch["img_metas"])
    loss, _, _, _ = _loss(output, track)
    loss.backward()

    def has_grad(module):
        if module is None:
            return False
        return any(p.grad is not None and float(p.grad.abs().sum()) > 0 for p in module.parameters())

    report = {
        "loss": float(loss.detach()),
        "query_proj": has_grad(model.query_proj),
        "bev_proj": has_grad(model.bev_proj),
        "forecast_head": has_grad(model.forecast),
        "detra_box": has_grad(model.detra.pose_update if model.detra is not None else None),
        "detra_lidar": has_grad(model.detra.blocks[0].lidar_attn if model.detra is not None else None),
        "detector_head": has_grad(model.detector.pts_bbox_head),
        "detector_lidar": has_grad(model.detector.pts_backbone),
    }
    model.zero_grad(set_to_none=True)
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["pred_head", "joint"], required=True)
    parser.add_argument("--steps", type=int, default=40)
    parser.add_argument("--blocks", type=int, default=2)
    args = parser.parse_args()
    logging.getLogger("mmcv").setLevel(logging.ERROR)
    os.chdir(os.path.join(ROOT, "third_party", "TransFusion"))
    cfg = Config.fromfile(os.path.join(ROOT, "configs", "transfusion_mini_voxel_LC.py"))
    train_cfg = cfg.data.val.copy()
    train_cfg["ann_file"] = cfg.data.train["dataset"]["ann_file"]
    train_cfg["test_mode"] = True
    dataset = build_dataset(train_cfg)
    detector = build_detector(cfg.model, train_cfg=cfg.get("train_cfg"), test_cfg=cfg.get("test_cfg"))
    ckpt = os.path.join(ROOT, "experiments", "transfusion_voxel_LC_mini", "epoch_1.pth")
    load_checkpoint(detector, ckpt, map_location="cpu")
    model = _move_detector(JointTransFusionDeTra(detector, mode=args.mode, blocks=args.blocks))
    import pickle

    tracks = pickle.load(open(os.path.join(ROOT, "data", "nuscenes_mini_future12.pkl"), "rb"))
    model.freeze_detector()
    batch = _batch(dataset, 0)
    token = batch["img_metas"][0]["sample_idx"]
    report = _grad_report(model, batch, tracks[token])
    if args.mode == "joint":
        for param in model.detector.pts_bbox_head.parameters():
            param.requires_grad_(True)
            break
        try:
            connected = _grad_report(model, batch, tracks[token])
            report["detector_head_connected"] = connected["detector_head"]
        except RuntimeError as exc:
            report["detector_head_connected"] = "oom: %s" % exc
        model.freeze_detector()
    print("grad", report)
    trainable = [param for param in model.parameters() if param.requires_grad]
    optim = torch.optim.AdamW(trainable, lr=1e-4, weight_decay=0.01)
    model.train()
    out_dir = os.path.join(ROOT, "experiments", "joint_%s_mini" % args.mode)
    os.makedirs(out_dir, exist_ok=True)
    for step in range(1, args.steps + 1):
        batch = _batch(dataset, (step - 1) % len(dataset))
        token = batch["img_metas"][0]["sample_idx"]
        optim.zero_grad(set_to_none=True)
        output = model(points=batch["points"], img=batch["img"], img_metas=batch["img_metas"])
        loss, forecast, det, matched = _loss(output, tracks[token])
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, 5.0)
        optim.step()
        if step == 1 or step % 10 == 0 or step == args.steps:
            print(
                "step", step,
                "loss", round(float(loss), 4),
                "forecast", round(float(forecast), 4),
                "det", round(float(det), 4),
                "matched", matched,
            )
    torch.save(
        {"mode": args.mode, "blocks": args.blocks, "state_dict": model.state_dict(), "grad": report},
        os.path.join(out_dir, "last.pt"),
    )
    print("saved", os.path.join(out_dir, "last.pt"))


if __name__ == "__main__":
    main()
