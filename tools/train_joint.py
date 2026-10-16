"""Train ablation variants B, C1, C2, and C3 on nuScenes-mini.

Every variant starts from the same TransFusion-LC checkpoint and sees the same
shuffled frame order. C2 and C3 share the architecture and the trainable set.
C3 keeps the forecast loss off the detection parameters.
"""

import argparse
import logging
import os
import pickle
import random
import sys
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, Sampler

logging.getLogger("mmcv").setLevel(logging.ERROR)

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "third_party", "MC-DeTra"))
sys.path.insert(0, os.path.join(ROOT, "third_party", "TransFusion"))

from mmcv import Config
from mmcv.parallel import DataContainer, collate, scatter
from mmcv.runner import load_checkpoint
from mmdet3d.datasets import build_dataset
from mmdet3d.models import build_detector

from infrauni.joint.loss import apply_backward, compute_joint_loss, compute_unified_loss
from infrauni.joint.model import JointTransFusionDeTra
from infrauni.joint.unified import UnifiedTransFusionDeTra
from infrauni.joint.poses import load_pose_index
from infrauni.joint.stability import (
    backup_trainable,
    first_nonfinite_grad,
    first_nonfinite_param,
    grad_norms,
    optimizer_to,
    restore_trainable,
    write_failure,
)

logging.getLogger("mmcv").setLevel(logging.ERROR)

LC_CHECKPOINT = os.path.join(ROOT, "experiments", "transfusion_voxel_LC_mini", "epoch_1.pth")
VARIANTS = {
    "b": {"mode": "pred_head", "unfreeze": False, "feedback": True, "box_mode": "full", "detach_anchor": False, "causal_time": False},
    "c1": {"mode": "joint", "unfreeze": False, "feedback": True, "box_mode": "full", "detach_anchor": False, "causal_time": False},
    "c2": {"mode": "joint", "unfreeze": True, "feedback": True, "box_mode": "full", "detach_anchor": False, "causal_time": False},
    "c3": {"mode": "joint", "unfreeze": True, "feedback": False, "box_mode": "full", "detach_anchor": False, "causal_time": False},
    # DeTra forecast, exported boxes stay the TransFusion proposals.
    "d1": {"mode": "joint", "unfreeze": False, "feedback": True, "box_mode": "none", "detach_anchor": True, "causal_time": False},
    # Only the current center is refined.
    "d2": {"mode": "joint", "unfreeze": False, "feedback": True, "box_mode": "center", "detach_anchor": False, "causal_time": False},
    # Full box refinement, detector frozen. Fixed query assignment.
    "d3": {"mode": "joint", "unfreeze": False, "feedback": True, "box_mode": "full", "detach_anchor": False, "causal_time": False},
    # Full box plus unfrozen fusion features. Forecast loss can move the center.
    "d4": {"mode": "joint", "unfreeze": True, "feedback": True, "box_mode": "full", "detach_anchor": False, "causal_time": False},
    # Shared features stay coupled, but the forecast loss does not drag the center.
    "d5": {"mode": "joint", "unfreeze": True, "feedback": True, "box_mode": "full", "detach_anchor": True, "causal_time": False},
    # Future queries cannot update the current detection query. Backward grads stay on.
    "f0": {"mode": "joint", "unfreeze": True, "feedback": True, "box_mode": "full", "detach_anchor": False, "causal_time": True, "memory": "none"},
    # Streaming priors. The detector stays frozen. Only the residual prior is new.
    "s_det": {"mode": "joint", "unfreeze": False, "feedback": True, "box_mode": "none", "detach_anchor": True, "causal_time": False, "memory": "det"},
    "s_forecast": {"mode": "joint", "unfreeze": False, "feedback": True, "box_mode": "none", "detach_anchor": True, "causal_time": False, "memory": "forecast"},
    "s_both": {"mode": "joint", "unfreeze": False, "feedback": True, "box_mode": "none", "detach_anchor": True, "causal_time": False, "memory": "both"},
    "s_reliable": {"mode": "joint", "unfreeze": False, "feedback": True, "box_mode": "none", "detach_anchor": True, "causal_time": False, "memory": "reliable"},
    # Unified DeTra decoder. TransFusion boxes are candidates only.
    "u0": {"arch": "unified", "mode": "joint", "unfreeze": False, "feedback": True, "box_mode": "full", "detach_anchor": True, "causal_time": False, "forecast_weight": 1.0},
    "u_nofore": {"arch": "unified", "mode": "joint", "unfreeze": False, "feedback": True, "box_mode": "full", "detach_anchor": True, "causal_time": False, "forecast_weight": 0.0},
    "u_causal": {"arch": "unified", "mode": "joint", "unfreeze": False, "feedback": True, "box_mode": "full", "detach_anchor": True, "causal_time": True, "forecast_weight": 1.0},
    "u_cost": {"arch": "unified", "mode": "joint", "unfreeze": False, "feedback": True, "box_mode": "full", "detach_anchor": True, "causal_time": False, "forecast_weight": 1.0, "match_mode": "cost"},
}

for _spec in VARIANTS.values():
    _spec.setdefault("memory", "none")
    _spec.setdefault("arch", "series")
    _spec.setdefault("forecast_weight", 1.0)
    _spec.setdefault("match_mode", "class_gate")


class EpochOrder(Sampler):
    """Yield one fixed permutation. Variants that share the seed match."""

    def __init__(self, indices):
        self.indices = [int(index) for index in indices]

    def __iter__(self):
        return iter(self.indices)

    def __len__(self):
        return len(self.indices)


def _unwrap(value):
    if isinstance(value, DataContainer):
        return value.data
    return value


def _points(value):
    """Return a list of ``[P, C]`` tensors, which is what voxelize iterates."""

    value = _unwrap(value)
    if torch.is_tensor(value):
        if value.ndim == 2:
            return [value]
        if value.ndim == 3:
            return list(value)
    if isinstance(value, list):
        if len(value) == 1 and isinstance(value[0], list):
            value = value[0]
        if value and torch.is_tensor(value[0]) and value[0].ndim == 2:
            return value
        if value and torch.is_tensor(value[0]) and value[0].ndim == 3:
            return list(value[0])
    raise TypeError("unrecognized points: %s" % type(value))


def _prepare(batch):
    points = _points(batch["points"])
    img = _unwrap(batch["img"])
    metas = _unwrap(batch["img_metas"])
    if isinstance(img, list):
        img = img[0]
    if isinstance(metas, dict):
        metas = [metas]
    elif isinstance(metas, list) and metas and isinstance(metas[0], list):
        metas = metas[0]
    prepared = {"points": points, "img": img, "img_metas": metas}
    if "gt_bboxes_3d" not in batch:
        return prepared
    boxes = _unwrap(batch["gt_bboxes_3d"])
    labels = _unwrap(batch["gt_labels_3d"])
    if isinstance(boxes, list) and boxes and not hasattr(boxes[0], "tensor"):
        boxes = boxes[0]
    if not isinstance(boxes, list):
        boxes = [boxes]
    if isinstance(labels, list) and labels and torch.is_tensor(labels[0]):
        pass
    elif torch.is_tensor(labels):
        labels = [labels]
    elif isinstance(labels, list) and labels and isinstance(labels[0], list):
        labels = labels[0]
    prepared["gt_bboxes_3d"] = boxes
    prepared["gt_labels_3d"] = labels
    return prepared


def _collate(samples):
    batch = collate(samples, samples_per_gpu=len(samples))
    if torch.cuda.is_available():
        batch = scatter(batch, [torch.cuda.current_device()])[0]
    return _prepare(batch)


def _finite_tree(value):
    if torch.is_tensor(value):
        return bool(torch.isfinite(value).all())
    if isinstance(value, dict):
        return all(_finite_tree(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(_finite_tree(item) for item in value)
    return True


def _official_loss(model, output, batch):
    raw = model.detector.pts_bbox_head.loss(
        batch["gt_bboxes_3d"], batch["gt_labels_3d"], output["outs"]
    )
    terms = [value for key, value in raw.items() if "loss" in key and torch.is_tensor(value)]
    if not terms:
        return output["boxes"].sum() * 0.0
    return sum(terms)


def _rng_state():
    return {
        "torch": torch.get_rng_state(),
        "numpy": np.random.get_state(),
        "python": random.getstate(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def _halt(log, work, reason, epoch, step, detail, model, optimizer, losses):
    """Record the first bad step and leave the weights unchanged."""

    payload = {
        "reason": reason,
        "epoch": epoch,
        "step": step,
        "detail": str(detail),
        "parameter": first_nonfinite_param(model),
        "gradient": first_nonfinite_grad(model),
        "grad_norms": grad_norms(
            model, ("decoder.1", "prediction_heads", "box_head", "future_head", "future_gru", "prior.")
        ),
        "learning_rates": [group["lr"] for group in optimizer.param_groups],
        "losses": {},
    }
    if losses is not None:
        for key, value in losses.items():
            if torch.is_tensor(value):
                payload["losses"][key] = None if not torch.isfinite(value) else float(value.detach())
            else:
                payload["losses"][key] = value
    path = os.path.join(work, "nonfinite.json")
    write_failure(path, payload)
    message = "halt %s epoch %d step %d grad %s param %s" % (
        reason, epoch, step, payload["gradient"], payload["parameter"]
    )
    print(message, payload["grad_norms"], payload["learning_rates"], flush=True)
    log.write(message + "\n")
    log.flush()
    raise SystemExit(2)


def _trainable(model):
    return sum(param.numel() for param in model.parameters() if param.requires_grad)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", required=True, choices=sorted(VARIANTS))
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--max-steps", type=int, default=0)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--eval-every", type=int, default=0)
    parser.add_argument("--resume", default="")
    parser.add_argument("--detector-lr", type=float, default=1e-6)
    parser.add_argument("--detector-clip", type=float, default=0.5)
    parser.add_argument("--module-lr", type=float, default=1e-4)
    parser.add_argument("--ordered", action="store_true")
    parser.add_argument("--work-dir", default="")
    parser.add_argument("--stop-epoch", type=int, default=0)
    args = parser.parse_args()
    logging.disable(logging.INFO)
    spec = VARIANTS[args.variant]
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    os.chdir(os.path.join(ROOT, "third_party", "TransFusion"))
    cfg = Config.fromfile(os.path.join(ROOT, "configs", "transfusion_mini_voxel_LC.py"))
    train_cfg = cfg.data.train
    if train_cfg["type"] == "CBGSDataset":
        train_cfg = train_cfg["dataset"]
    dataset = build_dataset(train_cfg)
    detector = build_detector(cfg.model, train_cfg=cfg.get("train_cfg"), test_cfg=cfg.get("test_cfg"))
    load_checkpoint(detector, LC_CHECKPOINT, map_location="cpu")
    if spec["arch"] == "unified":
        model = UnifiedTransFusionDeTra(
            detector,
            blocks=2,
            detach_anchor=spec["detach_anchor"],
            causal_time=spec["causal_time"],
            forecast_weight=spec["forecast_weight"],
            match_mode=spec["match_mode"],
        )
    else:
        model = JointTransFusionDeTra(
            detector,
            mode=spec["mode"],
            blocks=2,
            box_mode=spec["box_mode"],
            detach_anchor=spec["detach_anchor"],
            causal_time=spec["causal_time"],
            memory_mode=spec["memory"],
        )
    model.freeze_detector()
    start_epoch = 0
    if args.resume:
        resume_path = args.resume if os.path.isabs(args.resume) else os.path.join(ROOT, args.resume)
        resumed = torch.load(resume_path, map_location="cpu")
        model.load_state_dict(resumed["state_dict"])
        start_epoch = int(resumed.get("epoch", 0))
    if torch.cuda.is_available():
        model.cuda()
    model.train()
    if spec["unfreeze"]:
        model.unfreeze_fusion()
    groups = []
    detector_params = []
    new_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if name.startswith("detector."):
            detector_params.append(param)
        else:
            new_params.append(param)
    if new_params:
        groups.append({"params": new_params, "lr": args.module_lr, "initial_lr": args.module_lr})
    if detector_params:
        groups.append({"params": detector_params, "lr": args.detector_lr, "initial_lr": args.detector_lr})
    optimizer = torch.optim.AdamW(groups, weight_decay=0.01)
    if args.resume and "optimizer" in resumed:
        optimizer.load_state_dict(resumed["optimizer"])
        if torch.cuda.is_available():
            optimizer_to(optimizer, "cuda")
    work = args.work_dir or os.path.join(ROOT, "experiments", "ablation_%s_mini" % args.variant)
    if not os.path.isabs(work):
        work = os.path.join(ROOT, work)
    os.makedirs(work, exist_ok=True)
    tracks = pickle.load(open(os.path.join(ROOT, "data", "nuscenes_mini_future12.pkl"), "rb"))
    log = open(os.path.join(work, "train.log"), "a")
    print(
        "variant", args.variant,
        "mode", spec["mode"],
        "unfreeze", spec["unfreeze"],
        "feedback", spec["feedback"],
        "start_epoch", start_epoch,
        "trainable", _trainable(model),
        "frames", len(dataset),
        flush=True,
    )
    log.write("trainable %d frames %d\n" % (_trainable(model), len(dataset)))
    info_dir = os.path.join(ROOT, "third_party", "TransFusion", "data", "nuscenes")
    poses = load_pose_index(
        [
            os.path.join(info_dir, "nuscenes_infos_train.pkl"),
            os.path.join(info_dir, "nuscenes_infos_val.pkl"),
        ],
        sample_table="/home/kemove/devdata1/zrl/bevfusion/data/nuscenes/v1.0-mini/sample.json",
    ) if spec["memory"] != "none" else None
    if args.resume and resumed.get("rng"):
        torch.set_rng_state(resumed["rng"]["torch"])
        np.random.set_state(resumed["rng"]["numpy"])
        random.setstate(resumed["rng"]["python"])
        if resumed["rng"].get("cuda") and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(resumed["rng"]["cuda"])
    started = time.perf_counter()
    global_step = int(resumed["global_step"]) if args.resume and "global_step" in resumed else start_epoch * len(dataset)
    for epoch in range(start_epoch, args.epochs):
        if args.ordered:
            order = np.arange(len(dataset))
        else:
            order = np.random.RandomState(args.seed + epoch).permutation(len(dataset))
        if args.max_steps:
            order = order[: args.max_steps]
        loader = DataLoader(
            dataset,
            batch_size=1,
            sampler=EpochOrder(order),
            num_workers=args.workers,
            collate_fn=_collate,
        )
        total_steps = args.epochs * len(dataset) if args.max_steps <= 0 else args.epochs * args.max_steps
        for batch in loader:
            global_step += 1
            progress = min(global_step / max(total_steps, 1), 1.0)
            for group in optimizer.param_groups:
                base = group["initial_lr"]
                group["lr"] = base * (1.0 - progress) + 1e-6 * progress
            token = batch["img_metas"][0]["sample_idx"]
            full_gt = batch["gt_bboxes_3d"][0].tensor.to(batch["img"].device)
            gt_boxes = full_gt[:, :7]
            gt_velocity = full_gt[:, 7:9] if full_gt.shape[-1] >= 9 else None
            gt_labels = batch["gt_labels_3d"][0].to(gt_boxes.device).long()
            optimizer.zero_grad(set_to_none=True)
            if poses is not None:
                meta = poses.get(token)
                if meta is None:
                    model.set_frame(token, 0.0, np.eye(4))
                else:
                    model.set_frame(meta["scene"], meta["timestamp"], meta["lidar_to_global"])
            try:
                output = model(points=batch["points"], img=batch["img"], img_metas=batch["img_metas"])
            except RuntimeError as exc:
                if "out of memory" not in str(exc):
                    raise
                torch.cuda.empty_cache()
                try:
                    output = model(points=batch["points"], img=batch["img"], img_metas=batch["img_metas"])
                except RuntimeError as again:
                    _halt(log, work, "cuda_oom", epoch, global_step, again, model, optimizer, None)
                    return
            if global_step == 1:
                same = torch.allclose(
                    output["boxes"][0, :, :2],
                    output["trajectories"][0, :, 0, 0, :],
                    atol=1e-4,
                )
                print("current_box_matches_trajectory", bool(same), "queries", output["boxes"].shape[1], flush=True)
            if spec["arch"] == "unified":
                losses = compute_unified_loss(
                    output, gt_boxes[:, :7], gt_labels, tracks[token], gt_velocity
                )
            else:
                losses = compute_joint_loss(output, gt_boxes[:, :7], gt_labels, tracks[token])
            use_official = spec["arch"] != "unified" and (spec["unfreeze"] or spec["memory"] != "none")
            if use_official:
                if not _finite_tree(output["outs"]):
                    _halt(log, work, "nonfinite_outputs", epoch, global_step, "detection outputs", model, optimizer, losses)
                    return
                try:
                    official = _official_loss(model, output, batch)
                except ValueError as exc:
                    _halt(log, work, "official_assigner", epoch, global_step, exc, model, optimizer, losses)
                    return
                if not torch.isfinite(official):
                    _halt(log, work, "nonfinite_official_loss", epoch, global_step, official, model, optimizer, losses)
                    return
                losses["loss_det"] = losses["loss_det"] + official
                losses["loss_official"] = official.detach()
            if not (torch.isfinite(losses["loss_det"]) and torch.isfinite(losses["loss_forecast"])):
                _halt(log, work, "nonfinite_loss", epoch, global_step, "joint loss", model, optimizer, losses)
                return
            if spec["arch"] == "unified" and global_step == 1:
                probe_tokens = ("box_head", "class_head", "object_head", "query_proj", "time_attn", "future_head")
                losses["loss_det"].backward(retain_graph=True)
                det_probe = grad_norms(model, probe_tokens)
                optimizer.zero_grad(set_to_none=True)
                losses["loss_forecast"].backward(retain_graph=True)
                fore_probe = grad_norms(model, probe_tokens)
                optimizer.zero_grad(set_to_none=True)
                print("det_grads", {key: round(value, 5) for key, value in det_probe.items()}, flush=True)
                print("forecast_grads", {key: round(value, 5) for key, value in fore_probe.items()}, flush=True)
                log.write("det_grads %s\nforecast_grads %s\n" % (det_probe, fore_probe))
            saved = backup_trainable(model)
            apply_backward(model, losses, feedback=spec["feedback"])
            bad_grad = first_nonfinite_grad(model)
            if bad_grad is not None:
                restore_trainable(model, saved)
                _halt(log, work, "nonfinite_gradient", epoch, global_step, bad_grad, model, optimizer, losses)
                return
            detector_grads = []
            module_grads = []
            for name, param in model.named_parameters():
                if param.grad is None:
                    continue
                if name.startswith("detector."):
                    detector_grads.append(param)
                else:
                    module_grads.append(param)
            if detector_grads:
                torch.nn.utils.clip_grad_norm_(detector_grads, args.detector_clip)
            if module_grads:
                torch.nn.utils.clip_grad_norm_(module_grads, 1.0)
            optimizer.step()
            bad_param = first_nonfinite_param(model)
            if bad_param is not None:
                restore_trainable(model, saved)
                _halt(log, work, "nonfinite_parameter", epoch, global_step, bad_param, model, optimizer, losses)
                return
            if spec["memory"] != "none":
                model.store_frame(
                    output["boxes"][0],
                    output["scores"][0],
                    output["labels"][0],
                    output["trajectories"][0],
                    output["query"][0].transpose(0, 1),
                )
            if global_step == 1 and torch.cuda.is_available():
                print(
                    "gpu_memory_mb",
                    int(torch.cuda.max_memory_allocated() / (1024 * 1024)),
                    "reserved_mb",
                    int(torch.cuda.max_memory_reserved() / (1024 * 1024)),
                    flush=True,
                )
                grads = {}
                for name, param in model.named_parameters():
                    if param.grad is None:
                        continue
                    for marker in ("forecast.", "future_head", "future_gru", "box_head", "decoder.1", "prediction_heads", "prior."):
                        if marker in name:
                            grads[marker] = grads.get(marker, 0.0) + float(param.grad.detach().norm())
                print("grad_norms", {key: round(value, 5) for key, value in grads.items()}, flush=True)
            if global_step == 1 or global_step % 20 == 0:
                lrs = ",".join("%.2e" % group["lr"] for group in optimizer.param_groups)
                message = "epoch %d step %d lr %s loss_det %.4f loss_forecast %.4f matched %d switch %.3f" % (
                    epoch, global_step, lrs,
                    float(losses["loss_det"].detach()),
                    float(losses["loss_forecast"].detach()),
                    losses["matched"],
                    float(losses.get("switch_rate", 0.0)),
                )
                if "loss_official" in losses:
                    message += " loss_official %.4f" % float(losses["loss_official"])
                if "proposal_recall" in losses:
                    message += " recall %.3f center %.3f -> %.3f" % (
                        losses["proposal_recall"], losses["proposal_center"], losses["final_center"]
                    )
                print(message, flush=True)
                log.write(message + "\n")
                log.flush()
            if args.max_steps and global_step >= args.max_steps:
                break
        torch.save(
            {
                "state_dict": model.state_dict(),
                "mode": spec["mode"],
                "variant": args.variant,
                "blocks": 2,
                "epoch": epoch + 1,
                "feedback": spec["feedback"],
                "unfreeze": spec["unfreeze"],
                "box_mode": spec["box_mode"],
                "detach_anchor": spec["detach_anchor"],
                "causal_time": spec["causal_time"],
                "memory": spec["memory"],
                "arch": spec["arch"],
                "forecast_weight": spec["forecast_weight"],
                "match_mode": spec["match_mode"],
                "global_step": global_step,
                "optimizer": optimizer.state_dict(),
                "rng": _rng_state(),
                "detector_lr": args.detector_lr,
                "module_lr": args.module_lr,
                "finite": True,
            },
            os.path.join(work, "epoch_%d.pt" % (epoch + 1)),
        )
        if args.eval_every and (epoch + 1) % args.eval_every == 0:
            import subprocess
            # The val process loads a second copy. Move this one off the GPU first.
            model.cpu()
            optimizer_to(optimizer, "cpu")
            torch.cuda.empty_cache()
            evaluated = subprocess.run(
                [
                    sys.executable,
                    os.path.join(ROOT, "tools", "eval_joint.py"),
                    "--checkpoint",
                    os.path.join(work, "epoch_%d.pt" % (epoch + 1)),
                    "--metrics-name",
                    "metrics_e%02d.pkl" % (epoch + 1),
                ]
            )
            print("eval_exit", epoch + 1, evaluated.returncode, flush=True)
            if torch.cuda.is_available():
                model.cuda()
                optimizer_to(optimizer, "cuda")
            model.train()
            if spec["unfreeze"]:
                model.unfreeze_fusion()
        if args.max_steps and global_step >= args.max_steps:
            break
        if args.stop_epoch and epoch + 1 >= args.stop_epoch:
            break
    elapsed = time.perf_counter() - started
    log.write("seconds %.1f\n" % elapsed)
    log.close()
    print("seconds", round(elapsed, 1), "saved", work, flush=True)


if __name__ == "__main__":
    main()
