"""Train ablation variants B, C1, C2, and C3 on nuScenes-mini.

Every variant starts from the same TransFusion-LC checkpoint and sees the same
shuffled frame order. C2 and C3 share the architecture and the trainable set.
C3 keeps the forecast loss off the detection parameters.
"""

import argparse
import logging
import os
import pickle
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

from infrauni.joint.loss import apply_backward, compute_joint_loss
from infrauni.joint.model import JointTransFusionDeTra

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
    "f0": {"mode": "joint", "unfreeze": True, "feedback": True, "box_mode": "full", "detach_anchor": False, "causal_time": True},
}


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


def _official_loss(model, output, batch):
    raw = model.detector.pts_bbox_head.loss(
        batch["gt_bboxes_3d"], batch["gt_labels_3d"], output["outs"]
    )
    terms = [value for key, value in raw.items() if "loss" in key and torch.is_tensor(value)]
    if not terms:
        return output["boxes"].sum() * 0.0
    return sum(terms)


def _trainable(model):
    return sum(param.numel() for param in model.parameters() if param.requires_grad)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", required=True, choices=sorted(VARIANTS))
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--max-steps", type=int, default=0)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--eval-every", type=int, default=0)
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
    model = JointTransFusionDeTra(
        detector,
        mode=spec["mode"],
        blocks=2,
        box_mode=spec["box_mode"],
        detach_anchor=spec["detach_anchor"],
        causal_time=spec["causal_time"],
    )
    model.freeze_detector()
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
        groups.append({"params": new_params, "lr": 1e-4})
    if detector_params:
        groups.append({"params": detector_params, "lr": 1e-5})
    optimizer = torch.optim.AdamW(groups, weight_decay=0.01)
    work = os.path.join(ROOT, "experiments", "ablation_%s_mini" % args.variant)
    os.makedirs(work, exist_ok=True)
    tracks = pickle.load(open(os.path.join(ROOT, "data", "nuscenes_mini_future12.pkl"), "rb"))
    log = open(os.path.join(work, "train.log"), "a")
    print(
        "variant", args.variant,
        "mode", spec["mode"],
        "unfreeze", spec["unfreeze"],
        "feedback", spec["feedback"],
        "trainable", _trainable(model),
        "frames", len(dataset),
        flush=True,
    )
    log.write("trainable %d frames %d\n" % (_trainable(model), len(dataset)))
    started = time.perf_counter()
    global_step = 0
    for epoch in range(args.epochs):
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
            for group, base in zip(optimizer.param_groups, (1e-4, 1e-5)[: len(optimizer.param_groups)]):
                group["lr"] = base * (1.0 - progress) + 1e-6 * progress
            token = batch["img_metas"][0]["sample_idx"]
            gt_boxes = batch["gt_bboxes_3d"][0].tensor.to(batch["img"].device)
            gt_labels = batch["gt_labels_3d"][0].to(gt_boxes.device).long()
            optimizer.zero_grad(set_to_none=True)
            output = model(points=batch["points"], img=batch["img"], img_metas=batch["img_metas"])
            if global_step == 1:
                same = torch.allclose(
                    output["boxes"][0, :, :2],
                    output["trajectories"][0, :, 0, 0, :],
                    atol=1e-4,
                )
                print("current_box_matches_trajectory", bool(same), "queries", output["boxes"].shape[1], flush=True)
            losses = compute_joint_loss(output, gt_boxes[:, :7], gt_labels, tracks[token])
            if spec["unfreeze"]:
                official = _official_loss(model, output, batch)
                losses["loss_det"] = losses["loss_det"] + official
                losses["loss_official"] = official.detach()
            finite = torch.isfinite(losses["loss_det"]) and torch.isfinite(losses["loss_forecast"])
            if not finite:
                print("skip non-finite step", global_step, flush=True)
                log.write("skip non-finite step %d\n" % global_step)
                optimizer.zero_grad(set_to_none=True)
                continue
            apply_backward(model, losses, feedback=spec["feedback"])
            torch.nn.utils.clip_grad_norm_(
                [param for param in model.parameters() if param.requires_grad and param.grad is not None],
                1.0,
            )
            optimizer.step()
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
                    for token in ("forecast.", "future_head", "future_gru", "box_head", "decoder.1", "prediction_heads"):
                        if token in name:
                            grads[token] = grads.get(token, 0.0) + float(param.grad.detach().norm())
                print("grad_norms", {key: round(value, 5) for key, value in grads.items()}, flush=True)
            if global_step == 1 or global_step % 20 == 0:
                message = "epoch %d step %d loss_det %.4f loss_forecast %.4f matched %d switch %.3f" % (
                    epoch, global_step,
                    float(losses["loss_det"].detach()),
                    float(losses["loss_forecast"].detach()),
                    losses["matched"],
                    float(losses.get("switch_rate", 0.0)),
                )
                if "loss_official" in losses:
                    message += " loss_official %.4f" % float(losses["loss_official"])
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
            },
            os.path.join(work, "epoch_%d.pt" % (epoch + 1)),
        )
        if args.eval_every and (epoch + 1) % args.eval_every == 0:
            import subprocess
            torch.cuda.empty_cache()
            subprocess.run(
                [
                    sys.executable,
                    os.path.join(ROOT, "tools", "eval_joint.py"),
                    "--checkpoint",
                    os.path.join(work, "epoch_%d.pt" % (epoch + 1)),
                    "--metrics-name",
                    "metrics_e%02d.pkl" % (epoch + 1),
                ],
                check=True,
            )
            model.train()
            if spec["unfreeze"]:
                model.unfreeze_fusion()
        if args.max_steps and global_step >= args.max_steps:
            break
    elapsed = time.perf_counter() - started
    log.write("seconds %.1f\n" % elapsed)
    log.close()
    print("seconds", round(elapsed, 1), "saved", work, flush=True)


if __name__ == "__main__":
    main()
