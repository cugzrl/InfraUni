"""Train the TransFusion baseline or the joint DeTra model."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import yaml
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import infrauni  # noqa: F401
from infrauni.data import NuScenesDetectionDataset, collate_nuscenes
from infrauni.models import InfraUniDetector
from infrauni.models.detector import trainable_parameters


from infrauni.device import pick_device


def reduced_loss(losses):
    total = None
    logs = {}
    for key, value in losses.items():
        if not torch.is_tensor(value):
            continue
        logs[key] = float(value.detach())
        if value.requires_grad:
            total = value if total is None else total + value
    if total is None:
        raise RuntimeError("no differentiable loss")
    logs["loss"] = float(total.detach())
    return total, logs


def grad_report(model, batch):
    """Show that forecast loss reaches the detector only in the joint model."""
    model.zero_grad(set_to_none=True)
    output = model(batch)
    losses = model.loss(batch, output)
    report = {"mode": model.mode, "blocks": 0 if model.detra is None else model.detra.executed_blocks}
    if model.mode == "joint":
        center = output["centers"][-1]
        grad = torch.autograd.grad(losses["loss_traj"], center, retain_graph=True, allow_unused=True)[0]
        report["traj_updates_center"] = bool(grad is not None and torch.isfinite(grad).all() and grad.abs().sum() > 0)
        report["center_equals_traj"] = bool(torch.allclose(output["poses"][-1][:, :, :, 0, :], center[:, :, None, :], atol=1e-4))
    elif model.mode == "pred_head":
        report["center_equals_traj"] = bool(
            torch.allclose(output["poses"][0][:, :, :, 0, :], output["boxes"][:, :2][None, :, None, :], atol=1e-4)
        )
        report["traj_updates_center"] = False
    total, _ = reduced_loss(losses)
    total.backward()
    head_grad = any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.head.parameters() if p.requires_grad)
    lidar_grad = any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.lidar.parameters() if p.requires_grad)
    report["head_grad"] = head_grad
    report["lidar_grad"] = lidar_grad
    if model.detra is not None:
        report["detra_grad"] = any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.detra.parameters())
    model.zero_grad(set_to_none=True)
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/mini.yaml")
    parser.add_argument("--mode", default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--init", default=None)
    parser.add_argument("--grad-check", action="store_true")
    args = parser.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    if args.mode:
        cfg["mode"] = args.mode
        cfg["work_dir"] = f"experiments/mini_{args.mode}"
    if args.epochs:
        cfg["epochs"] = args.epochs
    device = pick_device()
    train_set = NuScenesDetectionDataset(ROOT / cfg["train_info"], train=True, image_size=cfg["image_size"], augment=cfg["augment"])
    loader = DataLoader(
        train_set,
        batch_size=cfg["batch_size"],
        shuffle=True,
        num_workers=cfg["workers"],
        collate_fn=collate_nuscenes,
        pin_memory=True,
    )
    model = InfraUniDetector(
        mode=cfg["mode"],
        num_proposals=cfg["num_proposals"],
        voxel_size=cfg["voxel_size"],
        refine_blocks=cfg["refine_blocks"],
    ).to(device)
    if args.init:
        state = torch.load(args.init, map_location="cpu")
        missing, unexpected = model.load_state_dict(state["model"], strict=False)
        print(f"init missing {len(missing)} unexpected {len(unexpected)}")
    work = ROOT / cfg["work_dir"]
    work.mkdir(parents=True, exist_ok=True)
    if args.grad_check:
        batch = next(iter(loader))
        report = grad_report(model, batch)
        (work / "grad_check.json").write_text(json.dumps(report, indent=2))
        print(json.dumps(report, indent=2))
    optimizer = torch.optim.AdamW(trainable_parameters(model), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    log_path = work / "train_log.jsonl"
    for epoch in range(cfg["epochs"]):
        model.train()
        for step, batch in enumerate(loader):
            optimizer.zero_grad(set_to_none=True)
            output = model(batch)
            total, logs = reduced_loss(model.loss(batch, output))
            total.backward()
            clip_grad_norm_(trainable_parameters(model), cfg["grad_clip"])
            optimizer.step()
            logs.update(epoch=epoch, step=step, mode=cfg["mode"])
            if model.detra is not None:
                logs["refine_blocks"] = model.detra.executed_blocks
            with log_path.open("a") as handle:
                handle.write(json.dumps(logs) + "\n")
            if step % 20 == 0:
                shown = " ".join(f"{key}={logs[key]:.3f}" for key in ("loss", "loss_heatmap", "layer_-1_loss_bbox", "loss_traj") if key in logs)
                print(f"epoch {epoch} step {step} {shown}", flush=True)
        torch.save({"model": model.state_dict(), "epoch": epoch, "cfg": cfg}, work / "last.pt")
        print(f"saved {work / 'last.pt'}", flush=True)


if __name__ == "__main__":
    main()
