"""Train the InfraUni base joint network on nuScenes-mini."""

from __future__ import annotations

import argparse
import json
import pickle
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from infrauni.data.nuscenes_mini import NuScenesMiniJoint, build_infos, collate_batch
from infrauni.engine.metrics import accumulate, predictions_from_output
from infrauni.models.joint_net import InfraUniBase
from infrauni.models.losses import JointLoss


def seed_all(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def pick_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if not torch.cuda.is_available():
        return torch.device("cpu")
    best_i, best_free = 0, -1
    for i in range(torch.cuda.device_count()):
        free, _ = torch.cuda.mem_get_info(i)
        if free > best_free:
            best_i, best_free = i, free
    return torch.device(f"cuda:{best_i}")


def to_device(batch, device):
    out = {}
    for key, value in batch.items():
        if torch.is_tensor(value):
            out[key] = value.to(device, non_blocking=True)
        else:
            out[key] = value
    return out


def grad_norm(module):
    total = 0.0
    for param in module.parameters():
        if param.grad is not None:
            total += float(param.grad.detach().float().pow(2).sum())
    return total ** 0.5


def load_infos(cfg):
    info_path = ROOT / cfg["data"]["info_path"]
    if not info_path.is_file():
        print(f"building infos -> {info_path}")
        built = build_infos(cfg, ROOT)
        print("class histogram", built["class_hist"])
        print("train", len(built["infos"]["train"]), "val", len(built["infos"]["val"]))
    with open(info_path, "rb") as f:
        return pickle.load(f)["infos"]


@torch.no_grad()
def evaluate(model, loader, device, cfg):
    model.eval()
    preds, gts = [], []
    for batch in loader:
        batch = to_device(batch, device)
        out = model(batch)
        pred_i, gt_i = predictions_from_output(out, batch, score_thresh=0.0)
        preds.extend(pred_i)
        gts.extend(gt_i)
    model.train()
    return accumulate(
        preds,
        gts,
        dist_thresh=float(cfg["eval"]["dist_thresh"]),
        score_thresh=float(cfg["eval"]["score_thresh"]),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/nuscenes_mini.yaml")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--max-steps", type=int, default=0, help="debug cap, 0 = full epochs")
    args = parser.parse_args()

    with open(ROOT / args.config, "r") as f:
        cfg = yaml.safe_load(f)
    if args.epochs is not None:
        cfg["train"]["epochs"] = args.epochs
    if args.batch_size is not None:
        cfg["train"]["batch_size"] = args.batch_size

    seed_all(int(cfg["seed"]))
    device = pick_device(args.device)
    out_dir = ROOT / cfg["out_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"device={device} out={out_dir}")

    infos = load_infos(cfg)
    train_set = NuScenesMiniJoint(infos, cfg, "train")
    val_set = NuScenesMiniJoint(infos, cfg, "val")
    train_loader = DataLoader(
        train_set,
        batch_size=int(cfg["train"]["batch_size"]),
        shuffle=True,
        num_workers=int(cfg["train"]["workers"]),
        collate_fn=collate_batch,
        pin_memory=True,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=int(cfg["train"]["batch_size"]),
        shuffle=False,
        num_workers=int(cfg["train"]["workers"]),
        collate_fn=collate_batch,
        pin_memory=True,
    )
    print(f"samples train={len(train_set)} val={len(val_set)}")

    model = InfraUniBase(cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"parameters={n_params / 1e6:.2f}M")
    criterion = JointLoss(cfg).to(device)
    optim = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg["train"]["lr"]),
        weight_decay=float(cfg["train"]["weight_decay"]),
    )
    epochs = int(cfg["train"]["epochs"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=max(epochs, 1))
    log_path = out_dir / "train_log.jsonl"
    log_path.write_text("")
    best_f1 = -1.0
    global_step = 0
    grad_report = None

    for epoch in range(1, epochs + 1):
        model.train()
        t0 = time.time()
        running = []
        for batch in train_loader:
            batch = to_device(batch, device)
            optim.zero_grad(set_to_none=True)
            out = model(batch)
            loss, stats, parts = criterion(out, batch)
            if not torch.isfinite(loss):
                print("non-finite loss, skip step", stats)
                continue
            if grad_report is None:
                watched = {
                    "pillar": model.pillar,
                    "image_encoder": model.image_encoder,
                    "delta_head": model.delta_head,
                    "motion_gate": model.motion_gate,
                    "motion_geom": model.motion_geom,
                    "head_motion": model.head_motion,
                }
                parts["forecast"].backward(retain_graph=True)
                from_forecast = {name: grad_norm(mod) for name, mod in watched.items()}
                optim.zero_grad(set_to_none=True)
                parts["det"].backward(retain_graph=True)
                from_det = {name: grad_norm(mod) for name, mod in watched.items()}
                optim.zero_grad(set_to_none=True)
                grad_report = {"from_forecast_loss": from_forecast, "from_detection_loss": from_det}
                print("grad split", json.dumps(grad_report))
                (out_dir / "grad_check.json").write_text(json.dumps(grad_report, indent=2))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), float(cfg["train"]["grad_clip"])
            )
            optim.step()
            running.append(stats)
            global_step += 1
            if global_step % int(cfg["train"]["log_every"]) == 0:
                print(
                    f"epoch {epoch} step {global_step} "
                    f"loss {stats['loss']:.3f} hm {stats['hm']:.3f} "
                    f"cls {stats['cls']:.3f} box {stats['box']:.3f} "
                    f"traj {stats['traj']:.3f} matched {stats['matched']}"
                )
            if args.max_steps and global_step >= args.max_steps:
                break
        scheduler.step()
        def _mean(key):
            return float(np.mean([row[key] for row in running])) if running else None

        mean_loss = _mean("loss")
        metrics = evaluate(model, val_loader, device, cfg)
        record = {
            "epoch": epoch,
            "train_loss": mean_loss,
            "hm": _mean("hm"),
            "cls": _mean("cls"),
            "box": _mean("box"),
            "traj": _mean("traj"),
            "mode": _mean("mode"),
            "lr": optim.param_groups[0]["lr"],
            "seconds": time.time() - t0,
            "val": metrics,
        }
        print(
            f"epoch {epoch} loss {mean_loss:.3f} hm {_mean('hm'):.3f} "
            f"box {_mean('box'):.3f} traj {_mean('traj'):.3f} "
            f"P {metrics['precision']:.3f} R {metrics['recall']:.3f} "
            f"F1 {metrics['f1']:.3f} minADE {metrics['minADE']} "
            f"minFDE {metrics['minFDE']} ({record['seconds']:.1f}s)"
        )
        with open(log_path, "a") as f:
            f.write(json.dumps(record) + "\n")
        ckpt = {
            "model": model.state_dict(),
            "epoch": epoch,
            "cfg": cfg,
            "val": metrics,
        }
        torch.save(ckpt, out_dir / "last.pt")
        if metrics["f1"] > best_f1:
            best_f1 = metrics["f1"]
            torch.save(ckpt, out_dir / "best.pt")
        if epoch == epochs and not args.max_steps:
            train_metrics = evaluate(model, train_loader, device, cfg)
            record["train_det"] = train_metrics
            print(
                f"train-split P {train_metrics['precision']:.3f} "
                f"R {train_metrics['recall']:.3f} F1 {train_metrics['f1']:.3f} "
                f"minADE {train_metrics['minADE']} minFDE {train_metrics['minFDE']}"
            )
            with open(log_path, "a") as f:
                f.write(json.dumps({"epoch": epoch, "train_det": train_metrics}) + "\n")
        if args.max_steps and global_step >= args.max_steps:
            break

    print(f"done best_f1={best_f1:.3f}")


if __name__ == "__main__":
    main()
