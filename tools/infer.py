"""Run a checkpoint on nuScenes-mini val and save BEV figures plus json."""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from infrauni.data.constants import CLASS_NAMES
from infrauni.data.nuscenes_mini import NuScenesMiniJoint, collate_batch
from infrauni.engine.metrics import accumulate, predictions_from_output
from infrauni.engine.visualize import save_bev
from infrauni.models.joint_net import InfraUniBase


def to_device(batch, device):
    out = {}
    for key, value in batch.items():
        if torch.is_tensor(value):
            out[key] = value.to(device, non_blocking=True)
        else:
            out[key] = value
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/nuscenes_mini.yaml")
    parser.add_argument("--ckpt", default="experiments/nuscenes_mini/last.pt")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--vis", type=int, default=6)
    parser.add_argument("--split", default="val")
    args = parser.parse_args()

    with open(ROOT / args.config, "r") as f:
        cfg = yaml.safe_load(f)
    if args.device == "auto":
        if not torch.cuda.is_available():
            device = torch.device("cpu")
        else:
            best_i, best_free = 0, -1
            for i in range(torch.cuda.device_count()):
                free, _ = torch.cuda.mem_get_info(i)
                if free > best_free:
                    best_i, best_free = i, free
            device = torch.device(f"cuda:{best_i}")
    else:
        device = torch.device(args.device)

    info_path = ROOT / cfg["data"]["info_path"]
    with open(info_path, "rb") as f:
        infos = pickle.load(f)["infos"]
    dataset = NuScenesMiniJoint(infos, cfg, args.split)
    loader = DataLoader(
        dataset,
        batch_size=int(cfg["train"]["batch_size"]),
        shuffle=False,
        num_workers=2,
        collate_fn=collate_batch,
    )
    ckpt = torch.load(ROOT / args.ckpt, map_location="cpu")
    model = InfraUniBase(ckpt.get("cfg", cfg)).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    out_dir = ROOT / cfg["out_dir"] / f"infer_{args.split}"
    out_dir.mkdir(parents=True, exist_ok=True)
    preds, gts = [], []
    saved = 0
    with torch.no_grad():
        for batch in loader:
            batch = to_device(batch, device)
            out = model(batch)
            pred_i, gt_i = predictions_from_output(out, batch, score_thresh=0.0)
            points = batch["points"].detach().cpu().numpy()
            point_mask = batch["point_mask"].detach().cpu().numpy()
            for b, (pred, gt) in enumerate(zip(pred_i, gt_i)):
                preds.append(pred)
                gts.append(gt)
                if saved >= args.vis:
                    continue
                pts = points[b][point_mask[b]]
                title = f"{gt['token'][:8]}  top-20 heatmap queries"
                save_bev(
                    out_dir / f"bev_{saved:02d}.png",
                    pts,
                    gt,
                    pred,
                    cfg["pc_range"],
                    title,
                )
                saved += 1

    metrics = accumulate(
        preds,
        gts,
        dist_thresh=float(cfg["eval"]["dist_thresh"]),
        score_thresh=float(cfg["eval"]["score_thresh"]),
    )
    dump = []
    for pred in preds:
        keep = np.argsort(-pred["query_score"])[:20]
        names = [CLASS_NAMES[int(i)] for i in pred["label"][keep]]
        dump.append(
            {
                "token": pred["token"],
                "names": names,
                "score": pred["score"][keep].round(4).tolist(),
                "box": pred["box"][keep].round(3).tolist(),
                "traj_mode0": pred["traj"][keep][:, 0].round(3).tolist(),
            }
        )
    (out_dir / "predictions.json").write_text(json.dumps(dump))
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))
    print(f"wrote {out_dir}")


if __name__ == "__main__":
    main()
