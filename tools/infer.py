"""Run validation, official detection metrics, and the joint forecast metrics."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import infrauni  # noqa: F401
from infrauni.data import NuScenesDetectionDataset, collate_nuscenes
from infrauni.data.constants import FORECAST_SCORE
from infrauni.engine import save_frame
from infrauni.eval.nusc_det import evaluate_detection, to_nuscenes_result
from infrauni.eval.protocol import accumulate_forecast, summarize_forecast
from infrauni.models import InfraUniDetector
from infrauni.device import pick_device


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/mini.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--mode", default=None)
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--vis", type=int, default=6)
    args = parser.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    state = torch.load(args.checkpoint, map_location="cpu")
    if args.mode:
        cfg["mode"] = args.mode
    elif isinstance(state.get("cfg"), dict) and "mode" in state["cfg"]:
        cfg["mode"] = state["cfg"]["mode"]
    device = pick_device()
    dataset = NuScenesDetectionDataset(ROOT / cfg["val_info"], train=False, image_size=cfg["image_size"], augment=False)
    model = InfraUniDetector(
        mode=cfg["mode"],
        num_proposals=cfg["num_proposals"],
        voxel_size=cfg["voxel_size"],
        refine_blocks=cfg["refine_blocks"],
    ).to(device)
    model.load_state_dict(state["model"], strict=False)
    model.eval()
    work = Path(args.checkpoint).resolve().parent / "infer"
    work.mkdir(parents=True, exist_ok=True)
    det_results = []
    predictions = []
    predictions_all = []
    ground_truth = []
    limit = len(dataset) if args.max_frames <= 0 else min(args.max_frames, len(dataset))
    for index in range(limit):
        sample = dataset[index]
        batch = collate_nuscenes([sample])
        with torch.no_grad():
            pred = model.predict(batch)
        boxes = pred["boxes"].detach().cpu()
        scores = pred["scores"].detach().cpu()
        labels = pred["labels"].detach().cpu()
        traj = pred["trajectories"].detach().cpu().numpy()
        det_results.append(to_nuscenes_result(boxes, scores, labels))
        score_np = scores.numpy()
        label_np = labels.numpy()
        keep = score_np >= cfg.get("forecast_score", FORECAST_SCORE)
        predictions.append({"labels": label_np[keep], "scores": score_np[keep], "trajectories": traj[keep]})
        predictions_all.append({"labels": label_np, "scores": score_np, "trajectories": traj})
        gt_boxes = sample["gt_bboxes_3d"]
        ground_truth.append(
            {
                "labels": sample["gt_labels_3d"].numpy(),
                "current": gt_boxes.gravity_center[:, :2].numpy(),
                "future": sample["gt_fut"].numpy(),
                "future_mask": sample["gt_fut_mask"].numpy(),
            }
        )
        if index < args.vis:
            save_frame(work / f"{index:03d}.png", sample["points"], gt_boxes, boxes, traj, scores)
        if index % 10 == 0:
            print(f"frame {index} boxes {len(scores)} refine {pred['refine_blocks']}", flush=True)
    if limit != len(dataset):
        print("partial run, skip official detection eval")
        metrics = {}
    else:
        metrics = evaluate_detection(str(ROOT / cfg["val_info"]), det_results, work)
    forecast = summarize_forecast(accumulate_forecast(predictions, ground_truth))
    forecast_all = summarize_forecast(accumulate_forecast(predictions_all, ground_truth))
    payload = {
        "detection": {key: metrics[key] for key in ("mAP", "NDS") if key in metrics},
        "forecast": {key: forecast[key] for key in ("EPA", "minADE", "minFDE", "MR", "matched")},
        "forecast_all_scores": {key: forecast_all[key] for key in ("EPA", "minADE", "minFDE", "MR", "matched")},
        "per_class": forecast["per_class"],
    }
    (work / "metrics.json").write_text(json.dumps(payload, indent=2))
    printable = dict(payload["detection"])
    printable["forecast"] = payload["forecast"]
    print(json.dumps(printable, indent=2))


if __name__ == "__main__":
    main()
