"""Detection and forecast evaluation for one ablation checkpoint.

Boxes, scores, labels, and trajectories are sliced by the same query index.
Constant-velocity is an eval-only extrapolation of the TransFusion-LC boxes.
"""

import argparse
import logging
import os
import pickle
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

logging.getLogger("mmcv").setLevel(logging.ERROR)

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "third_party", "MC-DeTra"))
sys.path.insert(0, os.path.join(ROOT, "third_party", "TransFusion"))

from mmcv import Config
from mmcv.parallel import collate, scatter
from mmcv.runner import load_checkpoint
from mmdet3d.core import bbox3d2result
from mmdet3d.datasets import build_dataset
from mmdet3d.models import build_detector

from infrauni.data.constants import FORECAST_SCORE, POST_CENTER_RANGE
from infrauni.eval.protocol import accumulate_forecast, summarize_forecast
from infrauni.joint.model import JointTransFusionDeTra
from infrauni.joint.postprocess import filter_query_indices, take_queries

LC_CHECKPOINT = os.path.join(ROOT, "experiments", "transfusion_voxel_LC_mini", "epoch_1.pth")


def _batch(dataset, index):
    batch = collate([dataset[index]], samples_per_gpu=1)
    if torch.cuda.is_available():
        batch = scatter(batch, [0])[0]
    return {
        "points": batch["points"][0],
        "img": batch["img"][0],
        "img_metas": batch["img_metas"][0],
    }


def _export(output, img_metas):
    """Range-filter one sample without searching for a new query."""

    boxes = output["boxes"][0]
    scores = output["scores"][0]
    labels = output["labels"][0]
    trajectories = output["trajectories"][0]
    index = filter_query_indices(scores, boxes, score_thresh=None, center_range=POST_CENTER_RANGE)
    kept = take_queries(index, boxes=boxes, scores=scores, labels=labels, trajectories=trajectories)
    box_type = img_metas[0]["box_type_3d"]
    return box_type(kept["boxes"], box_dim=kept["boxes"].shape[-1]), kept


def _draw(path, points, boxes, scores, trajectories, title):
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(points[::8, 0], points[::8, 1], s=0.2, c="0.7")
    draw = scores >= FORECAST_SCORE
    ax.scatter(boxes[draw, 0], boxes[draw, 1], s=8, c="r")
    for modes in trajectories[draw]:
        for mode in modes:
            ax.plot(mode[:, 0], mode[:, 1], lw=0.6, alpha=0.5)
    ax.set_xlim(-54, 54)
    ax.set_ylim(-54, 54)
    ax.set_aspect("equal")
    ax.set_title(title)
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--variant", default="", choices=["", "cv"])
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()
    os.chdir(os.path.join(ROOT, "third_party", "TransFusion"))
    cfg = Config.fromfile(os.path.join(ROOT, "configs", "transfusion_mini_voxel_LC.py"))
    dataset = build_dataset(cfg.data.val)
    detector = build_detector(cfg.model, train_cfg=cfg.get("train_cfg"), test_cfg=cfg.get("test_cfg"))
    if args.variant == "cv":
        load_checkpoint(detector, LC_CHECKPOINT, map_location="cpu")
        model = JointTransFusionDeTra(detector, mode="cv", blocks=2)
        work = os.path.join(ROOT, "experiments", "ablation_cv_mini")
        name = "cv"
    else:
        checkpoint = args.checkpoint if os.path.isabs(args.checkpoint) else os.path.join(ROOT, args.checkpoint)
        saved = torch.load(checkpoint, map_location="cpu")
        model = JointTransFusionDeTra(detector, mode=saved["mode"], blocks=saved.get("blocks", 2))
        current = model.state_dict()
        compatible = {
            key: value
            for key, value in saved["state_dict"].items()
            if key in current and current[key].shape == value.shape
        }
        model.load_state_dict(compatible, strict=False)
        work = os.path.dirname(checkpoint)
        name = saved.get("variant", saved["mode"])
    os.makedirs(work, exist_ok=True)
    model.freeze_detector()
    if torch.cuda.is_available():
        model.cuda()
    model.eval()
    tracks = pickle.load(open(os.path.join(ROOT, "data", "nuscenes_mini_future12.pkl"), "rb"))
    count = len(dataset) if args.limit <= 0 else min(args.limit, len(dataset))
    results, predictions, ground_truth = [], [], []
    vis_dir = os.path.join(work, "vis")
    os.makedirs(vis_dir, exist_ok=True)
    for index in range(count):
        batch = _batch(dataset, index)
        with torch.no_grad():
            output = model(points=batch["points"], img=batch["img"], img_metas=batch["img_metas"])
        boxes, kept = _export(output, batch["img_metas"])
        results.append({"pts_bbox": bbox3d2result(boxes, kept["scores"], kept["labels"])})
        scores = kept["scores"].detach().cpu().numpy()
        labels = kept["labels"].detach().cpu().numpy()
        trajectories = kept["trajectories"].detach().cpu().numpy()
        keep = scores >= FORECAST_SCORE
        predictions.append(
            {"labels": labels[keep], "scores": scores[keep], "trajectories": trajectories[keep]}
        )
        token = batch["img_metas"][0]["sample_idx"]
        track = tracks[token]
        ground_truth.append(
            {
                "labels": track["labels"],
                "current": track["current"],
                "future": track["future"],
                "future_mask": track["mask"],
            }
        )
        if index in (0, 10, 20):
            data = dataset.prepare_test_data(index)
            points = data["points"][0]._data.numpy()
            _draw(
                os.path.join(vis_dir, "%03d.png" % index),
                points,
                kept["boxes"].detach().cpu().numpy(),
                scores,
                trajectories,
                "%s frame %d" % (name, index),
            )
        if index % 20 == 0:
            print("frame", index, flush=True)
    metrics = dataset.evaluate(results, metric="bbox")
    forecast = summarize_forecast(accumulate_forecast(predictions, ground_truth))
    summary = {
        "variant": name,
        "frames": count,
        "mAP": metrics.get("pts_bbox_NuScenes/mAP", metrics),
        "NDS": metrics.get("pts_bbox_NuScenes/NDS"),
        "forecast": {
            key: forecast[key]
            for key in ("EPA", "minADE", "minFDE", "MR", "matched", "gt", "fp")
        },
    }
    pickle.dump(
        {"detection": metrics, "forecast": forecast, "summary": summary},
        open(os.path.join(work, "metrics.pkl"), "wb"),
    )
    print(summary, flush=True)


if __name__ == "__main__":
    main()
