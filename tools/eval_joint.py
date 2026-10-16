"""Official detection eval plus forecast metrics for a joint checkpoint."""

import argparse
import os
import pickle
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

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

from infrauni.data.constants import FORECAST_SCORE
from infrauni.eval.protocol import accumulate_forecast, summarize_forecast
from infrauni.joint.model import JointTransFusionDeTra


def _batch(dataset, index):
    batch = collate([dataset[index]], samples_per_gpu=1)
    if torch.cuda.is_available():
        batch = scatter(batch, [0])[0]
    return {
        "points": batch["points"][0],
        "img": batch["img"][0],
        "img_metas": batch["img_metas"][0],
    }


def _attach(output, img_metas):
    """Keep official scores and labels. Joint mode replaces box geometry."""

    head = output_model.detector.pts_bbox_head
    boxes, scores, labels = head.get_bboxes(output["outs"], img_metas)[0]
    if output_model.mode == "pred_head":
        return boxes, scores, labels, None
    proposal = output["proposal_boxes"][0, :, :2].detach()
    kept = boxes.tensor[:, :2].to(proposal.device)
    index = torch.cdist(kept, proposal).argmin(dim=1)
    updated = boxes.tensor.clone()
    refined = output["boxes"][0].detach()
    updated[:, :7] = refined[index, :7].to(updated.device)
    new_boxes = img_metas[0]["box_type_3d"](updated, box_dim=updated.shape[-1])
    return new_boxes, scores, labels, index.cpu()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()
    if not os.path.isabs(args.checkpoint):
        args.checkpoint = os.path.join(ROOT, args.checkpoint)
    saved = torch.load(args.checkpoint, map_location="cpu")
    os.chdir(os.path.join(ROOT, "third_party", "TransFusion"))
    cfg = Config.fromfile(os.path.join(ROOT, "configs", "transfusion_mini_voxel_LC.py"))
    dataset = build_dataset(cfg.data.val)
    detector = build_detector(cfg.model, train_cfg=cfg.get("train_cfg"), test_cfg=cfg.get("test_cfg"))
    global output_model
    output_model = JointTransFusionDeTra(detector, mode=saved["mode"], blocks=saved.get("blocks", 2))
    current = output_model.state_dict()
    compatible = {
        key: value
        for key, value in saved["state_dict"].items()
        if key in current and current[key].shape == value.shape
    }
    output_model.load_state_dict(compatible, strict=False)
    output_model.freeze_detector()
    if torch.cuda.is_available():
        output_model.cuda()
    output_model.eval()
    tracks = pickle.load(open(os.path.join(ROOT, "data", "nuscenes_mini_future12.pkl"), "rb"))
    count = len(dataset) if args.limit <= 0 else min(args.limit, len(dataset))
    results, predictions, ground_truth = [], [], []
    vis_dir = os.path.join(os.path.dirname(args.checkpoint), "vis")
    os.makedirs(vis_dir, exist_ok=True)
    for index in range(count):
        batch = _batch(dataset, index)
        with torch.no_grad():
            output = output_model(points=batch["points"], img=batch["img"], img_metas=batch["img_metas"])
        boxes, scores, labels, query_index = _attach(output, batch["img_metas"])
        results.append({"pts_bbox": bbox3d2result(boxes, scores, labels)})
        token = batch["img_metas"][0]["sample_idx"]
        track = tracks[token]
        keep = scores.detach().cpu().numpy() >= FORECAST_SCORE
        if query_index is None:
            proposal = output["proposal_boxes"][0, :, :2]
            query_index = torch.cdist(boxes.tensor[:, :2].to(proposal.device), proposal).argmin(dim=1).cpu()
        traj = output["trajectories"][0, query_index].detach().cpu().numpy()
        predictions.append(
            {
                "labels": labels.detach().cpu().numpy()[keep],
                "scores": scores.detach().cpu().numpy()[keep],
                "trajectories": traj[keep],
            }
        )
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
            fig, ax = plt.subplots(figsize=(6, 6))
            ax.scatter(points[::8, 0], points[::8, 1], s=0.2, c="0.7")
            shown = boxes.tensor.detach().cpu().numpy()
            score_np = scores.detach().cpu().numpy()
            draw = score_np >= FORECAST_SCORE
            ax.scatter(shown[draw, 0], shown[draw, 1], s=8, c="r")
            drawn = traj[draw]
            for modes in drawn:
                for mode in modes:
                    ax.plot(mode[:, 0], mode[:, 1], lw=0.6, alpha=0.5)
            ax.set_xlim(-54, 54)
            ax.set_ylim(-54, 54)
            ax.set_aspect("equal")
            ax.set_title("%s frame %d" % (saved["mode"], index))
            fig.savefig(os.path.join(vis_dir, "%03d.png" % index), dpi=120)
            plt.close(fig)
        if index % 20 == 0:
            print("frame", index)
    metrics = dataset.evaluate(results, metric="bbox")
    forecast = summarize_forecast(accumulate_forecast(predictions, ground_truth))
    summary = {
        "mode": saved["mode"],
        "frames": count,
        "mAP": metrics.get("pts_bbox_NuScenes/mAP", metrics),
        "NDS": metrics.get("pts_bbox_NuScenes/NDS"),
        "forecast": {key: forecast[key] for key in ("EPA", "minADE", "minFDE", "MR", "matched", "gt")},
    }
    pickle.dump({"detection": metrics, "forecast": forecast, "summary": summary}, open(os.path.join(os.path.dirname(args.checkpoint), "metrics.pkl"), "wb"))
    print(summary)


if __name__ == "__main__":
    main()
