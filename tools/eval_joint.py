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
from infrauni.joint.loss import _boxes_from_block
from infrauni.joint.model import JointTransFusionDeTra
from infrauni.joint.unified import UnifiedTransFusionDeTra
from infrauni.joint.postprocess import filter_query_indices, take_queries
from infrauni.joint.targets import match_queries

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
    parser.add_argument("--metrics-name", default="metrics_v3.pkl")
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
        if saved.get("arch") == "unified":
            model = UnifiedTransFusionDeTra(
                detector,
                blocks=saved.get("blocks", 2),
                detach_anchor=saved.get("detach_anchor", True),
                causal_time=saved.get("causal_time", False),
                forecast_weight=saved.get("forecast_weight", 1.0),
            )
        else:
            model = JointTransFusionDeTra(
                detector,
                mode=saved["mode"],
                blocks=saved.get("blocks", 2),
                box_mode=saved.get("box_mode", "full"),
                detach_anchor=saved.get("detach_anchor", False),
                causal_time=saved.get("causal_time", False),
            )
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
    results, proposal_results, fused_results = [], [], []
    predictions, proposal_predictions, ground_truth = [], [], []
    delta = {"dxy": 0.0, "dz": 0.0, "dsize": 0.0, "dyaw": 0.0, "n": 0.0,
             "err_before": 0.0, "err_after": 0.0, "matched": 0.0,
             "switched": 0.0, "switch_compared": 0.0}
    vis_dir = os.path.join(work, "vis")
    os.makedirs(vis_dir, exist_ok=True)
    for index in range(count):
        batch = _batch(dataset, index)
        with torch.no_grad():
            output = model(points=batch["points"], img=batch["img"], img_metas=batch["img_metas"])
        boxes, kept = _export(output, batch["img_metas"])
        results.append({"pts_bbox": bbox3d2result(boxes, kept["scores"], kept["labels"])})
        proposal = output["proposal_boxes"][0]
        proposal_scores = output["proposal_scores"][0] if "proposal_scores" in output else output["scores"][0]
        proposal_labels = output["proposal_labels"][0] if "proposal_labels" in output else output["labels"][0]
        proposal_kept = take_queries(
            filter_query_indices(proposal_scores, proposal, score_thresh=None, center_range=POST_CENTER_RANGE),
            boxes=proposal, scores=proposal_scores, labels=proposal_labels,
            trajectories=output["trajectories"][0],
        )
        proposal_boxes = batch["img_metas"][0]["box_type_3d"](
            proposal_kept["boxes"], box_dim=proposal_kept["boxes"].shape[-1]
        )
        proposal_results.append({"pts_bbox": bbox3d2result(proposal_boxes, proposal_kept["scores"], proposal_kept["labels"])})
        fused_scores = None
        if output["object_logits"] is not None and not output.get("unified"):
            prob = output["object_logits"][0].sigmoid()
            # prob 0.5 leaves the TransFusion score unchanged.
            fused = output["scores"][0] * (0.5 + prob)
            fused_kept = take_queries(
                filter_query_indices(fused, output["boxes"][0], score_thresh=None, center_range=POST_CENTER_RANGE),
                boxes=output["boxes"][0], scores=fused, labels=output["labels"][0],
            )
            fused_boxes = batch["img_metas"][0]["box_type_3d"](
                fused_kept["boxes"], box_dim=fused_kept["boxes"].shape[-1]
            )
            fused_results.append({"pts_bbox": bbox3d2result(fused_boxes, fused_kept["scores"], fused_kept["labels"])})
            fused_scores = fused_kept["scores"].detach().cpu().numpy()
        scores = kept["scores"].detach().cpu().numpy()
        labels = kept["labels"].detach().cpu().numpy()
        trajectories = kept["trajectories"].detach().cpu().numpy()
        keep = scores >= FORECAST_SCORE
        predictions.append(
            {"labels": labels[keep], "scores": scores[keep], "trajectories": trajectories[keep]}
        )
        p_scores = proposal_kept["scores"].detach().cpu().numpy()
        p_keep = p_scores >= FORECAST_SCORE
        proposal_predictions.append(
            {
                "labels": proposal_kept["labels"].detach().cpu().numpy()[p_keep],
                "scores": p_scores[p_keep],
                "trajectories": proposal_kept["trajectories"].detach().cpu().numpy()[p_keep],
            }
        )
        if output["detra"] is not None:
            paired = take_queries(
                filter_query_indices(output["scores"][0], output["boxes"][0], score_thresh=None, center_range=POST_CENTER_RANGE),
                refined=output["boxes"][0], base=output["proposal_boxes"][0], labels=proposal_labels,
            )
            diff = paired["refined"][:, :7] - paired["base"][:, :7]
            delta["dxy"] += diff[:, :2].norm(dim=-1).sum().item()
            delta["dz"] += diff[:, 2].abs().sum().item()
            delta["dsize"] += (paired["refined"][:, 3:6].clamp_min(1e-2).log() - paired["base"][:, 3:6].clamp_min(1e-2).log()).abs().mean(dim=-1).sum().item()
            yaw = torch.atan2(torch.sin(diff[:, 6]), torch.cos(diff[:, 6])).abs()
            delta["dyaw"] += yaw.sum().item()
            delta["n"] += diff.shape[0]
            token_track = tracks[batch["img_metas"][0]["sample_idx"]]
            gt_xy = torch.as_tensor(token_track["current"], device=diff.device, dtype=diff.dtype)
            gt_lab = torch.as_tensor(token_track["labels"], device=diff.device)
            if len(gt_xy):
                base_match = match_queries(paired["base"][:, :2], paired["labels"], gt_xy, gt_lab)
                ok = base_match >= 0
                if bool(ok.any()):
                    before = (paired["base"][ok, :2] - gt_xy[base_match[ok]]).norm(dim=-1)
                    after = (paired["refined"][ok, :2] - gt_xy[base_match[ok]]).norm(dim=-1)
                    delta["err_before"] += before.sum().item()
                    delta["err_after"] += after.sum().item()
                    delta["matched"] += int(ok.sum())
                anchor = match_queries(output["proposal_boxes"][0, :, :2], proposal_labels, gt_xy, gt_lab)
                for block in output["detra"]["aux_outputs"]:
                    alt = match_queries(_boxes_from_block(block)[:, :2], proposal_labels, gt_xy, gt_lab)
                    both = (anchor >= 0) & (alt >= 0)
                    delta["switch_compared"] += int(both.sum())
                    if bool(both.any()):
                        delta["switched"] += int((anchor[both] != alt[both]).sum())
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
    def _pack(metrics, forecast):
        return {
            "mAP": metrics.get("pts_bbox_NuScenes/mAP", metrics),
            "NDS": metrics.get("pts_bbox_NuScenes/NDS"),
            "forecast": {
                key: forecast[key]
                for key in ("EPA", "minADE", "minFDE", "MR", "matched", "gt", "fp", "ignored")
            },
        }

    metrics = dataset.evaluate(results, metric="bbox")
    forecast = summarize_forecast(accumulate_forecast(predictions, ground_truth))
    proposal_metrics = dataset.evaluate(proposal_results, metric="bbox")
    fused_pack = None
    if fused_results:
        fused_metrics = dataset.evaluate(fused_results, metric="bbox")
        fused_pack = {"mAP": fused_metrics.get("pts_bbox_NuScenes/mAP"), "NDS": fused_metrics.get("pts_bbox_NuScenes/NDS")}
    count_n = max(delta["n"], 1.0)
    matched_n = max(delta["matched"], 1.0)
    summary = {
        "variant": name,
        "protocol": "v3",
        "frames": count,
        **_pack(metrics, forecast),
        "proposal_boxes": {
            "mAP": proposal_metrics.get("pts_bbox_NuScenes/mAP"),
            "NDS": proposal_metrics.get("pts_bbox_NuScenes/NDS"),
        },
        "fused_score": fused_pack,
        "box_delta": {
            "dxy_m": delta["dxy"] / count_n,
            "dz_m": delta["dz"] / count_n,
            "dlog_size": delta["dsize"] / count_n,
            "dyaw_rad": delta["dyaw"] / count_n,
            "center_err_before_m": delta["err_before"] / matched_n,
            "center_err_after_m": delta["err_after"] / matched_n,
            "switch_rate": delta["switched"] / delta["switch_compared"] if delta["switch_compared"] else 0.0,
        },
    }
    pickle.dump(
        {"detection": metrics, "forecast": forecast, "summary": summary},
        open(os.path.join(work, args.metrics_name), "wb"),
    )
    print(summary, flush=True)


if __name__ == "__main__":
    main()
