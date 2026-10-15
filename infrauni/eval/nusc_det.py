"""Official nuScenes detection evaluation for one result list."""

from __future__ import annotations

from pathlib import Path

import torch
from mmdet3d.core.bbox import LiDARInstance3DBoxes
from mmdet3d.datasets import NuScenesDataset

from ..data.constants import CLASS_NAMES, NUSCENES_ROOT


def build_eval_dataset(ann_file, root=NUSCENES_ROOT):
    return NuScenesDataset(
        ann_file=ann_file,
        pipeline=[],
        dataset_root=root,
        object_classes=list(CLASS_NAMES),
        modality=dict(
            use_camera=True,
            use_lidar=True,
            use_radar=False,
            use_map=False,
            use_external=False,
        ),
        test_mode=True,
        box_type_3d="LiDAR",
        eval_version="detection_cvpr_2019",
        use_valid_flag=True,
        filter_empty_gt=False,
    )


def to_nuscenes_result(boxes, scores, labels):
    if not torch.is_tensor(boxes):
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
    boxes = boxes.detach().cpu()
    if boxes.numel() == 0:
        boxes = boxes.reshape(0, 9)
    return {
        "boxes_3d": LiDARInstance3DBoxes(boxes, box_dim=boxes.shape[-1]),
        "scores_3d": torch.as_tensor(scores, dtype=torch.float32).detach().cpu(),
        "labels_3d": torch.as_tensor(labels, dtype=torch.long).detach().cpu(),
    }


def evaluate_detection(ann_file, results, out_dir):
    dataset = build_eval_dataset(ann_file)
    if len(results) != len(dataset):
        raise RuntimeError(f"result length {len(results)} != dataset length {len(dataset)}")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics = dataset.evaluate(results, jsonfile_prefix=str(out_dir / "nusc"))
    return {"mAP": metrics["object/map"], "NDS": metrics["object/nds"], **metrics}
