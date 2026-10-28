"""Load one official train batch and one val batch from a converted split."""

import logging
import os
import sys

os.chdir("/home/kemove/devdata1/zrl/RESOLVE/tools")
sys.path.insert(0, "/home/kemove/devdata1/zrl/RESOLVE")
sys.path.insert(0, "/home/kemove/devdata1/zrl/RESOLVE/tools")
sys.path.append("/home/kemove/anaconda3/envs/bevfusion/lib/python3.8/site-packages")

from pcdet.config import cfg, cfg_from_yaml_file  # noqa: E402
from pcdet.datasets import build_dataloader  # noqa: E402


def _one(training, data_path):
    cfg_from_yaml_file("cfgs/sunlakes_models/transfusion.yaml", cfg)
    cfg.DATA_CONFIG.DATA_PATH = data_path
    cfg.DATA_CONFIG.DATA_AUGMENTOR.DISABLE_AUG_LIST = ["gt_sampling"]
    logger = logging.getLogger("resolve-loader")
    logger.setLevel(logging.INFO)
    loader = build_dataloader(
        dataset_cfg=cfg.DATA_CONFIG,
        class_names=cfg.CLASS_NAMES,
        batch_size=1,
        dist=False,
        workers=0,
        training=training,
        logger=logger,
    )[1]
    batch = next(iter(loader))
    points = batch["points"]
    print(
        "training" if training else "val",
        "samples", len(loader.dataset),
        "points", tuple(points.shape),
        "voxels", tuple(batch["voxels"].shape),
        "gt", tuple(batch["gt_boxes"].shape) if "gt_boxes" in batch else None,
        "range", list(cfg.DATA_CONFIG.POINT_CLOUD_RANGE),
        "voxel", list(cfg.DATA_CONFIG.DATA_PROCESSOR[-1].VOXEL_SIZE),
    )


if __name__ == "__main__":
    path = sys.argv[1]
    _one(True, path)
    _one(False, path)
