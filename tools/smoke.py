"""One-batch check: calibration, detection loss, and joint refinement."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import infrauni  # noqa: F401
from infrauni.data import NuScenesDetectionDataset, collate_nuscenes
from infrauni.device import pick_device
from infrauni.eval.protocol import self_check
from infrauni.models import InfraUniDetector
from tools.train import grad_report, reduced_loss


def projection_hits(sample):
    boxes = sample["gt_bboxes_3d"].gravity_center
    homo = torch.cat([boxes, torch.ones(len(boxes), 1)], dim=1).numpy()
    meta = sample["img_meta"]
    scale = float(meta["scale_factor"][0])
    height, width = meta["input_shape"][:2]
    hits = 0
    for row in homo:
        for matrix in meta["lidar2img"]:
            uv = row @ matrix.T
            if uv[2] <= 0:
                continue
            u, v = uv[0] / uv[2] * scale, uv[1] / uv[2] * scale
            if 0 <= u < width and 0 <= v < height:
                hits += 1
                break
    return hits, len(homo)


def run_mode(mode, batch, device):
    model = InfraUniDetector(mode=mode, num_proposals=200, refine_blocks=2).to(device)
    model.train()
    output = model(batch)
    total, logs = reduced_loss(model.loss(batch, output))
    report = grad_report(model, batch)
    print(mode, {key: round(logs[key], 4) for key in logs if key.startswith("loss") or key.startswith("layer")})
    print(mode, report)
    del model, output, total
    torch.cuda.empty_cache()


def main():
    self_check()
    print("epa self-check ok")
    dataset = NuScenesDetectionDataset(ROOT / "data/infos/nuscenes_infos_val.pkl", train=False, augment=False)
    sample = dataset[0]
    hits, total = projection_hits(sample)
    print(
        f"points {tuple(sample['points'].shape)} img {tuple(sample['img'].shape)} "
        f"gt {len(sample['gt_labels_3d'])} cameras_hit {hits}/{total} "
        f"future {tuple(sample['gt_fut'].shape)} sweeps_time {float(sample['points'][:, 4].max()):.3f}"
    )
    device = pick_device()
    print("device", device)
    batch = collate_nuscenes([sample])
    run_mode("det", batch, device)
    run_mode("joint", batch, device)
    run_mode("pred_head", batch, device)


if __name__ == "__main__":
    main()
