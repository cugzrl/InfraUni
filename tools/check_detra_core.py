"""Forward and backward check for the unmodified MC-DeTra refinement stack."""

import os
import sys

import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "third_party", "MC-DeTra"))

from detra_repro.config import DataConfig, ModelConfig
from detra_repro.losses import wta_forecasting_loss
from detra_repro.models.refinement import DeTraMini


def main():
    torch.manual_seed(0)
    # The toy map is 32 cells. ROI and voxel size put the origin inside stride 4.
    data = DataConfig(roi_m=(-8.0, -8.0, 8.0, 8.0), voxel_size_m=0.125, future_steps=12)
    cfg = ModelConfig(
        num_modes=6,
        num_times=13,
        hidden_dim=32,
        num_refinement_blocks=2,
        num_classes=10,
        deformable_points=4,
    )
    model = DeTraMini(cfg, data)
    feat = torch.randn(1, 32, 32, 32, requires_grad=True)
    poses = torch.zeros(1, 4, 3)
    poses[0, :, 0] = torch.linspace(-4, 4, 4)
    boxes = torch.zeros(1, 4, 7)
    boxes[..., :2] = poses[..., :2]
    boxes[..., 3:6] = torch.tensor([4.0, 1.8, 1.5])
    pred = model(feat, poses, boxes)
    assert len(pred["aux_outputs"]) == 2
    assert pred["future_xy"].shape == (1, 4, 6, 12, 2)
    same_mode = torch.allclose(pred["poses"][:, :, 0, 0, :2], pred["poses"][:, :, 1, 0, :2])
    assert same_mode
    target = pred["future_xy"].detach() + 1.0
    loss = wta_forecasting_loss(
        pred,
        target[:, :, 0],
        torch.ones(1, 4, 12, dtype=torch.bool),
        torch.ones(1, 4),
        loss_type="smooth_l1",
    )["loss_forecast"]
    loss.backward()
    offset_grad = model.blocks[0].lidar_attn.offsets.weight.grad
    box_grad = model.pose_update.box_head.weight.grad
    assert feat.grad is not None and feat.grad.abs().sum() > 0
    assert offset_grad is not None and offset_grad.abs().sum() > 0
    assert box_grad is not None and box_grad.abs().sum() > 0
    print("blocks", len(pred["aux_outputs"]))
    print("future", tuple(pred["future_xy"].shape))
    print("loss", float(loss))
    print("feature_grad", float(feat.grad.abs().mean()))
    print("lidar_offset_grad", float(offset_grad.abs().mean()))
    print("box_head_grad", float(box_grad.abs().mean()))


if __name__ == "__main__":
    main()
