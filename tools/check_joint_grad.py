"""Check that the forecast loss reaches DeTra and the detector feature path."""

import os
import sys

import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path[:0] = [
    ROOT,
    os.path.join(ROOT, "third_party", "MC-DeTra"),
    os.path.join(ROOT, "third_party", "TransFusion"),
]

from mmcv import Config
from mmcv.parallel import collate, scatter
from mmcv.runner import load_checkpoint
from mmdet3d.datasets import build_dataset
from mmdet3d.models import build_detector

from detra_repro.losses import wta_forecasting_loss
from infrauni.joint.model import JointTransFusionDeTra
from infrauni.joint.targets import align_future


def _nonzero(module):
    if module is None:
        return False
    return any(p.grad is not None and float(p.grad.abs().sum()) > 0 for p in module.parameters())


def main():
    os.chdir(os.path.join(ROOT, "third_party", "TransFusion"))
    cfg = Config.fromfile(os.path.join(ROOT, "configs", "transfusion_mini_voxel_LC.py"))
    dataset = build_dataset(cfg.data.val)
    detector = build_detector(cfg.model, train_cfg=cfg.get("train_cfg"), test_cfg=cfg.get("test_cfg"))
    load_checkpoint(
        detector,
        os.path.join(ROOT, "experiments", "transfusion_voxel_LC_mini", "epoch_1.pth"),
        map_location="cpu",
    )
    model = JointTransFusionDeTra(detector, mode="joint", blocks=2).cuda()
    for param in model.detector.parameters():
        param.requires_grad_(False)
    # These two modules produce the query and the BEV map consumed by DeTra.
    head = model.detector.pts_bbox_head
    for module in (model.detector.pts_neck, head.decoder[1], head.prediction_heads[-1]):
        for param in module.parameters():
            param.requires_grad_(True)
    batch = scatter(collate([dataset[0]], samples_per_gpu=1), [0])[0]
    batch = {"points": batch["points"][0], "img": batch["img"][0], "img_metas": batch["img_metas"][0]}
    output = model(points=batch["points"], img=batch["img"], img_metas=batch["img_metas"])
    token = batch["img_metas"][0]["sample_idx"]
    import pickle

    track = pickle.load(open(os.path.join(ROOT, "data", "nuscenes_mini_future12.pkl"), "rb"))[token]
    future, mask, targets, _ = align_future(output["boxes"][0, :, :2], track)
    loss = wta_forecasting_loss(output["detra"], future, mask, targets, loss_type="smooth_l1")["loss_forecast"]
    loss.backward()
    print("forecast_loss", float(loss))
    print("detra_box", _nonzero(model.detra.pose_update))
    print("detra_lidar_attn", _nonzero(model.detra.blocks[0].lidar_attn))
    print("bev_proj", _nonzero(model.bev_proj))
    print("query_proj", _nonzero(model.query_proj))
    print("pts_neck", _nonzero(model.detector.pts_neck))
    print("img_query_decoder", _nonzero(model.detector.pts_bbox_head.decoder[1]))
    print("pred_head", _nonzero(model.detector.pts_bbox_head.prediction_heads[-1]))
    print("blocks", len(output["detra"]["aux_outputs"]))


if __name__ == "__main__":
    main()
