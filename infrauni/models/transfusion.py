"""Build the official TransFusion detection head and read its queries."""

from __future__ import annotations

import copy

import torch
import torch.nn.functional as F
from mmcv import Config

from ..compat import load_transfusion_head_cls
from ..data.constants import (
    CLASS_NAMES,
    OUT_SIZE_FACTOR,
    PC_RANGE,
    POST_CENTER_RANGE,
    VOXEL_SIZE,
)


def _ensure_registries():
    import mmdet3d.core.bbox.assigners.hungarian_assigner  # noqa: F401
    import mmdet3d.core.bbox.coders.transfusion_bbox_coder  # noqa: F401
    import mmdet3d.core.bbox.iou_calculators.iou3d_calculator  # noqa: F401


def build_transfusion_head(num_proposals=200, voxel_size=None, grid_size=None):
    _ensure_registries()
    voxel_size = list(voxel_size or VOXEL_SIZE)
    if grid_size is None:
        span = [PC_RANGE[i + 3] - PC_RANGE[i] for i in range(3)]
        grid_size = [int(round(span[i] / voxel_size[i])) for i in range(3)]
    head_cls = load_transfusion_head_cls()
    train_cfg = Config(
        dict(
            dataset="nuScenes",
            assigner=dict(
                type="HungarianAssigner3D",
                iou_calculator=dict(type="BboxOverlaps3D", coordinate="lidar"),
                cls_cost=dict(type="FocalLossCost", gamma=2, alpha=0.25, weight=0.15),
                reg_cost=dict(type="BBoxBEVL1Cost", weight=0.25),
                iou_cost=dict(type="IoU3DCost", weight=0.25),
            ),
            pos_weight=-1,
            gaussian_overlap=0.1,
            min_radius=2,
            grid_size=grid_size,
            voxel_size=voxel_size,
            out_size_factor=OUT_SIZE_FACTOR,
            code_weights=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.2, 0.2],
            point_cloud_range=PC_RANGE,
        )
    )
    test_cfg = Config(
        dict(
            dataset="nuScenes",
            grid_size=grid_size,
            out_size_factor=OUT_SIZE_FACTOR,
            pc_range=PC_RANGE[:2],
            voxel_size=voxel_size[:2],
            nms_type=None,
        )
    )
    head = head_cls(
        fuse_img=True,
        num_views=6,
        in_channels_img=256,
        out_size_factor_img=4,
        num_proposals=num_proposals,
        auxiliary=True,
        in_channels=512,
        hidden_channel=128,
        num_classes=len(CLASS_NAMES),
        num_decoder_layers=1,
        num_heads=8,
        learnable_query_pos=False,
        initialize_by_heatmap=True,
        nms_kernel_size=3,
        ffn_channel=256,
        dropout=0.1,
        bn_momentum=0.1,
        activation="relu",
        common_heads=dict(center=(2, 2), height=(1, 2), dim=(3, 2), rot=(2, 2), vel=(2, 2)),
        bbox_coder=dict(
            type="TransFusionBBoxCoder",
            pc_range=PC_RANGE[:2],
            voxel_size=voxel_size[:2],
            out_size_factor=OUT_SIZE_FACTOR,
            post_center_range=POST_CENTER_RANGE,
            score_threshold=0.0,
            code_size=10,
        ),
        loss_cls=dict(type="FocalLoss", use_sigmoid=True, gamma=2, alpha=0.25, reduction="mean", loss_weight=1.0),
        loss_iou=dict(type="CrossEntropyLoss", use_sigmoid=True, reduction="mean", loss_weight=0.0),
        loss_bbox=dict(type="L1Loss", reduction="mean", loss_weight=0.25),
        loss_heatmap=dict(type="GaussianFocalLoss", reduction="mean", loss_weight=1.0),
        train_cfg=train_cfg,
        test_cfg=test_cfg,
    )

    def _capture(module, inputs, output):
        head._raw_query = inputs[0]

    def _capture_lidar_heatmap(module, inputs, output):
        head._lidar_heatmap = output

    head.prediction_heads[-1].register_forward_hook(_capture)
    head.heatmap_head.register_forward_hook(_capture_lidar_heatmap)
    return head


def fused_object_query(head):
    """Image-refined query, falling back to the LiDAR query off-camera."""
    raw = head._raw_query
    channels = head.shared_conv.out_channels
    image_query, lidar_query = raw[:, :channels], raw[:, channels:]
    mask = head.on_the_image_mask[:, None, :]
    return torch.where(mask, image_query, lidar_query)


def gaussian_heatmap_targets(head, gt_bboxes, gt_labels, heatmap_logits):
    """Same CenterPoint gaussian targets the official head uses for the image heatmap."""
    from mmdet3d.core import draw_heatmap_gaussian, gaussian_radius

    device = heatmap_logits.device
    channels, height, width = heatmap_logits.shape[1:]
    target = heatmap_logits.new_zeros(heatmap_logits.shape[0], channels, height, width)
    pc_range = heatmap_logits.new_tensor(head.train_cfg["point_cloud_range"])
    voxel = heatmap_logits.new_tensor(head.train_cfg["voxel_size"])
    factor = head.train_cfg["out_size_factor"]
    for batch_index, boxes in enumerate(gt_bboxes):
        labels = gt_labels[batch_index].to(device)
        if len(boxes) == 0:
            continue
        centers = torch.cat([boxes.gravity_center, boxes.tensor[:, 3:]], dim=1).to(device)
        for index in range(len(centers)):
            box_w = centers[index, 3] / voxel[0] / factor
            box_l = centers[index, 4] / voxel[1] / factor
            if box_w <= 0 or box_l <= 0:
                continue
            radius = gaussian_radius((box_l, box_w), min_overlap=head.train_cfg["gaussian_overlap"])
            radius = max(int(head.train_cfg["min_radius"]), int(radius))
            coor_x = (centers[index, 0] - pc_range[0]) / voxel[0] / factor
            coor_y = (centers[index, 1] - pc_range[1]) / voxel[1] / factor
            draw_heatmap_gaussian(target[batch_index, labels[index]], torch.tensor([coor_x, coor_y], device=device).int(), radius)
    return target


def detection_loss(head, gt_bboxes, gt_labels, preds):
    """Official head loss, still supervising boxes that miss every camera.

    The local checkout multiplies every positive weight by ``on_the_image_mask``.
    Queries that miss the cameras keep the LiDAR prediction, so they still need
    a detection loss.
    """
    saved = head.on_the_image_mask
    del head.on_the_image_mask
    try:
        losses = head.loss(gt_bboxes, gt_labels, preds)
    finally:
        head.on_the_image_mask = saved
    # The fusion checkpoint is absent, so the LiDAR heatmap never receives a
    # loss in the official forward (only the image heatmap is stored). Query
    # initialization still averages the two maps, so the LiDAR map is trained.
    if getattr(head, "_lidar_heatmap", None) is not None:
        target = gaussian_heatmap_targets(head, gt_bboxes, gt_labels, head._lidar_heatmap)
        positive = max(target.eq(1).float().sum().item(), 1)
        losses["loss_heatmap_lidar"] = head.loss_heatmap(head._lidar_heatmap.sigmoid().clamp(1e-4, 1 - 1e-4), target, avg_factor=positive)
    return losses


@torch.no_grad()
def decode_proposals(head, preds):
    """Decode every proposal. ``decode`` mutates its inputs, so pass copies."""
    layer = preds[0][0]
    count = head.num_proposals
    heatmap = layer["heatmap"][..., -count:].sigmoid()
    one_hot = F.one_hot(head.query_labels, num_classes=head.num_classes).permute(0, 2, 1).float()
    score = heatmap * layer["query_heatmap_score"] * one_hot
    decoded = head.bbox_coder.decode(
        copy.deepcopy(layer["heatmap"][..., -count:]),
        copy.deepcopy(layer["rot"][..., -count:]),
        copy.deepcopy(layer["dim"][..., -count:]),
        copy.deepcopy(layer["center"][..., -count:]),
        copy.deepcopy(layer["height"][..., -count:]),
        copy.deepcopy(layer["vel"][..., -count:]),
        filter=False,
    )
    boxes = decoded[0]["bboxes"]
    labels = score.argmax(dim=1)[0]
    scores = score.max(dim=1).values[0]
    return boxes, scores, labels
