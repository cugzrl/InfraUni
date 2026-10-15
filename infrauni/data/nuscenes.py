"""nuScenes samples for TransFusion, with futures in the current LiDAR frame.

The sample dict is also the contract later roadside datasets should fill.
See ``infrauni/data/sample.py``.
"""

from __future__ import annotations

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from mmdet3d.core.bbox import LiDARInstance3DBoxes
from mmdet3d.core.points import LiDARPoints
from mmdet3d.datasets.pipelines.loading import LoadPointsFromFile, LoadPointsFromMultiSweeps

from .constants import (
    CAMERA_TYPES,
    CLASS_TO_ID,
    IMG_MEAN,
    IMG_STD,
    NUM_FUTURE,
    PC_RANGE,
)

_POINT_LOADER = LoadPointsFromFile(coord_type="LIDAR", load_dim=5, use_dim=[0, 1, 2, 3, 4])


def _sweep_loader(train):
    return LoadPointsFromMultiSweeps(
        sweeps_num=10,
        load_dim=5,
        use_dim=[0, 1, 2, 3, 4],
        pad_empty_sweeps=True,
        remove_close=True,
        test_mode=not train,
    )


def _resize_pad(image, max_size, divisor=32):
    """Keep-ratio resize into ``max_size`` (W, H), then pad to a multiple of 32."""
    width, height = image.size
    max_w, max_h = max_size
    scale = min(max_w / width, max_h / height)
    resized = (max(1, int(width * scale)), max(1, int(height * scale)))
    image = image.resize(resized, Image.BILINEAR)
    pad_w = int(np.ceil(resized[0] / divisor) * divisor)
    pad_h = int(np.ceil(resized[1] / divisor) * divisor)
    canvas = Image.new("RGB", (pad_w, pad_h))
    canvas.paste(image, (0, 0))
    return canvas, scale, (resized[1], resized[0]), (pad_h, pad_w)


def _lidar2img(camera_info):
    """4x4 matrix mapping current-LiDAR points onto the raw image."""
    sensor2lidar_r = np.asarray(camera_info["sensor2lidar_rotation"], dtype=np.float32)
    sensor2lidar_t = np.asarray(camera_info["sensor2lidar_translation"], dtype=np.float32)
    lidar2camera_r = np.linalg.inv(sensor2lidar_r)
    lidar2camera_t = sensor2lidar_t @ lidar2camera_r.T
    lidar2camera = np.eye(4, dtype=np.float32)
    lidar2camera[:3, :3] = lidar2camera_r.T
    lidar2camera[3, :3] = -lidar2camera_t
    intrinsics = np.eye(4, dtype=np.float32)
    intrinsics[:3, :3] = np.asarray(camera_info["camera_intrinsics"], dtype=np.float32)
    return intrinsics @ lidar2camera.T


def _rotate_xy(xy, angle):
    """Apply the same right-multiply used by ``LiDARInstance3DBoxes.rotate``."""
    if xy.numel() == 0:
        return xy
    cosine, sine = np.cos(angle), np.sin(angle)
    matrix = xy.new_tensor([[cosine, -sine], [sine, cosine]])
    return xy @ matrix


class NuScenesDetectionDataset(Dataset):
    def __init__(self, info_path, train=True, image_size=(800, 448), augment=True):
        import mmcv

        payload = mmcv.load(info_path)
        infos = list(payload["infos"])
        infos.sort(key=lambda item: item["timestamp"])
        self.infos = infos
        self.train = train
        self.augment = augment and train
        self.image_size = tuple(image_size)
        self.sweeps = _sweep_loader(train)
        self.metadata = payload.get("metadata", {})

    def __len__(self):
        return len(self.infos)

    def __getitem__(self, index):
        if not self.train:
            return self._build(index)
        for shift in range(8):
            sample = self._build((index + shift) % len(self.infos))
            if sample["gt_labels_3d"].numel() > 0 and sample["points"].shape[0] > 0:
                return sample
        return sample

    def _build(self, index):
        info = self.infos[index]
        points = self._load_points(info)
        boxes, labels, future, future_mask = self._load_boxes(info)
        meta = self._empty_aug_meta()
        if self.augment:
            points, boxes, future, meta = self._augment(points, boxes, future, meta)
        points, boxes, labels, future, future_mask = self._range_filter(
            points, boxes, labels, future, future_mask
        )
        images, meta = self._load_images(info, meta)
        meta["token"] = info["token"]
        meta["box_type_3d"] = LiDARInstance3DBoxes
        return {
            "points": points.tensor[:, :5].contiguous(),
            "img": images,
            "img_meta": meta,
            "gt_bboxes_3d": boxes,
            "gt_labels_3d": labels,
            "gt_fut": future,
            "gt_fut_mask": future_mask,
            "token": info["token"],
        }

    def _load_points(self, info):
        loaded = _POINT_LOADER({"lidar_path": info["lidar_path"]})
        loaded["sweeps"] = info["sweeps"]
        loaded["timestamp"] = info["timestamp"]
        loaded = self.sweeps(loaded)
        return loaded["points"]

    def _load_boxes(self, info):
        mask = np.asarray(info["valid_flag"], dtype=bool)
        names = np.asarray(info["gt_names"])[mask]
        boxes = np.asarray(info["gt_boxes"], dtype=np.float32)[mask]
        velocity = np.asarray(info["gt_velocity"], dtype=np.float32)[mask]
        future = np.asarray(info["gt_fut"], dtype=np.float32)[mask]
        future_mask = np.asarray(info["gt_fut_mask"], dtype=np.float32)[mask]
        keep = np.array([name in CLASS_TO_ID for name in names], dtype=bool)
        names, boxes, velocity = names[keep], boxes[keep], velocity[keep]
        future, future_mask = future[keep], future_mask[keep]
        velocity = np.nan_to_num(velocity, nan=0.0)
        tensor = torch.from_numpy(np.concatenate([boxes, velocity], axis=1))
        labels = torch.tensor([CLASS_TO_ID[name] for name in names], dtype=torch.long)
        gt_boxes = LiDARInstance3DBoxes(tensor, box_dim=9, origin=(0.5, 0.5, 0.5))
        return gt_boxes, labels, torch.from_numpy(future), torch.from_numpy(future_mask)

    def _empty_aug_meta(self):
        return {
            "pcd_rotation": np.eye(3, dtype=np.float32),
            "pcd_scale_factor": 1.0,
            "pcd_trans": np.zeros(3, dtype=np.float32),
            "pcd_horizontal_flip": False,
            "pcd_vertical_flip": False,
            "transformation_3d_flow": [],
            "flip": False,
            "img_crop_offset": np.zeros(2, dtype=np.float32),
        }

    def _augment(self, points, boxes, future, meta):
        angle = float(np.random.uniform(-0.78539816, 0.78539816))
        scale = float(np.random.uniform(0.9, 1.1))
        translation = np.random.normal(0.0, [0.5, 0.5, 0.5]).astype(np.float32)
        horizontal = bool(np.random.rand() < 0.5)
        vertical = bool(np.random.rand() < 0.5)

        # points.rotate(-angle) matches boxes.rotate(angle); see LiDAR box docs.
        rotation = points.rotate(-angle)
        if len(boxes):
            boxes.rotate(angle)
        future = _rotate_xy(future, angle)

        points.scale(scale)
        if len(boxes):
            boxes.scale(scale)
        future = future * scale

        points.translate(translation)
        if len(boxes):
            boxes.translate(translation)
        future = future + future.new_tensor(translation[:2])

        if horizontal:
            points.flip("horizontal")
            if len(boxes):
                boxes.flip("horizontal")
            future = future.clone()
            future[..., 1] *= -1
        if vertical:
            points.flip("vertical")
            if len(boxes):
                boxes.flip("vertical")
            future = future.clone()
            future[..., 0] *= -1

        meta["pcd_rotation"] = rotation.detach().cpu().numpy().astype(np.float32)
        meta["pcd_scale_factor"] = scale
        meta["pcd_trans"] = translation
        meta["pcd_horizontal_flip"] = horizontal
        meta["pcd_vertical_flip"] = vertical
        flow = ["R", "S", "T"]
        if horizontal:
            flow.append("HF")
        if vertical:
            flow.append("VF")
        meta["transformation_3d_flow"] = flow
        return points, boxes, future, meta

    def _range_filter(self, points, boxes, labels, future, future_mask):
        point_keep = points.in_range_3d(PC_RANGE)
        points = points[point_keep]
        if len(boxes) == 0:
            return points, boxes, labels, future, future_mask
        box_keep = boxes.in_range_bev([PC_RANGE[0], PC_RANGE[1], PC_RANGE[3], PC_RANGE[4]])
        return points, boxes[box_keep], labels[box_keep], future[box_keep], future_mask[box_keep]

    def _load_images(self, info, meta):
        views = []
        matrices = []
        scale = None
        resized_hw = None
        for camera in CAMERA_TYPES:
            camera_info = info["cams"][camera]
            image = Image.open(camera_info["data_path"]).convert("RGB")
            image, scale, resized_hw, _ = _resize_pad(image, self.image_size)
            array = np.asarray(image, dtype=np.float32)
            tensor = torch.from_numpy(array).permute(2, 0, 1)
            mean = tensor.new_tensor(IMG_MEAN)[:, None, None]
            std = tensor.new_tensor(IMG_STD)[:, None, None]
            views.append((tensor - mean) / std)
            matrices.append(_lidar2img(camera_info))
        images = torch.stack(views, dim=0)
        meta["lidar2img"] = np.stack(matrices, axis=0)
        meta["scale_factor"] = np.array([scale, scale, scale, scale], dtype=np.float32)
        meta["img_shape"] = (resized_hw[0], resized_hw[1], 3)
        meta["input_shape"] = (images.shape[-2], images.shape[-1], 3)
        meta["pad_shape"] = meta["input_shape"]
        return images, meta


def collate_nuscenes(batch):
    return {
        "points": [item["points"] for item in batch],
        "img": torch.stack([item["img"] for item in batch], dim=0),
        "img_metas": [item["img_meta"] for item in batch],
        "gt_bboxes_3d": [item["gt_bboxes_3d"] for item in batch],
        "gt_labels_3d": [item["gt_labels_3d"] for item in batch],
        "gt_fut": [item["gt_fut"] for item in batch],
        "gt_fut_mask": [item["gt_fut_mask"] for item in batch],
        "token": [item["token"] for item in batch],
    }
