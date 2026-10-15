"""nuScenes-mini samples with current boxes and future waypoints.

Sample contract (also the target interface for RESOLVE / V2X-Seq):
    points      (N, 4) float32   x, y, z, intensity in the LiDAR frame
    image       (3, H, W) float32
    lidar2cam   (4, 4) float32
    K           (3, 3) float32   scaled to the resized image
    boxes       (M, 7) float32   x, y, z, w, l, h, yaw
    labels      (M,) int64
    traj        (M, T, 2) float32  future xy in the current LiDAR frame
    traj_mask   (M, T) float32
    token       str
"""

from __future__ import annotations

import json
import pickle
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from infrauni.data.constants import (
    CATEGORY_TO_CLASS,
    CLASS_NAMES,
    MINI_TRAIN_SCENES,
    MINI_VAL_SCENES,
)
from infrauni.data.geometry import global_box_to_lidar, transform_matrix

CLASS_TO_ID = {name: i for i, name in enumerate(CLASS_NAMES)}


def _load_table(root: Path, version: str, name: str):
    with open(root / version / f"{name}.json", "r") as f:
        return json.load(f)


def _index_by_token(rows):
    return {row["token"]: row for row in rows}


def _scene_sample_tokens(scene, samples):
    tokens = []
    token = scene["first_sample_token"]
    while token:
        tokens.append(token)
        token = samples[token]["next"]
    return tokens


def _in_range(center, pc_range):
    x, y, z = center
    return (
        pc_range[0] <= x <= pc_range[3]
        and pc_range[1] <= y <= pc_range[4]
        and pc_range[2] <= z <= pc_range[5]
    )


def build_infos(cfg, project_root: Path) -> dict:
    """Parse nuScenes JSON once and cache boxes / trajectories."""
    root = Path(cfg["data"]["dataroot"])
    version = cfg["data"].get("version", "v1.0-mini")
    num_future = int(cfg["num_future"])
    pc_range = cfg["pc_range"]

    scenes = _load_table(root, version, "scene")
    samples = _index_by_token(_load_table(root, version, "sample"))
    sample_data = _load_table(root, version, "sample_data")
    ego_poses = _index_by_token(_load_table(root, version, "ego_pose"))
    calibs = _index_by_token(_load_table(root, version, "calibrated_sensor"))
    sensors = _index_by_token(_load_table(root, version, "sensor"))
    anns = _load_table(root, version, "sample_annotation")
    instances = _index_by_token(_load_table(root, version, "instance"))
    categories = _index_by_token(_load_table(root, version, "category"))
    ann_by_token = _index_by_token(anns)

    anns_by_sample = defaultdict(list)
    for ann in anns:
        anns_by_sample[ann["sample_token"]].append(ann)

    keyframes = defaultdict(dict)
    for row in sample_data:
        if not row["is_key_frame"]:
            continue
        channel = sensors[calibs[row["calibrated_sensor_token"]]["sensor_token"]]["channel"]
        if channel in ("LIDAR_TOP", "CAM_FRONT"):
            keyframes[row["sample_token"]][channel] = row

    def sensor_to_global(row):
        calib = calibs[row["calibrated_sensor_token"]]
        ego = ego_poses[row["ego_pose_token"]]
        sensor_from_ego = transform_matrix(calib["translation"], calib["rotation"])
        ego_from_global = transform_matrix(ego["translation"], ego["rotation"])
        return ego_from_global @ sensor_from_ego

    split_names = {"train": set(MINI_TRAIN_SCENES), "val": set(MINI_VAL_SCENES)}
    infos = {"train": [], "val": []}
    class_hist = {name: 0 for name in CLASS_NAMES}

    for scene in scenes:
        split = None
        for name, scene_set in split_names.items():
            if scene["name"] in scene_set:
                split = name
                break
        if split is None:
            continue
        for sample_token in _scene_sample_tokens(scene, samples):
            frames = keyframes.get(sample_token, {})
            if "LIDAR_TOP" not in frames or "CAM_FRONT" not in frames:
                continue
            lidar_sd = frames["LIDAR_TOP"]
            cam_sd = frames["CAM_FRONT"]
            lidar_path = root / lidar_sd["filename"]
            image_path = root / cam_sd["filename"]
            if not lidar_path.is_file() or not image_path.is_file():
                continue

            global_from_lidar = sensor_to_global(lidar_sd)
            global_from_cam = sensor_to_global(cam_sd)
            lidar2cam = np.linalg.inv(global_from_cam) @ global_from_lidar
            cam_calib = calibs[cam_sd["calibrated_sensor_token"]]
            K = np.asarray(cam_calib["camera_intrinsic"], dtype=np.float64)

            boxes, labels, trajs, masks = [], [], [], []
            for ann in anns_by_sample.get(sample_token, []):
                if ann["num_lidar_pts"] <= 0:
                    continue
                cat_token = instances[ann["instance_token"]]["category_token"]
                cat_name = categories[cat_token]["name"]
                cls_name = CATEGORY_TO_CLASS.get(cat_name)
                if cls_name is None:
                    continue
                box = global_box_to_lidar(
                    ann["translation"],
                    ann["size"],
                    ann["rotation"],
                    global_from_lidar,
                )
                if not _in_range(box[:3], pc_range):
                    continue
                future = np.zeros((num_future, 2), dtype=np.float32)
                future_mask = np.zeros((num_future,), dtype=np.float32)
                cursor = ann
                for step in range(num_future):
                    nxt = cursor["next"]
                    if not nxt or nxt not in ann_by_token:
                        break
                    cursor = ann_by_token[nxt]
                    nxt_box = global_box_to_lidar(
                        cursor["translation"],
                        cursor["size"],
                        cursor["rotation"],
                        global_from_lidar,
                    )
                    future[step] = nxt_box[:2]
                    future_mask[step] = 1.0
                boxes.append(box)
                labels.append(CLASS_TO_ID[cls_name])
                trajs.append(future)
                masks.append(future_mask)
                class_hist[cls_name] += 1

            infos[split].append(
                {
                    "token": sample_token,
                    "scene": scene["name"],
                    "lidar_path": str(lidar_path),
                    "image_path": str(image_path),
                    "img_hw": (int(cam_sd["height"]), int(cam_sd["width"])),
                    "lidar2cam": lidar2cam.astype(np.float32),
                    "K": K.astype(np.float32),
                    "boxes": np.zeros((0, 7), np.float32)
                    if not boxes
                    else np.stack(boxes).astype(np.float32),
                    "labels": np.zeros((0,), np.int64)
                    if not labels
                    else np.asarray(labels, dtype=np.int64),
                    "traj": np.zeros((0, num_future, 2), np.float32)
                    if not trajs
                    else np.stack(trajs).astype(np.float32),
                    "traj_mask": np.zeros((0, num_future), np.float32)
                    if not masks
                    else np.stack(masks).astype(np.float32),
                }
            )

    info_path = project_root / cfg["data"]["info_path"]
    info_path.parent.mkdir(parents=True, exist_ok=True)
    with open(info_path, "wb") as f:
        pickle.dump({"infos": infos, "class_hist": class_hist}, f)
    return {"infos": infos, "class_hist": class_hist, "path": str(info_path)}


def _scale_intrinsics(K, src_hw, dst_hw):
    scale_y = dst_hw[0] / float(src_hw[0])
    scale_x = dst_hw[1] / float(src_hw[1])
    out = K.copy()
    out[0, :] *= scale_x
    out[1, :] *= scale_y
    return out


class NuScenesMiniJoint(Dataset):
    def __init__(self, infos, cfg, split: str):
        self.rows = infos[split]
        self.cfg = cfg
        self.image_h = int(cfg["image_h"])
        self.image_w = int(cfg["image_w"])
        self.pc_range = cfg["pc_range"]

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        points = np.fromfile(row["lidar_path"], dtype=np.float32).reshape(-1, 5)
        points = points[:, :4]
        points[:, 3] = np.clip(points[:, 3], 0, 255) / 255.0
        x, y, z = points[:, 0], points[:, 1], points[:, 2]
        pc = self.pc_range
        keep = (
            (x >= pc[0])
            & (x <= pc[3])
            & (y >= pc[1])
            & (y <= pc[4])
            & (z >= pc[2])
            & (z <= pc[5])
        )
        points = points[keep]

        image_bgr = cv2.imread(row["image_path"], cv2.IMREAD_COLOR)
        if image_bgr is None:
            raise FileNotFoundError(row["image_path"])
        src_h, src_w = image_bgr.shape[:2]
        image_bgr = cv2.resize(
            image_bgr, (self.image_w, self.image_h), interpolation=cv2.INTER_LINEAR
        )
        image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        image = np.transpose(image, (2, 0, 1))
        K = _scale_intrinsics(row["K"], (src_h, src_w), (self.image_h, self.image_w))

        return {
            "points": points.astype(np.float32),
            "image": image.astype(np.float32),
            "lidar2cam": row["lidar2cam"],
            "K": K.astype(np.float32),
            "boxes": row["boxes"],
            "labels": row["labels"],
            "traj": row["traj"],
            "traj_mask": row["traj_mask"],
            "token": row["token"],
        }


def collate_batch(batch):
    max_points = max(item["points"].shape[0] for item in batch)
    max_points = max(max_points, 1)
    max_boxes = max(item["boxes"].shape[0] for item in batch)
    max_boxes = max(max_boxes, 1)
    batch_size = len(batch)
    num_future = batch[0]["traj"].shape[1] if batch[0]["traj"].ndim == 3 else 0
    if batch[0]["traj"].ndim != 3:
        num_future = batch[0]["traj_mask"].shape[-1] if batch[0]["traj_mask"].size else 6

    points = np.zeros((batch_size, max_points, 4), dtype=np.float32)
    point_mask = np.zeros((batch_size, max_points), dtype=np.bool_)
    images = np.stack([item["image"] for item in batch], axis=0)
    lidar2cam = np.stack([item["lidar2cam"] for item in batch], axis=0)
    K = np.stack([item["K"] for item in batch], axis=0)
    boxes = np.zeros((batch_size, max_boxes, 7), dtype=np.float32)
    labels = np.zeros((batch_size, max_boxes), dtype=np.int64)
    traj = np.zeros((batch_size, max_boxes, num_future, 2), dtype=np.float32)
    traj_mask = np.zeros((batch_size, max_boxes, num_future), dtype=np.float32)
    gt_mask = np.zeros((batch_size, max_boxes), dtype=np.bool_)

    # num_future fallback when the first item is empty
    if batch[0]["traj"].shape[0] == 0:
        for item in batch:
            if item["traj"].shape[0] > 0:
                num_future = item["traj"].shape[1]
                break
        traj = np.zeros((batch_size, max_boxes, num_future, 2), dtype=np.float32)
        traj_mask = np.zeros((batch_size, max_boxes, num_future), dtype=np.float32)

    for i, item in enumerate(batch):
        n = item["points"].shape[0]
        points[i, :n] = item["points"]
        point_mask[i, :n] = True
        m = item["boxes"].shape[0]
        if m == 0:
            continue
        boxes[i, :m] = item["boxes"]
        labels[i, :m] = item["labels"]
        traj[i, :m] = item["traj"]
        traj_mask[i, :m] = item["traj_mask"]
        gt_mask[i, :m] = True

    def as_tensor(array, dtype=None):
        tensor = torch.from_numpy(array)
        return tensor if dtype is None else tensor.to(dtype)

    return {
        "points": as_tensor(points),
        "point_mask": as_tensor(point_mask),
        "image": as_tensor(images),
        "lidar2cam": as_tensor(lidar2cam),
        "K": as_tensor(K),
        "boxes": as_tensor(boxes),
        "labels": as_tensor(labels),
        "traj": as_tensor(traj),
        "traj_mask": as_tensor(traj_mask),
        "gt_mask": as_tensor(gt_mask),
        "tokens": [item["token"] for item in batch],
    }
