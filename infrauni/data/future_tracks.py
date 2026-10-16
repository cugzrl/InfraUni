"""Future centers for nuScenes keyframes, expressed in the current LiDAR frame.

Ground-truth instance ids are used only to build this supervision. They are
not model inputs.
"""

from __future__ import annotations

import pickle

import numpy as np
from nuscenes.nuscenes import NuScenes
from pyquaternion import Quaternion

from .constants import CLASS_TO_ID, NUSCENES_ROOT

FUTURE_STEPS = 12


def _global_to_lidar(info, xyz):
    ego_rot = Quaternion(info["ego2global_rotation"]).rotation_matrix
    ego_t = np.asarray(info["ego2global_translation"], dtype=np.float64)
    lidar_rot = Quaternion(info["lidar2ego_rotation"]).rotation_matrix
    lidar_t = np.asarray(info["lidar2ego_translation"], dtype=np.float64)
    point = np.asarray(xyz, dtype=np.float64) - ego_t
    point = ego_rot.T @ point
    point = lidar_rot.T @ (point - lidar_t)
    return point


def _frame_dt(nusc, sample):
    """Seconds from this keyframe to each of the next 12 keyframes.

    A missing sample leaves 0. nuScenes keyframes are 2 Hz, so a full horizon
    is 0.5, 1.0, ..., 6.0 seconds.
    """

    dt = np.zeros(FUTURE_STEPS, dtype=np.float32)
    cursor = sample
    origin = sample["timestamp"]
    for step in range(FUTURE_STEPS):
        if cursor["next"] == "":
            break
        cursor = nusc.get("sample", cursor["next"])
        dt[step] = (cursor["timestamp"] - origin) * 1e-6
    return dt


def build_future_tracks(info_paths, out_path, version="v1.0-mini", root=NUSCENES_ROOT):
    nusc = NuScenes(version=version, dataroot=root, verbose=False)
    tracks = {}
    for path in info_paths:
        infos = pickle.load(open(path, "rb"))["infos"]
        for info in infos:
            sample = nusc.get("sample", info["token"])
            anns = [nusc.get("sample_annotation", token) for token in sample["anns"]]
            ann_xy = np.stack([_global_to_lidar(info, ann["translation"])[:2] for ann in anns]) if anns else np.zeros((0, 2))
            frame_dt = _frame_dt(nusc, sample)
            current, future, mask, labels = [], [], [], []
            for box, name in zip(info["gt_boxes"], info["gt_names"]):
                if name not in CLASS_TO_ID or len(anns) == 0:
                    continue
                nearest = int(np.argmin(np.linalg.norm(ann_xy - box[:2], axis=1)))
                if np.linalg.norm(ann_xy[nearest] - box[:2]) > 0.5:
                    continue
                instance = anns[nearest]["instance_token"]
                cursor = sample
                step_xy, step_mask = [], []
                for _ in range(FUTURE_STEPS):
                    if cursor["next"] == "":
                        step_xy.append([0.0, 0.0])
                        step_mask.append(0)
                        continue
                    cursor = nusc.get("sample", cursor["next"])
                    found = None
                    for token in cursor["anns"]:
                        ann = nusc.get("sample_annotation", token)
                        if ann["instance_token"] == instance:
                            found = ann
                            break
                    if found is None:
                        step_xy.append([0.0, 0.0])
                        step_mask.append(0)
                    else:
                        step_xy.append(_global_to_lidar(info, found["translation"])[:2].tolist())
                        step_mask.append(1)
                current.append(box[:2])
                future.append(step_xy)
                mask.append(step_mask)
                labels.append(CLASS_TO_ID[name])
            tracks[info["token"]] = {
                "current": np.asarray(current, dtype=np.float32).reshape(-1, 2),
                "future": np.asarray(future, dtype=np.float32).reshape(-1, FUTURE_STEPS, 2),
                "mask": np.asarray(mask, dtype=np.float32).reshape(-1, FUTURE_STEPS),
                "labels": np.asarray(labels, dtype=np.int64),
                "dt": frame_dt,
            }
    pickle.dump(tracks, open(out_path, "wb"))
    covered = sum(item["mask"][:, -1].sum() for item in tracks.values())
    return len(tracks), int(covered)
