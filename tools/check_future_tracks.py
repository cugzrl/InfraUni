"""Check future tracks against nuScenes timestamps and the current LiDAR frame."""

import os
import pickle
import sys

import numpy as np
from nuscenes.nuscenes import NuScenes
from pyquaternion import Quaternion

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from infrauni.data.constants import CLASS_TO_ID, NUSCENES_ROOT
from infrauni.data.future_tracks import FUTURE_STEPS, _frame_dt, _global_to_lidar

INFO = os.path.join(ROOT, "third_party", "TransFusion", "data", "nuscenes", "nuscenes_infos_train.pkl")
CACHE = os.path.join(ROOT, "data", "nuscenes_mini_future12.pkl")


def _lidar_to_global(info, xyz):
    lidar_rot = Quaternion(info["lidar2ego_rotation"]).rotation_matrix
    lidar_t = np.asarray(info["lidar2ego_translation"], dtype=np.float64)
    ego_rot = Quaternion(info["ego2global_rotation"]).rotation_matrix
    ego_t = np.asarray(info["ego2global_translation"], dtype=np.float64)
    point = lidar_rot @ np.asarray(xyz, dtype=np.float64) + lidar_t
    return ego_rot @ point + ego_t


def main():
    infos = pickle.load(open(INFO, "rb"))["infos"]
    cache = pickle.load(open(CACHE, "rb"))
    nusc = NuScenes(version="v1.0-mini", dataroot=NUSCENES_ROOT, verbose=False)
    xy_err = []
    roundtrip = []
    dts = []
    scene_end = 0
    gaps = 0
    checked = 0
    for info in infos:
        sample = nusc.get("sample", info["token"])
        track = cache[info["token"]]
        dt = _frame_dt(nusc, sample)
        if "dt" in track:
            if not np.allclose(track["dt"], dt):
                raise SystemExit("stored dt does not match the sample timestamps")
        valid_dt = dt[dt > 0]
        if len(valid_dt):
            dts.append(valid_dt)
        if sample["next"] == "":
            scene_end += 1
            if track["mask"].size and track["mask"].any():
                raise SystemExit("scene-end frame still has a future label")
        anns = [nusc.get("sample_annotation", token) for token in sample["anns"]]
        if not anns or len(track["current"]) == 0:
            continue
        ann_xy = np.stack([_global_to_lidar(info, ann["translation"])[:2] for ann in anns])
        # One object per checked frame is enough to audit the transform.
        row = 0
        box = track["current"][row]
        label = int(track["labels"][row])
        nearest = int(np.argmin(np.linalg.norm(ann_xy - box, axis=1)))
        if np.linalg.norm(ann_xy[nearest] - box) > 0.5:
            continue
        name_index = int(np.argmin(np.linalg.norm(info["gt_boxes"][:, :2] - box, axis=1)))
        if CLASS_TO_ID[str(info["gt_names"][name_index])] != label:
            raise SystemExit("cached label does not match the info class")
        instance = anns[nearest]["instance_token"]
        cursor = sample
        for step in range(FUTURE_STEPS):
            if cursor["next"] == "":
                if track["mask"][row, step] != 0:
                    raise SystemExit("mask stays valid past the scene boundary")
                continue
            cursor = nusc.get("sample", cursor["next"])
            found = None
            for token in cursor["anns"]:
                ann = nusc.get("sample_annotation", token)
                if ann["instance_token"] == instance:
                    found = ann
                    break
            if found is None:
                gaps += 1
                if track["mask"][row, step] != 0:
                    raise SystemExit("a dropped instance is still supervised")
                continue
            fresh = _global_to_lidar(info, found["translation"])
            xy_err.append(np.linalg.norm(fresh[:2] - track["future"][row, step]))
            back = _lidar_to_global(info, np.array([track["future"][row, step][0], track["future"][row, step][1], fresh[2]]))
            roundtrip.append(np.linalg.norm(back[:2] - np.asarray(found["translation"])[:2]))
            if track["mask"][row, step] != 1:
                raise SystemExit("a visible future instance is masked out")
        checked += 1
        if checked >= 12:
            break
    stacked = np.concatenate(dts) if dts else np.zeros(0)
    # Nominal 2 Hz keyframes. nuScenes timestamps jitter by about 50 ms.
    if dts:
        gaps_dt = np.concatenate([np.diff(np.concatenate([[0.0], series])) for series in dts])
        if gaps_dt.min() < 0.4 or gaps_dt.max() > 0.6:
            raise SystemExit("keyframe interval outside 0.4-0.6 s: %s %s" % (gaps_dt.min(), gaps_dt.max()))
    else:
        gaps_dt = np.zeros(0)
    scene_end = 0
    for info in infos:
        sample = nusc.get("sample", info["token"])
        if sample["next"] != "":
            continue
        scene_end += 1
        if cache[info["token"]]["mask"].any():
            raise SystemExit("scene-end frame still has a future label")
    print(
        "checked_frames", checked,
        "scene_end_frames", scene_end,
        "instance_gaps", gaps,
        "max_lidar_xy_err_m", float(np.max(xy_err)) if xy_err else None,
        "max_roundtrip_xy_m", float(np.max(roundtrip)) if roundtrip else None,
        "dt_min", float(stacked.min()) if stacked.size else None,
        "dt_max", float(stacked.max()) if stacked.size else None,
        "interval_min", float(gaps_dt.min()) if gaps_dt.size else None,
        "interval_max", float(gaps_dt.max()) if gaps_dt.size else None,
    )


if __name__ == "__main__":
    main()
