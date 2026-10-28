"""Build one RESOLVE sequence of LiDAR-only OpenPCDet infos.

The official converter always enables cameras through ``store_true`` defaults.
This wrapper calls the same functions with ``use_cam=False`` and does not
replace missing point clouds with a neighboring frame.
"""

import argparse
import json
import os
import pickle
import sys
from pathlib import Path

RESOLVE_TOOLS = "/home/kemove/devdata1/zrl/RESOLVE/tools"
sys.path.insert(0, RESOLVE_TOOLS)
sys.path.append("/home/kemove/anaconda3/envs/bevfusion/lib/python3.8/site-packages")

from create_sunlakes_data_v3 import create_dataset_from_json  # noqa: E402
from sunlakes_dataset_utils import rotate_point_cloud, split_dataset_divisible  # noqa: E402

SOURCE = Path("/home/kemove/devdata1/zrl/dataset/RESOLVE")
LAYOUT = Path("/home/kemove/devdata1/zrl/dataset/RESOLVE_lidar_smoke")
FULL_ROOT = Path("/home/kemove/devdata1/zrl/dataset/RESOLVE_layout")
FULL_PREFIX = "2026-01-04_18_02_01-export"
SCENE_SPLIT = Path("/home/kemove/devdata1/zrl/InfraUni/infrauni/data/resolve_scene_split.json")


def _filter_annotation(bag, destination):
    frames = json.load(open(SOURCE / "HSQJN_3.json", "r"))
    kept = [frame for frame in frames if frame["info"].split("/")[-3] == bag]
    destination.parent.mkdir(parents=True, exist_ok=True)
    json.dump(kept, open(destination, "w"))
    missing = 0
    for frame in kept:
        name = frame["info"].split("/")[-1]
        if not (SOURCE / bag / "128_128b_sync" / name).exists():
            missing += 1
    return len(kept), missing


def _link_cloud(bag):
    src = SOURCE / bag / "128_128b_sync"
    dst = LAYOUT / "one_seq" / "resource" / bag / "128_128b_label"
    if dst.exists() or dst.is_symlink():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    os.symlink(src, dst)


def _write_scene_split(dataset, info_dir):
    groups = json.load(open(SCENE_SPLIT))
    buckets = {name: [] for name in ("train", "val", "test")}
    unknown = 0
    for sample in dataset.values():
        name = None
        for part in buckets:
            if sample["scene_id"] in groups[part]:
                name = part
                break
        if name is None:
            unknown += 1
            continue
        buckets[name].append(sample)
    for name, rows in buckets.items():
        pickle.dump(rows, open(info_dir / ("sunlakes_infos_scene_%s.pkl" % name), "wb"))
    print("scene split", {name: len(rows) for name, rows in buckets.items()}, "unknown", unknown)


def convert_full(rot_pc, workers):
    info_dir = FULL_ROOT / "v1.0-trainval"
    info_dir.mkdir(parents=True, exist_ok=True)
    rotate_point_cloud(str(FULL_ROOT), rot_pc, FULL_PREFIX, "os128", workers=workers)
    dataset = create_dataset_from_json(
        str(FULL_ROOT), FULL_PREFIX, rot_pc, "os128", None, True, False, False
    )
    train_keys, val_keys, dropped = split_dataset_divisible(
        dataset, 0.8, m=4, seed=42, drop_from="val"
    )
    pickle.dump([dataset[key] for key in train_keys], open(info_dir / "sunlakes_infos_train.pkl", "wb"))
    pickle.dump([dataset[key] for key in val_keys], open(info_dir / "sunlakes_infos_val.pkl", "wb"))
    _write_scene_split(dataset, info_dir)
    print("official split", len(train_keys), len(val_keys), "dropped", len(dropped), "kept", len(dataset))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bag", default="rosbag2_2026_01_13-16_05_56")
    parser.add_argument("--rot-pc", type=float, default=-142.5)
    parser.add_argument("--full", action="store_true")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    if args.full:
        convert_full(args.rot_pc, args.workers)
        return
    annotation = LAYOUT / "one_seq" / "HSQJN_3.json"
    count, missing = _filter_annotation(args.bag, annotation)
    if missing:
        raise SystemExit("%s still has %d missing clouds" % (args.bag, missing))
    _link_cloud(args.bag)
    print("frames", count, "bag", args.bag)
    rotate_point_cloud(str(LAYOUT), args.rot_pc, "one_seq", "os128", workers=4)
    dataset = create_dataset_from_json(
        str(LAYOUT), "one_seq", args.rot_pc, "os128", None, True, False, False
    )
    train_keys, val_keys, dropped = split_dataset_divisible(
        dataset, 0.8, m=1, seed=42, drop_from="val"
    )
    info_dir = LAYOUT / "v1.0-trainval"
    info_dir.mkdir(parents=True, exist_ok=True)
    pickle.dump([dataset[key] for key in train_keys], open(info_dir / "sunlakes_infos_train.pkl", "wb"))
    pickle.dump([dataset[key] for key in val_keys], open(info_dir / "sunlakes_infos_val.pkl", "wb"))
    import numpy as np

    sample = dataset[train_keys[0]]
    points = np.fromfile(sample["lidar_path"], dtype=np.float32).reshape(-1, 5)
    raw = next(frame for frame in json.load(open(annotation)) if Path(frame["info"]).stem == sample["token"])
    raw_box = next(label for label in raw["labels"] if label["drawType"] == "box3d")
    theta = np.deg2rad(args.rot_pc)
    rotation = np.array(
        [[np.cos(theta), -np.sin(theta), 0], [np.sin(theta), np.cos(theta), 0], [0, 0, 1]]
    )
    expected = np.asarray(raw_box["points"][:3], dtype=np.float64) @ rotation.T
    converted = sample["gt_boxes"][:, :3]
    nearest = np.linalg.norm(converted - expected, axis=1).min()
    print(
        "infos", len(train_keys), len(val_keys), "dropped", len(dropped),
        "points", tuple(points.shape),
        "unique_timestamp", len(set(points[:, 4].tolist())),
        "boxes", tuple(sample["gt_boxes"].shape),
        "names", sorted(set(sample["gt_names"].tolist())),
        "rotation_error_m", float(nearest),
        "dt_velocity_nonzero", int(np.count_nonzero(np.linalg.norm(sample["gt_boxes"][:, 7:9], axis=1) > 1e-3)),
    )


if __name__ == "__main__":
    main()
