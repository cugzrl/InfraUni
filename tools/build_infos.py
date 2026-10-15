"""Build nuScenes-mini infos with 10 sweeps and 6 s future trajectories.

Uses BEVFusion's converter so LiDAR boxes, camera calibration and sweeps match
the TransFusion / mmdet3d convention. Futures are in the current LiDAR frame.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import mmcv
import numpy as np
from nuscenes.nuscenes import NuScenes
from nuscenes.utils import splits
from pyquaternion import Quaternion

ROOT = Path(__file__).resolve().parents[1]
BEVFUSION = Path("/home/kemove/devdata1/zrl/bevfusion")
sys.path.insert(0, str(BEVFUSION))

NUM_FUTURE = 12


def _load_converter():
    path = BEVFUSION / "tools/data_converter/nuscenes_converter.py"
    spec = importlib.util.spec_from_file_location("nusc_converter", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _global_to_lidar(xyz, info):
    e2g_r = Quaternion(info["ego2global_rotation"]).rotation_matrix
    e2g_t = np.asarray(info["ego2global_translation"], dtype=np.float64)
    l2e_r = Quaternion(info["lidar2ego_rotation"]).rotation_matrix
    l2e_t = np.asarray(info["lidar2ego_translation"], dtype=np.float64)
    xyz = np.asarray(xyz, dtype=np.float64) - e2g_t
    xyz = xyz @ np.linalg.inv(e2g_r).T
    xyz = xyz - l2e_t
    xyz = xyz @ np.linalg.inv(l2e_r).T
    return xyz.astype(np.float32)


def _add_futures(nusc: NuScenes, infos):
    by_token = {sample["token"]: sample for sample in nusc.sample}
    for info in infos:
        sample = by_token[info["token"]]
        anns = [nusc.get("sample_annotation", token) for token in sample["anns"]]
        if len(anns) != len(info["gt_boxes"]):
            raise RuntimeError(
                f"annotation count {len(anns)} != boxes {len(info['gt_boxes'])} "
                f"for {info['token']}"
            )
        futures = np.zeros((len(anns), NUM_FUTURE, 2), dtype=np.float32)
        masks = np.zeros((len(anns), NUM_FUTURE), dtype=np.float32)
        tokens = []
        for i, ann in enumerate(anns):
            tokens.append(ann["instance_token"])
            cursor = ann
            for step in range(NUM_FUTURE):
                nxt = cursor["next"]
                if not nxt:
                    break
                cursor = nusc.get("sample_annotation", nxt)
                center = _global_to_lidar(cursor["translation"], info)
                futures[i, step] = center[:2]
                masks[i, step] = 1.0
        info["gt_instance_tokens"] = tokens
        info["gt_fut"] = futures
        info["gt_fut_mask"] = masks


def main():
    dataroot = "/home/kemove/devdata1/zrl/bevfusion/data/nuscenes"
    converter = _load_converter()
    nusc = NuScenes(version="v1.0-mini", dataroot=dataroot, verbose=True)
    available = converter.get_available_scenes(nusc)
    names = [scene["name"] for scene in available]
    train_scenes = [name for name in splits.mini_train if name in names]
    val_scenes = [name for name in splits.mini_val if name in names]
    train_tokens = {available[names.index(name)]["token"] for name in train_scenes}
    val_tokens = {available[names.index(name)]["token"] for name in val_scenes}
    train_infos, val_infos = converter._fill_trainval_infos(
        nusc, train_tokens, val_tokens, test=False, max_sweeps=10
    )
    _add_futures(nusc, train_infos)
    _add_futures(nusc, val_infos)
    out_dir = ROOT / "data" / "infos"
    out_dir.mkdir(parents=True, exist_ok=True)
    meta = {"version": "v1.0-mini"}
    mmcv.dump({"infos": train_infos, "metadata": meta}, out_dir / "nuscenes_infos_train.pkl")
    mmcv.dump({"infos": val_infos, "metadata": meta}, out_dir / "nuscenes_infos_val.pkl")
    sweeps = [len(info["sweeps"]) for info in train_infos]
    fut = np.concatenate([info["gt_fut_mask"] for info in train_infos], axis=0)
    print(
        f"train {len(train_infos)} val {len(val_infos)} "
        f"sweeps mean {np.mean(sweeps):.2f} future steps {fut.sum(1).mean():.2f}"
    )


if __name__ == "__main__":
    main()
