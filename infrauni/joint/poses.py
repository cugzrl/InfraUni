"""Lidar poses for aligning a past prediction into the current frame.

nuScenes ego motion is applied even on the mini split. A roadside sensor that
does not move is the same transform with an identity ego pose. Memory resets
when ``scene_token`` changes.
"""

from __future__ import annotations

import json
import os
import pickle

import numpy as np
from pyquaternion import Quaternion


def _pose(rotation, translation):
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] = Quaternion(rotation).rotation_matrix
    matrix[:3, 3] = np.asarray(translation, dtype=np.float64)
    return matrix


def load_pose_index(info_paths, sample_table=""):
    """Map a sample token to scene id, timestamp and lidar-to-global."""

    scenes = {}
    if sample_table and os.path.exists(sample_table):
        for row in json.load(open(sample_table, "r")):
            scenes[row["token"]] = row["scene_token"]
    index = {}
    for path in info_paths:
        if not os.path.exists(path):
            continue
        infos = pickle.load(open(path, "rb"))["infos"]
        for info in infos:
            lidar_to_ego = _pose(info["lidar2ego_rotation"], info["lidar2ego_translation"])
            ego_to_global = _pose(info["ego2global_rotation"], info["ego2global_translation"])
            token = info["token"]
            index[token] = {
                "scene": info.get("scene_token") or scenes.get(token, token),
                "timestamp": float(info["timestamp"]) * 1e-6,
                "lidar_to_global": ego_to_global @ lidar_to_ego,
            }
    return index


def lidar_to_lidar(source, target):
    """Rigid map from a past lidar frame into the current lidar frame."""

    return np.linalg.inv(target) @ source


def apply_xy(transform, xy):
    """Apply a 4x4 transform to ``[N, 2]`` points on the ground plane."""

    ones = np.ones((xy.shape[0], 1), dtype=np.float64)
    zeros = np.zeros((xy.shape[0], 1), dtype=np.float64)
    homo = np.concatenate([xy.astype(np.float64), zeros, ones], axis=1)
    mapped = homo @ transform.T
    return mapped[:, :2]
