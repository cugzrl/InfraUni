"""SE(3) helpers for nuScenes ego / sensor frames."""

from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation


def transform_matrix(translation, rotation_wxyz) -> np.ndarray:
    """Build a 4x4 matrix. Quaternion is nuScenes (w, x, y, z)."""
    mat = np.eye(4, dtype=np.float64)
    quat_xyzw = [
        rotation_wxyz[1],
        rotation_wxyz[2],
        rotation_wxyz[3],
        rotation_wxyz[0],
    ]
    mat[:3, :3] = Rotation.from_quat(quat_xyzw).as_matrix()
    mat[:3, 3] = np.asarray(translation, dtype=np.float64)
    return mat


def yaw_from_rotmat(rot: np.ndarray) -> float:
    return float(np.arctan2(rot[1, 0], rot[0, 0]))


def transform_points(points: np.ndarray, mat: np.ndarray) -> np.ndarray:
    """Apply a 4x4 matrix to (N, 3) points."""
    rot = mat[:3, :3]
    trans = mat[:3, 3]
    return points @ rot.T + trans


def global_box_to_lidar(translation, size_wlh, rotation_wxyz, global_from_lidar):
    """Map one global box into the current LiDAR frame.

    Returns x, y, z, w, l, h, yaw.
    """
    lidar_from_global = np.linalg.inv(global_from_lidar)
    center = transform_points(
        np.asarray(translation, dtype=np.float64)[None, :], lidar_from_global
    )[0]
    quat_xyzw = [rotation_wxyz[1], rotation_wxyz[2], rotation_wxyz[3], rotation_wxyz[0]]
    rot_box = Rotation.from_quat(quat_xyzw).as_matrix()
    rot_lidar = lidar_from_global[:3, :3] @ rot_box
    yaw = yaw_from_rotmat(rot_lidar)
    width, length, height = size_wlh
    return np.array(
        [center[0], center[1], center[2], width, length, height, yaw],
        dtype=np.float32,
    )
