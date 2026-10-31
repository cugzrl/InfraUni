"""Compare camera-lifted BEV with lidar BEV on time-aligned RESOLVE frames.

Depth is predicted by intersecting each camera ray with the ground plane in the
lidar frame. Image edges are then splatted with that depth (one-bin LSS). The
lift does not use lidar returns on the objects, so a far-range shift is the
spatial gap between the image BEV and the lidar boxes.
"""
from __future__ import annotations

import csv
import os
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Polygon

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "resolve_multimodal"
OUT = ROOT / "visual" / "spatial_offsets_visual"
CAMERAS = ("axis1", "axis2", "axis3", "axis4")
VEHICLES = {"car", "truck", "bus", "trailer", "construction_vehicle"}
BINS = ((0.0, 30.0), (30.0, 60.0), (60.0, 100.0), (100.0, 250.0))
BIN_LABELS = ("0-30 m", "30-60 m", "60-100 m", ">=100 m")
# Macaron RGB matching the camera-projection palette.
CLASS_RGB = {
    "car": (0xFD, 0xB1, 0xDF),
    "truck": (0xF3, 0x99, 0x9B),
    "bus": (0xFB, 0xA8, 0x74),
    "trailer": (0xFF, 0xC7, 0x78),
    "construction_vehicle": (0xCF, 0xDB, 0x8E),
    "motorcycle": (0x86, 0xD7, 0xA7),
    "bicycle": (0xA1, 0xE2, 0xE7),
    "traffic_cone": (0x82, 0xC9, 0xE5),
    "pedestrian": (0xBB, 0xA4, 0xEB),
    "barrier": (0xB4, 0xB4, 0xB4),
}
X_MIN, X_MAX = -90.0, 90.0
Y_MIN, Y_MAX = -90.0, 90.0
RESOLUTION = 0.25


def parse_stamp(path):
    sec, frac = os.path.basename(path).rsplit(".", 1)[0].split("-")
    return int(sec) + int(frac) / 1e9


def residuals():
    """Per-frame camera-lidar delay after removing each sequence's fixed clock offset."""
    found = {}
    for split in ("train", "val", "test"):
        split_dir = DATA / split
        for seq in os.listdir(split_dir):
            if not seq.startswith("rosbag"):
                continue
            rows = []
            with (split_dir / seq / "paired_frames.csv").open(newline="", encoding="utf-8-sig") as handle:
                for row in csv.DictReader(handle):
                    lidar = parse_stamp(row["lidar"])
                    stamps = {cam: parse_stamp(row[cam]) for cam in CAMERAS}
                    rows.append((int(row["frame_index"]), lidar, stamps))
            med = {
                cam: float(np.median([stamps[cam] - lidar for _, lidar, stamps in rows]))
                for cam in CAMERAS
            }
            for frame_index, lidar, stamps in rows:
                found[(split, seq, frame_index)] = {
                    cam: stamps[cam] - lidar - med[cam] for cam in CAMERAS
                }
    return found


def fit_ground(points):
    """Robust ground plane n·x + d = 0, with n pointing upward."""
    pts = np.asarray(points[:, :3], dtype=np.float64)
    low = pts[pts[:, 2] < np.percentile(pts[:, 2], 30)]
    if len(low) > 25000:
        low = low[np.random.default_rng(0).choice(len(low), 25000, replace=False)]
    rng = np.random.default_rng(1)
    best = None
    for _ in range(60):
        chosen = low[rng.choice(len(low), 3, replace=False)]
        normal = np.cross(chosen[1] - chosen[0], chosen[2] - chosen[0])
        norm = np.linalg.norm(normal)
        if norm < 1e-6:
            continue
        normal = normal / norm
        if abs(normal[2]) < 0.85:
            continue
        offset = -float(np.dot(normal, chosen[0]))
        residual = np.abs(low @ normal + offset)
        inliers = residual < 0.15
        count = int(inliers.sum())
        if best is None or count > best[0]:
            best = (count, normal, offset)
    normal, offset = best[1], best[2]
    if normal[2] < 0:
        normal, offset = -normal, -offset
    return normal, offset


def rays_from_pixels(uv, k_mat, distortion):
    uv = np.asarray(uv, dtype=np.float64).reshape(-1, 1, 2)
    if distortion is not None and np.any(np.abs(distortion) > 1e-12):
        normed = cv2.undistortPoints(uv, k_mat, distortion)
        xy = normed.reshape(-1, 2)
    else:
        xy = np.column_stack([
            (uv[:, 0, 0] - k_mat[0, 2]) / k_mat[0, 0],
            (uv[:, 0, 1] - k_mat[1, 2]) / k_mat[1, 1],
        ])
    return np.column_stack([xy, np.ones(len(xy))])


def lift_rays(dirs_cam, lidar2cam, normal, offset):
    """Intersect camera rays with the ground plane. Returns lidar xyz and camera-frame depth."""
    rotation = lidar2cam[:3, :3]
    translation = lidar2cam[:3, 3]
    origin = -rotation.T @ translation
    direction = dirs_cam @ rotation
    denom = direction @ normal
    numer = -(origin @ normal + offset)
    valid = np.abs(denom) > 1e-6
    scale = np.zeros(len(dirs_cam))
    scale[valid] = numer / denom[valid]
    points = origin + scale[:, None] * direction
    camera = (points @ rotation.T) + translation
    keep = valid & (scale > 0) & (camera[:, 2] > 0.5)
    return points, camera[:, 2], keep


def project_lidar(points, lidar2cam, k_mat):
    rotation = lidar2cam[:3, :3]
    translation = lidar2cam[:3, 3]
    camera = points @ rotation.T + translation
    depth = camera[:, 2]
    uv = np.column_stack([
        k_mat[0, 0] * camera[:, 0] / np.maximum(depth, 1e-6) + k_mat[0, 2],
        k_mat[1, 1] * camera[:, 1] / np.maximum(depth, 1e-6) + k_mat[1, 2],
    ])
    return uv, depth


def box_samples(box, count=5):
    """Points on and inside one box. Last axis is local height, -0.5 at the bottom."""
    xs = np.linspace(-0.5, 0.5, count)
    ys = np.linspace(-0.5, 0.5, max(3, count // 2))
    zs = np.linspace(-0.5, 0.5, count)
    grid = np.stack(np.meshgrid(xs, ys, zs, indexing="ij"), -1).reshape(-1, 3)
    grid = grid * box[3:6]
    yaw = float(box[6])
    cos, sin = np.cos(yaw), np.sin(yaw)
    rotated = np.column_stack([
        cos * grid[:, 0] - sin * grid[:, 1],
        sin * grid[:, 0] + cos * grid[:, 1],
        grid[:, 2],
    ])
    return rotated + box[:3], grid[:, 2]


def corners_xy(boxes):
    boxes = np.asarray(boxes, dtype=np.float64)
    boxes = boxes.reshape(boxes.shape[0], -1)[:, :7]
    local = boxes[:, None, 3:5] * np.array([[-0.5, -0.5], [0.5, -0.5], [0.5, 0.5], [-0.5, 0.5]])
    cos, sin = np.cos(boxes[:, 6]), np.sin(boxes[:, 6])
    rot = np.stack([cos, -sin, sin, cos], 1).reshape(-1, 2, 2)
    return np.einsum("nij,nkj->nki", rot, local) + boxes[:, None, :2]


def bin_index(distance):
    for index, (low, high) in enumerate(BINS):
        if low <= distance < high:
            return index
    return None


def camera_response(info, normal, offset):
    """Sample each ground-plane cell from the camera that sees it most closely."""
    height = int(round((Y_MAX - Y_MIN) / RESOLUTION))
    width = int(round((X_MAX - X_MIN) / RESOLUTION))
    xs = np.linspace(X_MIN + RESOLUTION / 2, X_MAX - RESOLUTION / 2, width)
    ys = np.linspace(Y_MIN + RESOLUTION / 2, Y_MAX - RESOLUTION / 2, height)
    xx, yy = np.meshgrid(xs, ys)
    zz = -(normal[0] * xx + normal[1] * yy + offset) / normal[2]
    ground = np.stack([xx, yy, zz], -1).reshape(-1, 3)
    canvas = np.zeros((height * width, 3), dtype=np.float32)
    nearest = np.full(height * width, np.inf, dtype=np.float32)
    for cam_index, _name in enumerate(info["camera_names"]):
        image = cv2.imread(info["images"][cam_index], cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(info["images"][cam_index])
        uv, depth = project_lidar(ground, info["lidar2cam"][cam_index], info["K"][cam_index])
        image_h, image_w = image.shape[:2]
        valid = (
            (depth > 0.5)
            & (uv[:, 0] >= 0) & (uv[:, 0] < image_w - 1)
            & (uv[:, 1] >= 0) & (uv[:, 1] < image_h - 1)
            & (depth < nearest)
        )
        if not np.any(valid):
            continue
        cols = uv[valid, 0]
        rows = uv[valid, 1]
        c0 = np.floor(cols).astype(np.int32)
        r0 = np.floor(rows).astype(np.int32)
        wc = cols - c0
        wr = rows - r0
        color = (
            image[r0, c0] * ((1 - wr) * (1 - wc))[:, None]
            + image[r0, np.clip(c0 + 1, 0, image_w - 1)] * ((1 - wr) * wc)[:, None]
            + image[np.clip(r0 + 1, 0, image_h - 1), c0] * (wr * (1 - wc))[:, None]
            + image[np.clip(r0 + 1, 0, image_h - 1), np.clip(c0 + 1, 0, image_w - 1)] * (wr * wc)[:, None]
        )
        canvas[valid] = color[:, ::-1]
        nearest[valid] = depth[valid]
    return canvas.reshape(height, width, 3)


def draw_boxes(axis, boxes, names):
    polygons = corners_xy(boxes)
    for polygon, name in zip(polygons, names):
        color = tuple(channel / 255.0 for channel in CLASS_RGB.get(str(name), (180, 180, 180)))
        axis.add_patch(Polygon(polygon, closed=True, fill=False, edgecolor=color, linewidth=1.6, zorder=3))


def style_axis(axis):
    axis.set_xlim(X_MIN, X_MAX)
    axis.set_ylim(Y_MIN, Y_MAX)
    axis.set_aspect("equal", adjustable="box")
    axis.set_xlabel("x (m)")
    axis.set_ylabel("y (m)")
    axis.tick_params(labelsize=8)


def measure_frame(info, normal, offset):
    boxes = np.asarray(info["gt_boxes"], dtype=np.float64)
    names = [str(name) for name in info["gt_names"]]
    records = []
    for box, name in zip(boxes, names):
        if name not in VEHICLES:
            continue
        samples, local_z = box_samples(box)
        lifted = []
        bottom = []
        for cam_index in range(len(info["camera_names"])):
            uv, depth = project_lidar(samples, info["lidar2cam"][cam_index], info["K"][cam_index])
            image_h, image_w = 1080, 1920
            visible = (
                (depth > 1.0)
                & (uv[:, 0] >= 0) & (uv[:, 0] < image_w)
                & (uv[:, 1] >= 0) & (uv[:, 1] < image_h)
            )
            if not np.any(visible):
                continue
            dirs = rays_from_pixels(uv[visible], info["K"][cam_index], info["distortion"][cam_index])
            points, _depth, keep = lift_rays(dirs, info["lidar2cam"][cam_index], normal, offset)
            if not np.any(keep):
                continue
            xy = points[keep, :2]
            z_local = local_z[visible][keep]
            lifted.append(xy)
            bottom.append(xy[z_local < -0.25 * box[5]])
        if not lifted:
            continue
        stacked = np.concatenate(lifted, 0)
        center = box[:2]
        error = np.median(stacked, 0) - center
        radial = center / max(np.linalg.norm(center), 1e-6)
        lateral = np.array([-radial[1], radial[0]])
        record = {
            "distance": float(np.linalg.norm(center)),
            "radial": float(np.dot(error, radial)),
            "lateral": float(np.dot(error, lateral)),
            "name": name,
        }
        if any(len(part) for part in bottom):
            contact = np.concatenate([part for part in bottom if len(part)], 0)
            contact_error = np.median(contact, 0) - center
            record["contact_radial"] = float(np.dot(contact_error, radial))
        records.append(record)
    return records


def roundtrip_error(points, info, normal, offset):
    """Project ground points and lift the pixels back. Returns median XY error in metres."""
    residual = points[:, :3] @ normal + offset
    ground = points[np.abs(residual) < 0.1][:400, :3]
    if len(ground) < 20:
        return None
    errors = []
    for cam_index in range(len(info["camera_names"])):
        uv, depth = project_lidar(ground, info["lidar2cam"][cam_index], info["K"][cam_index])
        visible = (depth > 1.0) & (uv[:, 0] >= 0) & (uv[:, 0] < 1920) & (uv[:, 1] >= 0) & (uv[:, 1] < 1080)
        if int(visible.sum()) < 10:
            continue
        dirs = rays_from_pixels(uv[visible], info["K"][cam_index], info["distortion"][cam_index])
        lifted, _depth, keep = lift_rays(dirs, info["lidar2cam"][cam_index], normal, offset)
        if not np.any(keep):
            continue
        errors.append(np.linalg.norm(lifted[keep, :2] - ground[visible][keep, :2], axis=1))
    if not errors:
        return None
    return float(np.median(np.concatenate(errors)))


def view_limits(boxes):
    xy = np.asarray(boxes, dtype=np.float64).reshape(len(boxes), -1)[:, :2]
    pad = 20.0
    xmin = float(np.clip(xy[:, 0].min() - pad, X_MIN, X_MAX - 40))
    xmax = float(np.clip(xy[:, 0].max() + pad, xmin + 40, X_MAX))
    ymin = float(np.clip(xy[:, 1].min() - pad, Y_MIN, Y_MAX - 40))
    ymax = float(np.clip(xy[:, 1].max() + pad, ymin + 40, Y_MAX))
    return xmin, xmax, ymin, ymax


def save_pair(path, points, response, info, title):
    boxes = np.asarray(info["gt_boxes"])
    names = [str(name) for name in info["gt_names"]]
    xmin, xmax, ymin, ymax = view_limits(boxes)
    keep = (
        (points[:, 0] >= xmin) & (points[:, 0] <= xmax)
        & (points[:, 1] >= ymin) & (points[:, 1] <= ymax)
    )
    cloud = points[keep]
    if len(cloud) > 80000:
        cloud = cloud[:: len(cloud) // 80000]
    extent = [X_MIN, X_MAX, Y_MIN, Y_MAX]
    fig, axes = plt.subplots(1, 2, figsize=(12.2, 6.2), facecolor="white")
    axes[0].set_facecolor("white")
    low, high = np.percentile(cloud[:, 2], [5, 98])
    axes[0].scatter(cloud[:, 0], cloud[:, 1], c=cloud[:, 2], cmap="viridis", vmin=low, vmax=high,
                    s=1.2, linewidths=0, rasterized=True)
    draw_boxes(axes[0], boxes, names)
    style_axis(axes[0])
    axes[0].set_xlim(xmin, xmax)
    axes[0].set_ylim(ymin, ymax)
    axes[0].set_title("LiDAR BEV")
    axes[1].imshow(np.clip(response / 255.0, 0, 1), origin="lower", extent=extent, interpolation="nearest")
    draw_boxes(axes[1], boxes, names)
    style_axis(axes[1])
    axes[1].set_xlim(xmin, xmax)
    axes[1].set_ylim(ymin, ymax)
    axes[1].set_title("Camera BEV")
    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=140, facecolor="white")
    plt.close(fig)


def save_summary(path, radial, contact):
    fig, axis = plt.subplots(figsize=(7.2, 4.4), facecolor="white")
    positions = np.arange(len(BINS))
    body, body_n, contact_m, contact_n = [], [], [], []
    for index in range(len(BINS)):
        values = radial[index]
        contacts = contact[index]
        body.append(float(np.median(values)) if values else np.nan)
        body_n.append(len(values))
        contact_m.append(float(np.median(contacts)) if contacts else np.nan)
        contact_n.append(len(contacts))
    axis.plot(positions, body, "o-", color="#C45C76", label="whole vehicle")
    axis.plot(positions, contact_m, "s--", color="#3C7A6A", label="ground contact")
    axis.axhline(0, color="#888888", linewidth=0.8)
    axis.set_xticks(positions)
    axis.set_xticklabels([f"{label}\n(n={count})" for label, count in zip(BIN_LABELS, body_n)])
    axis.set_ylabel("median range error (m)")
    axis.set_title("Camera BEV minus LiDAR box center")
    axis.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=140, facecolor="white")
    plt.close(fig)
    return body, body_n, contact_m, contact_n


def main():
    from infrauni.resolve.multimodal import load_infos

    OUT.mkdir(parents=True, exist_ok=True)
    delay = residuals()
    chosen = []
    for split in ("train", "val"):
        for info in load_infos(split):
            key = (split, info["scene_token"], int(info["frame_index"]))
            gap = delay.get(key)
            if gap is None or any(abs(gap[cam]) >= 0.02 for cam in CAMERAS):
                continue
            boxes = np.asarray(info["gt_boxes"])
            names = [str(name) for name in info["gt_names"]]
            distances = [
                float(np.hypot(box[0], box[1]))
                for box, name in zip(boxes, names)
                if name in VEHICLES
            ]
            if not distances:
                continue
            present = {bin_index(distance) for distance in distances}
            present.discard(None)
            chosen.append((info, present, distances, gap))
    print(f"time-aligned vehicle frames: {len(chosen)}", flush=True)

    far_first = [index for index, item in enumerate(chosen) if 3 in item[1]]
    others = [index for index in range(len(chosen)) if index not in set(far_first)]
    rng = np.random.default_rng(0)
    order = far_first + rng.permutation(others).tolist()
    quota = {0: 160, 1: 160, 2: 160, 3: 10**9}
    taken = {index: 0 for index in range(4)}
    radial = {index: [] for index in range(len(BINS))}
    contact = {index: [] for index in range(len(BINS))}
    lateral = {index: [] for index in range(len(BINS))}
    roundtrips = []
    visuals = []
    for step, index in enumerate(order):
        info, present, _distances, _gap = chosen[index]
        if 3 not in present and all(taken[bin_id] >= quota[bin_id] for bin_id in present):
            continue
        points = np.fromfile(info["lidar_path"], dtype=np.float32).reshape(-1, 4)
        normal, offset = fit_ground(points)
        trip = roundtrip_error(points, info, normal, offset)
        if trip is None or trip > 0.5:
            continue
        roundtrips.append(trip)
        records = measure_frame(info, normal, offset)
        for record in records:
            bin_id = bin_index(record["distance"])
            if bin_id is None:
                continue
            radial[bin_id].append(record["radial"])
            lateral[bin_id].append(abs(record["lateral"]))
            if "contact_radial" in record:
                contact[bin_id].append(record["contact_radial"])
        for bin_id in present:
            taken[bin_id] += 1
        far = [record["radial"] for record in records if record["distance"] >= 60]
        near = [record["radial"] for record in records if record["distance"] < 30]
        if far and near:
            visuals.append((float(np.median(far)), trip, info, present))
        if step >= len(far_first) and all(taken[bin_id] >= quota[bin_id] for bin_id in (0, 1, 2)):
            break
    print("measured frames", len(roundtrips), "roundtrip median m", float(np.median(roundtrips)), flush=True)

    visuals.sort(key=lambda item: item[0], reverse=True)
    used = set()
    saved = []
    for score, trip, info, present in visuals:
        if info["scene_token"] in used:
            continue
        points = np.fromfile(info["lidar_path"], dtype=np.float32).reshape(-1, 4)
        normal, offset = fit_ground(points)
        response = camera_response(info, normal, offset)
        stem = f"{len(saved)+1:02d}_{info['scene_token']}_f{int(info['frame_index']):06d}"
        title = f"{info['scene_token']}  frame {int(info['frame_index'])}"
        save_pair(OUT / f"{stem}.jpg", points, response, info, title)
        saved.append((stem, score, trip, sorted(present)))
        used.add(info["scene_token"])
        print("saved", stem, "far-median-shift", round(score, 2), "roundtrip", round(trip, 3), flush=True)
        if len(saved) >= 4:
            break
    body, body_n, contact_m, contact_n = save_summary(OUT / "summary_range_error.jpg", radial, contact)
    print("RANGE_MEDIAN", list(zip(BIN_LABELS, body, body_n)))
    print("CONTACT_MEDIAN", list(zip(BIN_LABELS, contact_m, contact_n)))
    for index, label in enumerate(BIN_LABELS):
        values = lateral[index]
        print(label, "lateral median", float(np.median(values)) if values else None, "n", len(values))


if __name__ == "__main__":
    main()
