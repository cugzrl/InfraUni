"""Side elevation of one RESOLVE road, Open3D, point cloud only.

The camera looks across the road the vehicles are driving on, the way a
roadside elevation sketch does. Height on screen is stretched so a two-degree
grade is not squeezed into a flat line. Color uses the real height and the
whole colormap, so a small height change is a large color change. Earlier
figures in visual/road_height are not overwritten.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d

from infrauni.motivation.road_height import analyze, middle_frames

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "visual" / "road_height"
WIDTH = 1600
HEIGHT = 900
HALF_ALONG = 40.0
STRETCH = 4.0

SEQUENCES = (
    "rosbag2_2026_01_07-15_01_29",
    "rosbag2_2025_11_11-16_22_03",
    "rosbag2_2026_02_01-12_19_50",
)


def grade_direction(coeff):
    direction = np.array([coeff[0], coeff[1]], dtype=np.float64)
    direction /= np.linalg.norm(direction)
    return direction


def road_direction(boxes, grade):
    """Heading of the road that climbs, taken from the vehicle yaws."""
    yaw = np.mod(np.asarray(boxes, dtype=np.float64)[:, 6], np.pi)
    edges = np.linspace(0.0, np.pi, 13)
    hist, _ = np.histogram(yaw, edges)
    best = None
    for index, count in enumerate(hist):
        if count < 1:
            continue
        chosen = yaw[(yaw >= edges[index]) & (yaw < edges[index + 1])]
        angle = float(np.angle(np.mean(np.exp(1j * 2.0 * chosen))) / 2.0)
        road = np.array([np.cos(angle), np.sin(angle)])
        score = float(count) * (0.25 + abs(float(road @ grade)))
        if best is None or score > best[0]:
            best = (score, road)
    return best[1]


def corridor(xyz, coeff, boxes):
    grade = grade_direction(coeff)
    road = road_direction(boxes, grade)
    side = np.array([-road[1], road[0]])
    along = xyz[:, :2] @ road
    across = xyz[:, :2] @ side
    plane = coeff[0] * xyz[:, 0] + coeff[1] * xyz[:, 1] + coeff[2]
    ground = (np.abs(xyz[:, 2] - plane) < 0.45) & (np.abs(along) < HALF_ALONG + 5.0)
    bins = np.arange(-40.0, 42.0, 2.0)
    hist, edges = np.histogram(across[ground], bins)
    lane = float(edges[int(np.argmax(hist))] + 1.0)
    keep = (
        (np.abs(across - lane) < 8.0)
        & (np.abs(along) <= HALF_ALONG)
        & (xyz[:, 2] > plane - 0.6)
        & (xyz[:, 2] < plane + 2.4)
    )
    return xyz[keep], road, side, lane, float(np.array(coeff[:2]) @ road)


def height_colors(points, coeff):
    """Full colormap across the road surface, so vehicle roofs do not swallow the scale."""
    z = points[:, 2]
    plane = coeff[0] * points[:, 0] + coeff[1] * points[:, 1] + coeff[2]
    ground = np.abs(z - plane) < 0.4
    sample = z[ground] if int(ground.sum()) > 50 else z
    lo, hi = np.percentile(sample, [3, 97])
    span = max(float(hi - lo), 0.25)
    tone = np.clip((z - lo) / span, 0.0, 1.0)
    return plt.cm.turbo(tone)[:, :3].astype(np.float64), float(lo), float(hi)


def render(path, xyz, coeff, side, lane, z_mid):
    colors, lo, hi = height_colors(xyz, coeff)
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(xyz.astype(np.float64))
    cloud.colors = o3d.utility.Vector3dVector(colors)
    renderer = o3d.visualization.rendering.OffscreenRenderer(WIDTH, HEIGHT)
    renderer.scene.set_background([0.0, 0.0, 0.0, 1.0])
    material = o3d.visualization.rendering.MaterialRecord()
    material.shader = "defaultUnlit"
    material.point_size = 6.0
    renderer.scene.add_geometry("cloud", cloud, material)

    look = lane * side
    look = np.array([look[0], look[1], z_mid], dtype=np.float32)
    eye_xy = look[:2] + np.asarray(side, dtype=np.float32) * 50.0
    eye = np.array([eye_xy[0], eye_xy[1], z_mid], dtype=np.float32)
    renderer.setup_camera(20.0, look, eye, np.array([0.0, 0.0, 1.0], dtype=np.float32))
    half_z = 0.5625 * (2.0 * HALF_ALONG) / STRETCH / 2.0
    renderer.scene.camera.set_projection(
        o3d.visualization.rendering.Camera.Projection.Ortho,
        -HALF_ALONG, HALF_ALONG, -half_z, half_z, 5.0, 120.0,
    )
    image = renderer.render_to_image()
    o3d.io.write_image(str(path), image)
    return lo, hi


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    wanted = {token: None for token in SEQUENCES}
    for info in middle_frames():
        if info["scene_token"] in wanted:
            wanted[info["scene_token"]] = info
    for token, info in wanted.items():
        if info is None:
            print("missing", token)
            continue
        record = analyze(info)
        xyz = np.fromfile(info["lidar_path"], dtype=np.float32).reshape(-1, 4)[:, :3]
        finite = np.isfinite(xyz).all(1)
        xyz = xyz[finite]
        boxes = np.asarray(info["gt_boxes"], dtype=np.float64)
        kept, _road, side, lane, along_slope = corridor(xyz, record["coeff"], boxes)
        z_mid = float(np.median(kept[:, 2]))
        path = OUT / f"o3d_profile_{token}_f{int(info['frame_index']):06d}.png"
        lo, hi = render(path, kept, record["coeff"], side, lane, z_mid)
        print(
            f"saved {path.name} along-road slope {np.degrees(np.arctan(abs(along_slope))):.2f} deg "
            f"color {lo:.2f}..{hi:.2f} m points {len(kept)} stretch x{STRETCH:.0f}",
            flush=True,
        )


if __name__ == "__main__":
    main()
