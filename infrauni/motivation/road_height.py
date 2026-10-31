"""Road height from RESOLVE lidar: slope versus leftover undulation.

Each sequence contributes its middle labeled frame. Road height in a cell is a
low percentile of z, so vehicles do not set the surface. A plane fitted to
those cells is the 'flat road' model. Residuals are the bumps and sags that a
single plane cannot absorb.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "visual" / "road_height"
CELL = 2.0
RADIUS = 100.0
RINGS = tuple(range(0, 100, 10))


def middle_frames():
    from infrauni.resolve.multimodal import load_infos

    grouped = {}
    for split in ("train", "val", "test"):
        for info in load_infos(split):
            grouped.setdefault(info["scene_token"], []).append(info)
    picked = []
    for token, frames in grouped.items():
        frames.sort(key=lambda item: int(item["frame_index"]))
        picked.append(frames[len(frames) // 2])
    return picked


def road_cells(points):
    xy = points[:, :2]
    z = points[:, 2]
    keep = np.isfinite(z) & (np.linalg.norm(xy, axis=1) < RADIUS)
    xy, z = xy[keep], z[keep]
    ix = np.floor((xy[:, 0] + RADIUS) / CELL).astype(np.int32)
    iy = np.floor((xy[:, 1] + RADIUS) / CELL).astype(np.int32)
    span = int(np.ceil(2 * RADIUS / CELL))
    key = ix * span + iy
    order = np.argsort(key, kind="mergesort")
    key, z, xy = key[order], z[order], xy[order]
    cuts = np.flatnonzero(np.diff(key)) + 1
    starts = np.r_[0, cuts]
    ends = np.r_[cuts, len(key)]
    centers, heights, counts = [], [], []
    for start, end in zip(starts, ends):
        count = end - start
        if count < 25:
            continue
        sample = z[start:end]
        height = float(np.percentile(sample, 8))
        centers.append(xy[start:end].mean(0))
        heights.append(height)
        counts.append(count)
    if len(heights) < 30:
        return None
    return np.asarray(centers), np.asarray(heights), np.asarray(counts)


def fit_plane(centers, heights):
    """Iteratively fit z = a x + b y + c, dropping gross outliers."""
    keep = np.ones(len(heights), dtype=bool)
    coeff = None
    for _ in range(4):
        design = np.column_stack([centers[keep, 0], centers[keep, 1], np.ones(keep.sum())])
        coeff, _, _, _ = np.linalg.lstsq(design, heights[keep], rcond=None)
        residual = heights - (coeff[0] * centers[:, 0] + coeff[1] * centers[:, 1] + coeff[2])
        keep = np.abs(residual) < 0.6
        if keep.sum() < 30:
            break
    residual = heights - (coeff[0] * centers[:, 0] + coeff[1] * centers[:, 1] + coeff[2])
    return coeff, residual


def ring_profile(centers, heights, coeff):
    radius = np.linalg.norm(centers, axis=1)
    near = radius < 30.0
    if near.sum() < 15:
        near = np.ones(len(radius), dtype=bool)
    local, _ = fit_plane(centers[near], heights[near])
    rows = []
    for low in RINGS:
        high = low + 10.0
        mask = (radius >= low) & (radius < high)
        if mask.sum() < 8:
            rows.append(None)
            continue
        full = coeff[0] * centers[mask, 0] + coeff[1] * centers[mask, 1] + coeff[2]
        local_z = local[0] * centers[mask, 0] + local[1] * centers[mask, 1] + local[2]
        rows.append({
            "range": low + 5.0,
            "height": float(np.median(heights[mask])),
            "plane_gap": float(np.median(heights[mask] - full)),
            "near_plane_gap": float(np.median(heights[mask] - local_z)),
        })
    return rows


def analyze(info):
    points = np.fromfile(info["lidar_path"], dtype=np.float32).reshape(-1, 4)
    parsed = road_cells(points)
    if parsed is None:
        return None
    centers, heights, _counts = parsed
    coeff, residual = fit_plane(centers, heights)
    slope = float(np.hypot(coeff[0], coeff[1]))
    keep = np.abs(residual) < 0.6
    used = residual[keep]
    return {
        "sequence": info["scene_token"],
        "frame": int(info["frame_index"]),
        "split": info.get("dataset_split") or "",
        "centers": centers[keep],
        "heights": heights[keep],
        "residual": used,
        "coeff": coeff,
        "slope_deg": float(np.degrees(np.arctan(slope))),
        "grade_pct": slope * 100.0,
        "std": float(np.std(used)),
        "p95": float(np.percentile(np.abs(used), 95)),
        "span": float(np.percentile(used, 98) - np.percentile(used, 2)),
        "profile": ring_profile(centers[keep], heights[keep], coeff),
        "points": points,
    }


STRETCH = 15.0


def slope_axes(centers, coeff):
    direction = np.array([coeff[0], coeff[1]], dtype=np.float64)
    direction /= np.linalg.norm(direction)
    side = np.array([-direction[1], direction[0]])
    along = centers @ direction
    across = centers @ side
    return along, across


def draw_oblique(axis, record, limit):
    """Oblique view along the road grade. Height is stretched so the ramp and the bumps both show."""
    centers, heights = record["centers"], record["heights"]
    along, across = slope_axes(centers, record["coeff"])
    order = np.argsort(along)
    horizontal = across[order] + 0.55 * along[order]
    vertical = heights[order] * STRETCH + 0.15 * along[order]
    color = record["residual"][order]
    axis.scatter(horizontal, vertical, c=color, cmap="coolwarm", vmin=-limit, vmax=limit,
                 s=9, linewidths=0, rasterized=True)
    axis.set_xticks([])
    axis.set_yticks([])
    for spine in axis.spines.values():
        spine.set_visible(False)
    axis.set_title(f"oblique road, height x{STRETCH:.0f}")


def draw_heatmap(axis, record, limit):
    centers, residual = record["centers"], record["residual"]
    ix = np.floor((centers[:, 0] + RADIUS) / CELL).astype(np.int32)
    iy = np.floor((centers[:, 1] + RADIUS) / CELL).astype(np.int32)
    span = int(np.ceil(2 * RADIUS / CELL))
    grid = np.full((span, span), np.nan)
    grid[iy, ix] = residual
    image = axis.imshow(
        grid, cmap="coolwarm", vmin=-limit, vmax=limit, origin="lower",
        extent=[-RADIUS, RADIUS, -RADIUS, RADIUS],
    )
    axis.set_xlim(-80, 80)
    axis.set_ylim(-80, 80)
    axis.set_xlabel("x (m)")
    axis.set_ylabel("y (m)")
    axis.set_title("height left after one plane")
    axis.tick_params(labelsize=8)
    return image


def draw_profile(axis, record):
    centers, heights = record["centers"], record["heights"]
    along, _across = slope_axes(centers, record["coeff"])
    coeff = record["coeff"]
    plane = coeff[0] * centers[:, 0] + coeff[1] * centers[:, 1] + coeff[2]
    edges = np.arange(-80.0, 85.0, 5.0)
    mid, road, fitted = [], [], []
    for low, high in zip(edges[:-1], edges[1:]):
        mask = (along >= low) & (along < high)
        if mask.sum() < 6:
            continue
        mid.append(0.5 * (low + high))
        road.append(float(np.median(heights[mask])))
        fitted.append(float(np.median(plane[mask])))
    axis.plot(mid, road, "o-", color="#C45C76", label="road")
    axis.plot(mid, fitted, "--", color="#888888", label="one plane")
    axis.set_xlabel("distance along the slope (m)")
    axis.set_ylabel("height (m)")
    axis.set_title("side profile")
    axis.legend(fontsize=8, frameon=False)
    axis.tick_params(labelsize=8)


def save_example(path, record, limit):
    fig, axes = plt.subplots(1, 3, figsize=(14.6, 4.8), facecolor="white",
                             gridspec_kw={"width_ratios": [1.25, 1.0, 1.05]})
    draw_oblique(axes[0], record, limit)
    image = draw_heatmap(axes[1], record, limit)
    draw_profile(axes[2], record)
    fig.colorbar(image, ax=axes[1], fraction=0.046, pad=0.04).set_label("m")
    fig.suptitle(
        f"{record['sequence']}  f{record['frame']}   "
        f"slope {record['slope_deg']:.2f} deg   "
        f"leftover std {record['std']*100:.0f} cm",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(path, dpi=140, facecolor="white")
    plt.close(fig)


def save_mean_map(path, records, limit):
    span = int(np.ceil(2 * RADIUS / CELL))
    total = np.zeros((span, span), dtype=np.float64)
    count = np.zeros((span, span), dtype=np.int32)
    for record in records:
        ix = np.floor((record["centers"][:, 0] + RADIUS) / CELL).astype(np.int32)
        iy = np.floor((record["centers"][:, 1] + RADIUS) / CELL).astype(np.int32)
        np.add.at(total, (iy, ix), record["residual"])
        np.add.at(count, (iy, ix), 1)
    mean = np.divide(total, count, out=np.full_like(total, np.nan), where=count >= 20)
    fig, axis = plt.subplots(figsize=(6.2, 5.6), facecolor="white")
    image = axis.imshow(
        mean, cmap="coolwarm", vmin=-limit, vmax=limit, origin="lower",
        extent=[-RADIUS, RADIUS, -RADIUS, RADIUS],
    )
    axis.set_xlim(-80, 80)
    axis.set_ylim(-80, 80)
    axis.set_xlabel("x (m)")
    axis.set_ylabel("y (m)")
    axis.set_title("mean leftover height, all sequences")
    fig.colorbar(image, ax=axis, fraction=0.046, pad=0.04).set_label("m")
    fig.tight_layout()
    fig.savefig(path, dpi=140, facecolor="white")
    plt.close(fig)


def save_summary(path, records):
    std = np.array([item["std"] for item in records]) * 100.0
    p95 = np.array([item["p95"] for item in records]) * 100.0
    slope = np.array([item["slope_deg"] for item in records])
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.2), facecolor="white")
    axes[0].hist(std, bins=20, color="#8FB8C9", edgecolor="white")
    axes[0].set_xlabel("road residual std after one plane (cm)")
    axes[0].set_ylabel("sequences")
    axes[0].set_title("deviation from a flat plane")
    axes[1].hist(slope, bins=20, color="#E7B89A", edgecolor="white")
    axes[1].set_xlabel("best-fit road slope (deg)")
    axes[1].set_ylabel("sequences")
    axes[1].set_title("tilt of that plane")
    fig.tight_layout()
    fig.savefig(path, dpi=140, facecolor="white")
    plt.close(fig)
    return std, p95, slope


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    records = []
    for info in middle_frames():
        record = analyze(info)
        if record is not None:
            records.append(record)
            print(
                f"{record['sequence']} f{record['frame']} "
                f"slope {record['slope_deg']:.2f} deg std {record['std']*100:.1f} cm "
                f"p95 {record['p95']*100:.1f} cm",
                flush=True,
            )
    std, p95, slope = save_summary(OUT / "summary_road_flatness.jpg", records)
    limit = 0.45
    save_mean_map(OUT / "mean_leftover_height.jpg", records, limit)
    ranked = sorted(records, key=lambda item: item["std"])
    by_slope = sorted(records, key=lambda item: item["slope_deg"])
    chosen = [
        ("01_flattest", ranked[0]),
        ("02_typical", ranked[len(ranked) // 2]),
        ("03_steep", by_slope[-1]),
        ("04_most_uneven", ranked[-1]),
    ]
    seen = set()
    for stem, record in chosen:
        if record["sequence"] in seen and stem != "03_steep":
            continue
        seen.add(record["sequence"])
        save_example(OUT / f"{stem}_{record['sequence']}_f{record['frame']:06d}.jpg", record, limit)
        print("saved", stem, record["sequence"], flush=True)
    gaps = []
    for record in records:
        far = [row["near_plane_gap"] for row in record["profile"] if row and row["range"] >= 70]
        if far:
            gaps.append(float(np.median(far)))
    print("SEQUENCES", len(records))
    print("STD_CM median", float(np.median(std)), "p90", float(np.percentile(std, 90)), "max", float(std.max()))
    print("P95_CM median", float(np.median(p95)), "p90", float(np.percentile(p95, 90)), "max", float(p95.max()))
    print("SLOPE_DEG median", float(np.median(slope)), "p90", float(np.percentile(slope, 90)), "max", float(slope.max()))
    print("STD>10cm", int((std > 10).sum()), "STD>20cm", int((std > 20).sum()), "STD>30cm", int((std > 30).sum()))
    print("SLOPE>1deg", int((slope > 1).sum()), "SLOPE>2deg", int((slope > 2).sum()))
    if gaps:
        gaps = np.asarray(gaps)
        print("FAR70_NEAR_PLANE_GAP_M median", float(np.median(gaps)), "p10", float(np.percentile(gaps, 10)), "p90", float(np.percentile(gaps, 90)))
    clearance = np.array([-item["coeff"][2] for item in records])
    grade = np.tan(np.radians(slope))
    print("CLEARANCE_M median", float(np.median(clearance)), "p10", float(np.percentile(clearance, 10)), "p90", float(np.percentile(clearance, 90)))
    print("GRADE_M_PER_M median", float(np.median(grade)))
    for distance in (30.0, 60.0, 100.0):
        drop = grade * distance
        print(f"HORIZONTAL_MISS_M at {distance:.0f}m median", float(np.median(drop)), "p10", float(np.percentile(drop, 10)), "p90", float(np.percentile(drop, 90)))


if __name__ == "__main__":
    main()
