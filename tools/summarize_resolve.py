"""Summarize RESOLVE annotations without converting point clouds.

The official converter splits frames at random. This script writes a scene-level
split instead, so adjacent frames of one track stay in the same subset.
"""

import collections
import json
import os
import random

ROOT = "/home/kemove/devdata1/zrl/dataset/RESOLVE"
ANNOTATION = os.path.join(ROOT, "HSQJN_3.json")
SPLIT_PATH = os.path.join(
    os.path.dirname(__file__), "..", "infrauni", "data", "resolve_scene_split.json"
)


def _timestamp(filename):
    stem = filename[:-4] if filename.endswith(".bin") else filename
    sec, nano = stem.split("-")
    return int(sec) + int(nano) / 1e9


def main():
    frames = json.load(open(ANNOTATION, "r"))
    scenes = collections.defaultdict(list)
    classes = collections.Counter()
    tracks = collections.defaultdict(list)
    missing = collections.Counter()
    for frame in frames:
        parts = frame["info"].split("/")
        bag, filename = parts[-3], parts[-1]
        timestamp = _timestamp(filename)
        scene = frame["batchId"]
        scenes[scene].append(timestamp)
        if not os.path.exists(os.path.join(ROOT, bag, "128_128b_sync", filename)):
            missing[bag] += 1
        for label in frame["labels"]:
            if label.get("drawType") != "box3d":
                continue
            classes[label["label"]] += 1
            tracks[(scene, int(label["id"]))].append(timestamp)

    intervals = []
    for timestamps in scenes.values():
        timestamps = sorted(set(timestamps))
        intervals.extend(b - a for a, b in zip(timestamps, timestamps[1:]))
    intervals.sort()
    spans = []
    full_6s = full_3s = 0
    for observations in tracks.values():
        timestamps = sorted(set(observations))
        if len(timestamps) < 2:
            continue
        kept = [timestamps[0]]
        for timestamp in timestamps[1:]:
            if timestamp - kept[-1] >= 0.45:
                kept.append(timestamp)
        span = kept[-1] - kept[0]
        spans.append(span)
        if span >= 6.0 and len(kept) >= 13:
            full_6s += 1
        if span >= 3.0 and len(kept) >= 7:
            full_3s += 1
    spans.sort()
    median_dt = intervals[len(intervals) // 2]
    print(
        "frames", len(frames),
        "scenes", len(scenes),
        "tracks", len(spans),
        "median_dt", round(median_dt, 4),
        "median_span_s", round(spans[len(spans) // 2], 3),
        "tracks_3s_at_2hz", full_3s,
        "tracks_6s_at_2hz", full_6s,
        "bags_missing_bins", len(missing),
        "missing_bins", sum(missing.values()),
    )
    print("classes", classes.most_common())

    ordered = sorted(scenes)
    # Fixed permutation, independent of the official per-frame split.
    permutation = list(range(len(ordered)))
    random.Random(42).shuffle(permutation)
    n_holdout = int(round(0.15 * len(ordered)))
    val_ids = [ordered[i] for i in permutation[:n_holdout]]
    test_ids = [ordered[i] for i in permutation[n_holdout: 2 * n_holdout]]
    train_ids = [ordered[i] for i in permutation[2 * n_holdout:]]
    payload = {
        "seed": 42,
        "rule": "whole scene, one bag each",
        "train": train_ids,
        "val": val_ids,
        "test": test_ids,
    }
    os.makedirs(os.path.dirname(SPLIT_PATH), exist_ok=True)
    json.dump(payload, open(SPLIT_PATH, "w"), indent=2)
    print("split", len(train_ids), len(val_ids), len(test_ids), SPLIT_PATH)


if __name__ == "__main__":
    main()
