"""Build the 6 s / 12-step future-track cache for nuScenes-mini."""

import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from infrauni.data.future_tracks import build_future_tracks

INFO_ROOT = os.path.join(ROOT, "third_party", "TransFusion", "data", "nuscenes")
OUT = os.path.join(ROOT, "data", "nuscenes_mini_future12.pkl")


if __name__ == "__main__":
    paths = [
        os.path.join(INFO_ROOT, "nuscenes_infos_train.pkl"),
        os.path.join(INFO_ROOT, "nuscenes_infos_val.pkl"),
    ]
    frames, covered = build_future_tracks(paths, OUT)
    print("frames", frames, "objects_with_final_step", covered, "path", OUT)
