"""Point the official RESOLVE converter at the downloaded bags.

The converter reads ``resource/<bag>/128_128b_label`` and one calibration file
per camera. The release stores clouds in ``128_128b_sync`` and a calibration
text per frame. This script only adds symlinks and copies the first calibration
text of each bag. It does not move or delete the archives.
"""

import os
import shutil

SOURCE = "/home/kemove/devdata1/zrl/dataset/RESOLVE"
LAYOUT = "/home/kemove/devdata1/zrl/dataset/RESOLVE_layout/2026-01-04_18_02_01-export"
SENSORS = {
    "128_128b_label": "128_128b_sync",
    "64_64b_label": "64_64b_sync",
    "16_16b_label": "16_16b_sync",
    "axis1_label": "axis1_sync",
    "axis2_label": "axis2_sync",
    "axis3_label": "axis3_sync",
    "axis4_label": "axis4_sync",
}
CALIBRATION = {
    "axis1_os128b.txt": "axis1__ouster128b",
    "axis2_os128b.txt": "axis2__ouster128b",
    "axis3_os128b.txt": "axis3__ouster128b",
    "axis4_os128b.txt": "axis4__ouster128b",
}


def _link(src, dst):
    if os.path.lexists(dst):
        return
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    os.symlink(src, dst)


def main():
    export = LAYOUT
    os.makedirs(export, exist_ok=True)
    annotation = os.path.join(export, "HSQJN_3.json")
    if not os.path.lexists(annotation):
        os.symlink(os.path.join(SOURCE, "HSQJN_3.json"), annotation)
    linked = 0
    calibs = 0
    for name in sorted(os.listdir(SOURCE)):
        bag = os.path.join(SOURCE, name)
        if not name.startswith("rosbag2_") or not os.path.isdir(bag):
            continue
        for alias, real in SENSORS.items():
            src = os.path.join(bag, real)
            if os.path.isdir(src):
                _link(src, os.path.join(export, "resource", name, alias))
                linked += 1
        calib_root = os.path.join(export, "per_bag", name)
        os.makedirs(calib_root, exist_ok=True)
        for filename, folder in CALIBRATION.items():
            src_dir = os.path.join(bag, "calibration_txt", folder)
            dst = os.path.join(calib_root, filename)
            if os.path.exists(dst) or not os.path.isdir(src_dir):
                continue
            files = sorted(os.listdir(src_dir))
            if not files:
                continue
            shutil.copyfile(os.path.join(src_dir, files[0]), dst)
            calibs += 1
    print("links", linked, "calibration_files", calibs, "layout", export)


if __name__ == "__main__":
    main()
