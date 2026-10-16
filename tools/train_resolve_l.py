"""Launch the official RESOLVE trainer on a LiDAR-only split.

The model file stays tools/cfgs/sunlakes_models/transfusion.yaml. This wrapper
only selects the data root and turns camera-free conversion into a training run.
"""

import os
import sys

TOOLS = "/home/kemove/devdata1/zrl/RESOLVE/tools"
os.chdir(TOOLS)
sys.path.insert(0, "/home/kemove/devdata1/zrl/RESOLVE")
sys.path.insert(0, TOOLS)
sys.path.append("/home/kemove/anaconda3/envs/bevfusion/lib/python3.8/site-packages")

DATA = {
    "oneseq": "/home/kemove/devdata1/zrl/dataset/RESOLVE_lidar_smoke",
    "official": "/home/kemove/devdata1/zrl/dataset/RESOLVE_layout",
}


def main():
    split = sys.argv[1] if len(sys.argv) > 1 else "oneseq"
    extra = sys.argv[2:]
    sys.argv = [
        "train.py",
        "--cfg_file", "cfgs/sunlakes_models/transfusion.yaml",
        "--extra_tag", "infrauni_%s" % split,
        *extra,
        "--set",
        "DATA_CONFIG.DATA_PATH", DATA[split],
        "DATA_CONFIG.DATA_AUGMENTOR.DISABLE_AUG_LIST", "gt_sampling",
    ]
    from train import main as train_main

    train_main()


if __name__ == "__main__":
    main()
