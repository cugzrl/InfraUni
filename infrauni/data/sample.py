"""Dataset contract shared by nuScenes, RESOLVE and V2X-Seq.

Every training sample is one synchronized frame in the current LiDAR frame.
Roadside loaders should fill the same keys. Detection metrics stay
dataset-specific; do not call a roadside score "nuScenes NDS".
"""

SAMPLE_KEYS = (
    "points",          # FloatTensor [P, 5], x y z intensity time
    "img",             # FloatTensor [V, 3, H, W], ImageNet-normalized RGB
    "img_meta",        # lidar2img [V, 4, 4], scale_factor, 3D aug, token
    "gt_bboxes_3d",    # LiDARInstance3DBoxes, bottom center, SECOND yaw
    "gt_labels_3d",    # LongTensor [M]
    "gt_fut",          # FloatTensor [M, T, 2], future xy in the current frame
    "gt_fut_mask",     # FloatTensor [M, T], 1 where that future step exists
    "token",
)


class RoadsideDataset:
    """Placeholder for RESOLVE and V2X-Seq.

    A later loader only needs to produce ``SAMPLE_KEYS``. The detector, the
    DeTra decoder and the forecast metrics do not depend on nuScenes paths.
    """

    name = "roadside"

    def __init__(self, root, split="train"):
        self.root = root
        self.split = split

    def __len__(self):
        raise NotImplementedError(
            f"{self.name} is not wired yet. Implement __getitem__ with keys {SAMPLE_KEYS}."
        )

    def __getitem__(self, index):
        raise NotImplementedError(SAMPLE_KEYS)


class ResolveDataset(RoadsideDataset):
    name = "RESOLVE"


class V2XSeqDataset(RoadsideDataset):
    name = "V2X-Seq"
