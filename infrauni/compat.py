"""Load the official TransFusion head on BEVFusion's compiled mmdet3d."""

from __future__ import annotations

import importlib.util
import inspect
import sys
import textwrap
from pathlib import Path

BEVFUSION_ROOT = Path("/home/kemove/devdata1/zrl/bevfusion")
TRANSFUSION_ROOT = Path("/home/kemove/devdata1/zrl/TransFusion")
HEAD_PATH = TRANSFUSION_ROOT / "mmdet3d/models/dense_heads/transfusion_head.py"
COORD_PATH = TRANSFUSION_ROOT / "mmdet3d/models/fusion_layers/coord_transform.py"

_READY = False
_HEAD_CLS = None


def setup():
    """Put BEVFusion on the path and install the two symbols the head needs."""
    global _READY
    root = str(BEVFUSION_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)
    if _READY:
        return
    import mmdet3d.models.utils as utils

    if not hasattr(utils, "clip_sigmoid"):

        def clip_sigmoid(x, eps=1e-4):
            return x.sigmoid().clamp(min=eps, max=1.0 - eps)

        utils.clip_sigmoid = clip_sigmoid

    name = "mmdet3d.models.fusion_layers"
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, COORD_PATH)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    _READY = True


def _patch_batch1_reverse(cls):
    """Always undo 3D aug before image projection.

    The local TransFusion checkout skips that undo when the batch size is 1.
    Training uses batch 1, so the skip would project augmented boxes with the
    original calibration.
    """
    source = inspect.getsource(cls.forward_single)
    old = (
        "if batch_size == 1:  # skip during inference to save time\n"
        "                    points = query_pos_3d_with_corners.T\n"
        "                else:\n"
        "                    points = apply_3d_transformation("
        "query_pos_3d_with_corners.T, 'LIDAR', img_metas[sample_idx], "
        "reverse=True).detach()"
    )
    new = (
        "points = apply_3d_transformation("
        "query_pos_3d_with_corners.T, 'LIDAR', img_metas[sample_idx], "
        "reverse=True).detach()"
    )
    if old not in source:
        raise RuntimeError("Could not patch TransFusion image projection.")
    source = textwrap.dedent(source.replace(old, new))
    namespace = {}
    exec(source, cls.forward_single.__globals__, namespace)
    cls.forward_single = namespace["forward_single"]


def load_transfusion_head_cls():
    """Import the official head class without compiling its private CUDA ops."""
    global _HEAD_CLS
    setup()
    if _HEAD_CLS is not None:
        return _HEAD_CLS
    from mmdet3d.models.builder import HEADS

    # bevfusion already registers a LiDAR-only head under the same name.
    HEADS.module_dict.pop("TransFusionHead", None)
    spec = importlib.util.spec_from_file_location("infrauni_transfusion_head", HEAD_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    _patch_batch1_reverse(module.TransFusionHead)
    _HEAD_CLS = module.TransFusionHead
    return _HEAD_CLS
