"""Dataset adapters. New roadside sets should emit the same sample fields."""

from infrauni.data.nuscenes_mini import NuScenesMiniJoint, build_infos, collate_batch

__all__ = ["NuScenesMiniJoint", "build_infos", "collate_batch"]
