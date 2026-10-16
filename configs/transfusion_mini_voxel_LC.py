# Runtime overrides for official TransFusion-LC on nuScenes-mini.
# The fusion model, six-camera pipeline, and load_from stay in the official config.
_base_ = ["../third_party/TransFusion/configs/transfusion_nusc_voxel_LC.py"]

data = dict(samples_per_gpu=1, workers_per_gpu=2)
gpu_ids = range(0, 1)
total_epochs = 1
runner = dict(type="EpochBasedRunner", max_epochs=1)
evaluation = dict(interval=1)
checkpoint_config = dict(interval=1)
log_config = dict(interval=20, hooks=[dict(type="TextLoggerHook")])

work_dir = "/home/kemove/devdata1/zrl/InfraUni/experiments/transfusion_voxel_LC_mini"
# Official staged init. Training must stop if this file is absent.
load_from = "checkpoints/fusion_voxel0075_R50.pth"
resume_from = None
