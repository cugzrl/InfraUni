# Runtime overrides for official TransFusion-L on nuScenes-mini.
# Model, pipeline, losses, and augmentation stay in the official config.
_base_ = ["../third_party/TransFusion/configs/transfusion_nusc_voxel_L.py"]

# Official schedule assumes 8 GPUs x 2 samples. This machine uses 1 visible GPU.
data = dict(samples_per_gpu=1, workers_per_gpu=2)
gpu_ids = range(0, 1)

# One epoch is a smoke run. It is not the 20-epoch full-nuScenes schedule.
total_epochs = 1
runner = dict(type="EpochBasedRunner", max_epochs=1)
evaluation = dict(interval=1)
checkpoint_config = dict(interval=1)
log_config = dict(interval=20, hooks=[dict(type="TextLoggerHook")])

work_dir = "/home/kemove/devdata1/zrl/InfraUni/experiments/transfusion_voxel_L_mini"
load_from = None
resume_from = None
