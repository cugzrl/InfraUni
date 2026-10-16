"""Shared nuScenes geometry used by TransFusion-L."""

CLASS_NAMES = [
    "car",
    "truck",
    "construction_vehicle",
    "bus",
    "trailer",
    "barrier",
    "motorcycle",
    "bicycle",
    "pedestrian",
    "traffic_cone",
]
CLASS_TO_ID = {name: index for index, name in enumerate(CLASS_NAMES)}
# Forecasting scores moving agents. Cones and barriers stay in the per-class table.
STATIC_FORECAST_CLASSES = ("barrier", "traffic_cone")
FORECAST_CLASSES = tuple(name for name in CLASS_NAMES if name not in STATIC_FORECAST_CLASSES)

# Pedestrian and traffic cone use a 1x1 heatmap NMS kernel in TransFusion.
PEDESTRIAN_ID = CLASS_TO_ID["pedestrian"]
TRAFFIC_CONE_ID = CLASS_TO_ID["traffic_cone"]

PC_RANGE = [-54.0, -54.0, -5.0, 54.0, 54.0, 3.0]
VOXEL_SIZE = [0.075, 0.075, 0.2]
OUT_SIZE_FACTOR = 8
POST_CENTER_RANGE = [-61.2, -61.2, -10.0, 61.2, 61.2, 10.0]

NUM_VIEWS = 6
CAMERA_TYPES = [
    "CAM_FRONT",
    "CAM_FRONT_RIGHT",
    "CAM_FRONT_LEFT",
    "CAM_BACK",
    "CAM_BACK_LEFT",
    "CAM_BACK_RIGHT",
]

NUM_MODES = 6
NUM_FUTURE = 12
FUTURE_HZ = 2.0

IMG_MEAN = [123.675, 116.28, 103.53]
IMG_STD = [58.395, 57.12, 57.375]

NUSCENES_ROOT = "/home/kemove/devdata1/zrl/bevfusion/data/nuscenes"
RESNET50_WEIGHTS = "/home/kemove/.cache/torch/hub/checkpoints/resnet50-0676ba61.pth"

# ViP3D / MASAR operating point for joint detection-forecast metrics.
MATCH_DIST = 2.0
EPA_FDE = 2.0
EPA_ALPHA = 0.5
FORECAST_SCORE = 0.1
