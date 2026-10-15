"""Shared labels and nuScenes-mini scene splits."""

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

# Official category name -> detection class.
CATEGORY_TO_CLASS = {
    "vehicle.car": "car",
    "vehicle.truck": "truck",
    "vehicle.construction": "construction_vehicle",
    "vehicle.bus.bendy": "bus",
    "vehicle.bus.rigid": "bus",
    "vehicle.trailer": "trailer",
    "movable_object.barrier": "barrier",
    "vehicle.motorcycle": "motorcycle",
    "vehicle.bicycle": "bicycle",
    "human.pedestrian.adult": "pedestrian",
    "human.pedestrian.child": "pedestrian",
    "human.pedestrian.construction_worker": "pedestrian",
    "human.pedestrian.police_officer": "pedestrian",
    "movable_object.traffic_cone": "traffic_cone",
}

# Official nuScenes mini split (nuscenes.utils.splits).
MINI_TRAIN_SCENES = [
    "scene-0061",
    "scene-0553",
    "scene-0655",
    "scene-0757",
    "scene-0796",
    "scene-1077",
    "scene-1094",
    "scene-1100",
]
MINI_VAL_SCENES = [
    "scene-0103",
    "scene-0916",
]
