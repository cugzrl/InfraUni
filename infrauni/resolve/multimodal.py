"""Shared file access for LiDAR, camera and fused RESOLVE baselines."""
import hashlib,pickle
from pathlib import Path
import cv2
import numpy as np

DEFAULT_ROOT=Path(__file__).resolve().parents[2]/'data/resolve_multimodal'

def load_infos(split='train',data_root=None):
    root=Path(data_root or DEFAULT_ROOT).resolve()
    if not (root/'_READY').is_file():raise RuntimeError('Dataset validation/publication is incomplete: '+str(root))
    if split not in ('train','val','test'):raise ValueError(split)
    with (root/'infos'/(split+'.pkl')).open('rb') as f:return pickle.load(f)

def instance_ids(info):
    """Stable numeric track IDs for temporal camera models; original IDs stay intact."""
    return np.asarray([int.from_bytes(hashlib.blake2b(str(v).encode(),digest_size=8).digest(),'little')&((1<<63)-1) for v in info['gt_track_ids']],dtype=np.int64)

def read_frame(info,modality='multimodal',temporal_lidar=False):
    if modality not in ('lidar','camera','multimodal'):raise ValueError(modality)
    sample=dict(metadata=info,gt_boxes=info['gt_boxes'],gt_labels=info['gt_labels'],gt_track_ids=info['gt_track_ids'],instance_ids=instance_ids(info),gt_future=info['gt_future'],gt_future_mask=info['gt_future_mask'])
    if modality in ('lidar','multimodal'):
        if temporal_lidar:
            with np.load(info['cache_path'],allow_pickle=False) as z:sample['points']=z['points']
        else:sample['points']=np.fromfile(info['lidar_path'],dtype=np.float32).reshape(-1,4)
    if modality in ('camera','multimodal'):
        images=[]
        for path in info['images']:
            image=cv2.imread(path)
            if image is None:raise FileNotFoundError(path)
            images.append(cv2.cvtColor(image,cv2.COLOR_BGR2RGB))
        sample.update(images=images,K=info['K'],lidar2cam=info['lidar2cam'],projection_mat=info['projection_mat'],camera_valid=info['camera_valid'])
    return sample
