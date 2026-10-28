"""Read the shared cache into the canonical evaluation representation."""
from pathlib import Path
import numpy as np
from infrauni.eval.resolve_detection import CLASS_NAMES

def ground_truth(cache_dir):
    grouped={}
    for path in sorted(Path(cache_dir).glob('*/sample_*.npz')):
        with np.load(path,allow_pickle=True) as d:
            token=str(d['context_name'])+'_'+str(int(d['frame_index']))
            if token in grouped:raise ValueError('Duplicate token '+token)
            grouped[token]=[dict(translation=b[:3].tolist(),size=b[3:6].tolist(),yaw=float(b[6]),
                velocity=d['gt_velocity'][i].tolist(),name=CLASS_NAMES[int(d['gt_types'][i])],
                future_xy=d['gt_future'][i],future_mask=d['gt_future_mask'][i]) for i,b in enumerate(d['gt_boxes'])]
    if not grouped:raise ValueError('Empty cache '+str(cache_dir))
    return grouped
