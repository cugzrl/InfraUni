"""Canonical RESOLVE GT plus five raw LiDAR sweeps over the expanded ROI."""
from pathlib import Path
from functools import lru_cache
import json,pickle
from bisect import bisect_left
import numpy as np
from detra_repro.data.dataset import CachedWaymoDataset as BaseCachedDataset
RANGE=np.array([-120.,-96.,-10.,72.,96.,2.])

def timestamp_ns(path):
    a,b=Path(path).stem.split('-');return int(a)*10**9+int(b.ljust(9,'0'))

@lru_cache(maxsize=64)
def raw_xyz(path):
    p=np.fromfile(path,dtype=np.float32).reshape(-1,4)[:,:3]
    return p[np.isfinite(p).all(1)&((p>=RANGE[:3])&(p<RANGE[3:])).all(1)]

class ResolveRawDataset(BaseCachedDataset):
    def __init__(self,cache_dir,*args,**kwargs):
        super().__init__(cache_dir,*args,**kwargs)
        root=Path(cache_dir).resolve();data=root.parent
        if data.name!='resolve_multimodal' or not (data/'_READY').is_file():raise ValueError('Use the published resolve_multimodal root')
        with (data/'infos'/(root.name+'.pkl')).open('rb') as f:self.infos=pickle.load(f)
        self.files=[Path(d['cache_path']) for d in self.infos]
        self.raw_by_scene={}
        for d in self.infos:
            scene=d['scene_token']
            if scene in self.raw_by_scene:continue
            paths=sorted(Path(d['lidar_path']).parent.glob('*.bin'),key=timestamp_ns)
            self.raw_by_scene[scene]=([timestamp_ns(p) for p in paths],paths)
        print('RESOLVE_RAW_DATASET',root.name,len(self.files),'range',RANGE.tolist(),'history_sweeps',5,flush=True)
    def __getitem__(self,index):
        out=super().__getitem__(index);d=self.infos[index]
        assert np.array_equal(out['gt_boxes'],d['gt_boxes'][:,:7])
        assert np.array_equal(out['gt_types'],d['gt_labels'])
        pieces=[];now=int(d['timestamp_ns'])
        times,paths=self.raw_by_scene[d['scene_token']]
        # Cached dt identifies the released sweeps, not their cropped points.
        # Match raw filename timestamps within floating-point dt precision.
        # The 2 us tolerance is far below the 100 ms inter-scan interval.
        for old_dt in np.unique(out['points'][:,3]):
            target=now+int(round(float(old_dt)*1e9));at=bisect_left(times,target)
            choices=[i for i in [at-1,at] if 0<=i<len(times)]
            nearest=min(choices,key=lambda i:abs(times[i]-target))
            if abs(times[nearest]-target)>2000:raise ValueError('Cannot identify cached raw sweep '+d['token'])
            dt=(times[nearest]-now)/1e9;xyz=raw_xyz(str(paths[nearest]))
            pieces.append(np.column_stack((xyz,np.full(len(xyz),dt,np.float32))))
        if not pieces:raise ValueError('Missing current scan '+d['token'])
        out['points']=np.concatenate(pieces).astype(np.float32,copy=False)
        return out
