"""One authoritative cache/split/GT for all RESOLVE models."""
import json
import pickle
from pathlib import Path
import numpy as np
from pcdet.datasets.carla.carla_dataset import CarlaDataset
from infrauni.eval.resolve_detection import CLASS_NAMES, evaluate_boxes

class ResolveSequenceDataset(CarlaDataset):
    def get_lidar_pts(self,index):
        points=np.fromfile(self.infos[index]['lidar_path'],dtype=np.float32).reshape(-1,4)
        return np.column_stack((points,np.zeros(len(points),dtype=np.float32)))

    def evaluation(self,det_annos,class_names,**kwargs):
        gt={};pred={}
        for info in self.infos:
            token=info['token'];gt[token]=[]
            for b,n in zip(info['gt_boxes'],info['gt_names']):
                gt[token].append(dict(translation=b[:3].tolist(),size=b[3:6].tolist(),yaw=float(b[6]),velocity=b[7:9].tolist(),name=str(n)))
        for a in det_annos:
            token=a['metadata']['token'];pred[token]=[]
            for b,n,s in zip(a['boxes_lidar'],a['name'],a['score']):
                pred[token].append(dict(translation=b[:3].tolist(),size=b[3:6].tolist(),yaw=float(b[6]),velocity=b[7:9].tolist(),name=str(n),score=float(s)))
        result=evaluate_boxes(gt,pred)
        out=Path(kwargs['output_path']);out.mkdir(parents=True,exist_ok=True)
        (out/'metrics_resolve.json').write_text(json.dumps(result,indent=2))
        return 'RESOLVE mAP %.5f NDS9 %.5f (legacy10 %.5f)'%(result['mAP'],result['NDS'],result['NDS_legacy10']),dict(mAP=result['mAP'],NDS=result['NDS'])
