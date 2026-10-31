"""Accumulate MC-DeTra predictions with the same full-split RESOLVE evaluator."""
from pathlib import Path
import json,pickle,os,time
import numpy as np
import torch
from infrauni.eval.resolve_detection import CLASS_NAMES,evaluate_boxes
from detra_repro.losses import current_boxes_from_pred

class SharedDetectionAccumulator:
    def __init__(self,dataset):
        while hasattr(dataset,'dataset'):dataset=dataset.dataset
        self.root=Path(dataset.cache_dir).resolve();data=self.root.parent
        with (data/'infos'/(self.root.name+'.pkl')).open('rb') as f:self.infos=pickle.load(f)
        self.truth={d['token']:[dict(translation=b[:3].tolist(),size=b[3:6].tolist(),yaw=float(b[6]),velocity=b[7:9].tolist(),name=CLASS_NAMES[int(c)]) for b,c in zip(d['gt_boxes'],d['gt_labels'])] for d in self.infos}
        self.pred={}
    def add(self,batch,pred):
        boxes=current_boxes_from_pred(pred)
        probs=pred['class_logits'].softmax(-1);labels=probs.argmax(-1)
        scores=pred['object_logits'].sigmoid()*probs.gather(-1,labels.unsqueeze(-1)).squeeze(-1)
        modes=pred['mode_logits'].softmax(-1).argmax(-1)
        for j in range(boxes.shape[0]):
            token=str(batch['context_names'][j])+'_'+str(int(batch['frame_indices'][j]));assert token not in self.pred
            b=boxes[j];q=torch.arange(len(b),device=b.device)
            future=pred['future_xy'][j];vel=(future[q,modes[j],0,:]-b[:,:2])/.5
            keep=torch.topk(scores[j],min(500,len(b))).indices
            bb=b[keep].detach().cpu().numpy();vv=vel[keep].detach().cpu().numpy();ss=scores[j][keep].detach().cpu().numpy();ll=labels[j][keep].detach().cpu().numpy()
            assert np.isfinite(bb).all() and np.isfinite(vv).all() and np.isfinite(ss).all()
            self.pred[token]=[dict(translation=x[:3].tolist(),size=x[3:6].tolist(),yaw=float(x[6]),velocity=v.tolist(),name=CLASS_NAMES[int(c)],score=float(s)) for x,v,s,c in zip(bb,vv,ss,ll)]
    def finish(self):
        if set(self.truth)!=set(self.pred):raise ValueError('Validation must include every canonical frame exactly once')
        report=evaluate_boxes(self.truth,self.pred);report['frames']=len(self.truth);report['dataset_root']=str(self.root.parent)
        out=Path(os.environ['INFRAUNI_SHARED_EVAL_DIR']);out.mkdir(parents=True,exist_ok=True)
        stamp=time.strftime('%Y%m%d_%H%M%S');text=json.dumps(report,indent=2)
        (out/('metrics_'+stamp+'.json')).write_text(text);(out/'latest_metrics.json').write_text(text)
        print('SHARED_RESOLVE_EVAL',len(self.truth),'mAP',report['mAP'],'NDS',report['NDS'],flush=True)
        return {'mAP':float(report['mAP']),'NDS':float(report['NDS'])}
