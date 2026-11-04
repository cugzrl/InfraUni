import sys,json,time,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'third_party/UniBEV/tools'))
import resolve_runtime_unibev
import torch,numpy as np
from mmcv import Config
from mmcv.parallel import collate,scatter
from mmdet3d.datasets import build_dataset
from mmdet3d.models import build_model
cfg=Config.fromfile(str(ROOT/'third_party/UniBEV/projects/UniBEV/configs/unibev/unibev_resolve_roi_v2.py'))
run=Path(cfg.work_dir)
train=build_dataset(cfg.data.train);val=build_dataset(cfg.data.val)
protocol=json.loads((ROOT/'experiments/resolve_roi_v2_20261029/protocol.json').read_text())
for split,n in [('train',11909),('val',3382)]:
 f=ROOT/'data/resolve_multimodal/infos'/f'{split}.pkl'
 assert hashlib.sha256(f.read_bytes()).hexdigest()==protocol['splits'][split]['sha256']
assert len(train)==11909 and len(val)==3382
assert sum(len(x['gt_boxes'])==0 for x in train.data_infos)==612
assert sum(len(x['gt_boxes'])==0 for x in val.data_infos)==50
m=build_model(cfg.model);m.init_weights();m.cuda().train()
m.pts_bbox_head.transformer.drop_modality=0.
results=[]
empty=next(i for i,x in enumerate(train.data_infos) if len(x['gt_boxes'])==0)
for idx in [0,empty]:
 data=scatter(collate([train[idx],train[idx]],samples_per_gpu=2),[0])[0]
 start=time.time();m.zero_grad(set_to_none=True)
 losses=m(return_loss=True,**data)
 loss=sum(x for k,x in losses.items() if 'loss' in k)
 assert torch.isfinite(loss);loss.backward()
 branch={}
 for name in ['img_backbone','pts_middle_encoder','pts_bbox_head']:
  params=list(getattr(m,name).parameters());grads=[p.grad for p in params if p.grad is not None]
  assert grads and all(torch.isfinite(g).all() for g in grads)
  branch[name]=float(torch.stack([g.norm() for g in grads]).norm())
 results.append(dict(index=idx,empty_gt=idx==empty,loss=float(loss),losses={k:float(v) for k,v in losses.items()},branch_grad_norm=branch,seconds=time.time()-start,peak_mb=torch.cuda.max_memory_allocated()/2**20))
 print(json.dumps(results[-1]),flush=True)
(run/'smoke_checks.json').write_text(json.dumps(dict(train_frames=len(train),val_frames=len(val),checks=results),indent=2))
print('UNIBEV_SMOKE_OK',flush=True)
