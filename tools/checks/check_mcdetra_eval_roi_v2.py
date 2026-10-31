from pathlib import Path
import sys,os,json,torch,numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT),str(ROOT/'third_party/MC-DeTra'),str(ROOT/'third_party/CMT/.runtime_deps')]
from infrauni.resolve.mcdetra_dataset import ResolveRawDataset
from infrauni.resolve.mcdetra_evaluation import SharedDetectionAccumulator
from detra_repro.data.dataset import collate_cached_waymo
from detra_repro.eval.resolve_detection import load_resolve_modules
from detra_repro.train_cached import forward_batch
cp=ROOT/'experiments/resolve_roi_v2_20261029/checks/mc_smoke/model_step25.pt'
ckpt,dc,mc,lidar,proposal,map_encoder,model=load_resolve_modules(cp,'cuda')
ds=ResolveRawDataset(ROOT/'data/resolve_multimodal/val');batch=collate_cached_waymo([ds[0]],max_points=400000,max_objects=600,max_map_tokens=16,deterministic_seed=0)
with torch.no_grad():
 pred,loss,targets=forward_batch(batch,lidar,proposal,map_encoder,model,dc,mc,'cuda',dense_history_motion_weight=.3,occ_aux_future=True)
acc=SharedDetectionAccumulator(ds);acc.add(batch,pred)
assert len(acc.pred)==1 and all(np.isfinite(x['score']) for boxes in acc.pred.values() for x in boxes)
print('REAL_MC_FORWARD_AND_SHARED_EVAL_ADAPTER_PASS',len(next(iter(acc.pred.values()))),flush=True)
# Complete canonical validation GT-as-pred sanity check, never used for training.
acc.pred={k:[dict(x,score=1.) for x in v] for k,v in acc.truth.items()}
os.environ['INFRAUNI_SHARED_EVAL_DIR']=str(ROOT/'experiments/resolve_roi_v2_20261029/checks/oracle_eval')
metric=acc.finish();present=len({b['name'] for rows in acc.truth.values() for b in rows});expected=present/10
assert abs(metric['mAP']-expected)<1e-5 and abs(metric['NDS']-expected)<1e-5,(metric,expected)
print('FULL_3382_FRAME_SHARED_PROTOCOL_ORACLE_PASS',metric,flush=True)
