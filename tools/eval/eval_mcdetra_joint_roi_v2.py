"""Full canonical MC-DeTra detection and 5s forecast evaluation of a frozen checkpoint."""
import sys,os,json,time,pickle,inspect,random,hashlib,argparse,shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT),str(ROOT/'tools/train'),str(ROOT/'third_party/MC-DeTra'),str(ROOT/'third_party/CMT/.runtime_deps')]
import train_mcdetra_resolve_v2
import torch,numpy as np
from infrauni.resolve.mcdetra_dataset import ResolveRawDataset
from infrauni.resolve.mcdetra_evaluation import SharedDetectionAccumulator
from infrauni.eval.resolve_detection import evaluate_boxes
from infrauni.eval.resolve_forecast import evaluate_end_to_end
from detra_repro.eval.resolve_detection import load_resolve_modules
from detra_repro.train_cached import build_loader,forward_batch
parser=argparse.ArgumentParser();parser.add_argument('--checkpoint');parser.add_argument('--output',default=str(ROOT/'experiments/resolve_roi_v2_20261029/mc-detra/joint_eval_best38714'));parser.add_argument('--expected-step',type=int);args=parser.parse_args()
OUT=Path(args.output).resolve();OUT.mkdir(parents=True,exist_ok=True)
if args.checkpoint:
 source=Path(args.checkpoint).resolve();snapshot=OUT/'checkpoint.pt'
 if not snapshot.exists():shutil.copy2(source,snapshot)
 elif source!=snapshot:assert hashlib.sha256(source.read_bytes()).digest()==hashlib.sha256(snapshot.read_bytes()).digest(),'Use a new output directory for a different checkpoint'
assert (OUT/'checkpoint.pt').is_file()
cfg=json.load(open(ROOT/'experiments/resolve_roi_v2_20261029/mc-detra/config.json'))['config']
seed=int(cfg['deterministic_seed']);random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
ckpt,dc,mc,lidar,proposal,map_encoder,model=load_resolve_modules(OUT/'checkpoint.pt','cuda')
print('CHECKPOINT_STEP',ckpt.get('step'),flush=True)
if args.expected_step is not None:assert int(ckpt['step'])==args.expected_step
for module in [lidar,proposal,map_encoder,model]:
 if module is not None:module.eval()
ds=ResolveRawDataset(ROOT/'data/resolve_multimodal/val');assert len(ds)==3382
_,loader=build_loader(ROOT/'data/resolve_multimodal/val',1,cfg['max_points'],mc.num_queries,False,max_map_tokens=cfg['max_map_tokens'],deterministic_seed=seed,num_workers=2,pin_memory=True,dataset=ds)
kwargs={k:cfg[k] for k in inspect.signature(forward_batch).parameters if k in cfg and k not in ['device']};kwargs['compute_diagnostics']=False
assert kwargs.get('proposal_mode','decoded')=='decoded' and not kwargs.get('gt_history_occupancy',False) and kwargs.get('query_velocity_source','none')=='none',kwargs
print('INFERENCE_KWARGS',kwargs,flush=True)
acc=SharedDetectionAccumulator(ds)
for d in ds.infos:
 for b,f,mask in zip(acc.truth[d['token']],d['gt_future'],d['gt_future_mask']):b.update(future_xy=np.asarray(f),future_mask=np.asarray(mask,dtype=bool))
started=time.monotonic()
with torch.inference_mode():
 for i,batch in enumerate(loader):
  pred,_,_=forward_batch(batch,lidar,proposal,map_encoder,model,dc,mc,'cuda',**kwargs);acc.add(batch,pred)
  cls=pred['class_logits'].softmax(-1);lab=cls.argmax(-1);scores=pred['object_logits'].sigmoid()*cls.gather(-1,lab.unsqueeze(-1)).squeeze(-1);keep=torch.topk(scores[0],min(500,scores.shape[1])).indices
  fut=pred['future_xy'][0,keep].cpu().numpy();prob=pred['mode_logits'][0,keep].softmax(-1).cpu().numpy();token=str(batch['context_names'][0])+'_'+str(int(batch['frame_indices'][0]))
  assert np.isfinite(fut).all() and np.isfinite(prob).all()
  for b,f,p in zip(acc.pred[token],fut,prob):
   if b['score']>=.1:b.update(future_xy=f.copy(),mode_probs=p.copy())
  if (i+1)%100==0:
   status=dict(frames=i+1,total=3382,elapsed_s=time.monotonic()-started);(OUT/'progress.json').write_text(json.dumps(status));print('PROGRESS',status,flush=True)
assert set(acc.pred)==set(acc.truth) and len(acc.pred)==3382
print('SCORING',flush=True);det=evaluate_boxes(acc.truth,acc.pred);forecast=evaluate_end_to_end(acc.truth,acc.pred)
result=dict(checkpoint=str(OUT/'checkpoint.pt'),checkpoint_step=ckpt['step'],checkpoint_sha256=hashlib.sha256((OUT/'checkpoint.pt').read_bytes()).hexdigest(),frames=3382,dataset_root=str(ROOT/'data/resolve_multimodal'),point_cloud_range=[-120,-96,-10,72,96,2],detection=det,forecast=forecast,elapsed_s=time.monotonic()-started)
(OUT/'metrics.json').write_text(json.dumps(result,indent=2));pickle.dump(acc.pred,(OUT/'predictions.pkl').open('wb'),protocol=4)
print('RESULT',json.dumps({k:det[k] for k in ['mAP','NDS','protocol']}),flush=True);print('FORECAST',json.dumps({k:v for k,v in forecast.items() if k!='per_class'}),flush=True)
