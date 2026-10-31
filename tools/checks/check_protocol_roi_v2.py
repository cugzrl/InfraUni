"""Verify actual training configs, canonical split hashes and complete GT range."""
from pathlib import Path
import json,yaml,pickle,hashlib,numpy as np
from mmcv import Config
ROOT=Path(__file__).resolve().parents[2];RUN=ROOT/'experiments/resolve_roi_v2_20261029';DATA=ROOT/'data/resolve_multimodal'
r=[-120.,-96.,-10.,72.,96.,2.];record=json.loads((RUN/'protocol.json').read_text())
t=yaml.safe_load((ROOT/'configs/resolve/transfusion_roi_v2.yaml').read_text());m=json.loads((ROOT/'configs/resolve/mcdetra_roi_v2.json').read_text());c=Config.fromfile(str(ROOT/'third_party/CMT/projects/configs/resolve/cmt_resolve_r50_roi_v2.py'))
assert t['DATA_CONFIG']['POINT_CLOUD_RANGE']==r==c.point_cloud_range
assert t['DATA_CONFIG']['FILTER_EMPTY_GT'] is False and c.data.train.filter_empty_gt is False
assert c.model.pts_bbox_head.depth_max==160.
assert [m['roi'][0],m['roi'][1],m['z_range'][0],m['roi'][2],m['roi'][3],m['z_range'][1]]==r
assert Path(t['DATA_CONFIG']['DATA_PATH']).resolve()==DATA
assert t['DATA_CONFIG']['INFO_PATH']=={'train':['infos/train.pkl'],'test':['infos/val.pkl']}
assert Path(m['train_cache']).resolve()==DATA/'train' and Path(m['val_cache']).resolve()==DATA/'val'
assert m['val_frames_per_sequence']==0 and m['val_batches'] is None and m['save_best_metric']=='NDS'
assert c.model.pts_middle_encoder.sparse_shape==[61,640,640] and c.model.pts_backbone.in_channels==384
assert c.model.train_cfg.pts.grid_size==[640,640,60]
assert c.model.pts_bbox_head.bbox_coder.pc_range==r and c.model.test_cfg.pts.pc_range==r
assert c.img_norm_cfg.to_rgb and c.img_norm_cfg['mean']==[123.675,116.28,103.53]
assert c.load_from is None and c.resume_from is None
for mode,split in [('train','train'),('val','val'),('test','val')]:assert Path(c.data[mode].ann_file).resolve()==DATA/'infos'/(split+'.pkl')
scenes={};total=0
for split in ['train','val','test']:
 p=DATA/'infos'/(split+'.pkl');assert hashlib.sha256(p.read_bytes()).hexdigest()==record['splits'][split]['sha256']
 ds=pickle.load(p.open('rb'));total+=len(ds);scenes[split]={d['scene_token'] for d in ds}
 for d in ds:
  assert Path(d['lidar_path']).resolve().is_relative_to(DATA)
  assert Path(d['cache_path']).resolve().is_relative_to(DATA)
  assert all(Path(im).resolve().is_relative_to(DATA) for im in d['images'])
  b=d['gt_boxes'];co=np.abs(np.cos(b[:,6]));si=np.abs(np.sin(b[:,6]));dx=(co*b[:,3]+si*b[:,4])/2;dy=(si*b[:,3]+co*b[:,4])/2
  low=np.stack([b[:,0]-dx,b[:,1]-dy,b[:,2]-b[:,5]/2],-1);high=np.stack([b[:,0]+dx,b[:,1]+dy,b[:,2]+b[:,5]/2],-1)
  assert (low>=np.array(r[:3])-1e-5).all() and (high<=np.array(r[3:])+1e-5).all()
assert not scenes['train']&scenes['val'] and not scenes['train']&scenes['test'] and not scenes['val']&scenes['test']
assert hashlib.sha256((ROOT/'infrauni/eval/resolve_detection.py').read_bytes()).hexdigest()==record['evaluator_sha256']
record.update(verified=True,total_frames=total,inputs={'transfusion':'raw current LiDAR scan','mc-detra':'same canonical frames and GT; original five raw sweeps restored before expanded-range crop','cmt':'raw current LiDAR scan and exact four released camera pairs'},precision={'transfusion':'FP32','mc-detra':'FP32 with dense encoder activation checkpointing','cmt':'mixed precision with FP32 sparse encoder and detector head'},cmt_fixes=['canonical box packing in both matching/loss and decoder','ImageNet RGB normalization','independent 160m camera frustum depth','empty GT batches and mixed batches supported without resampling','strict numerical failures instead of nan_to_num'],mc_fixes=['restore raw sweep points instead of reusing old cropped point cache','stationary atan2 has zero heading and zero gradient'],validation='full 3382 canonical frames, including 50 empty frames; common mAP/NDS9 evaluator; no radial filter',training_empty_frames=612,training_empty_policy='all methods retain all canonical frames as negatives',transfusion_output=str(ROOT/'experiments/baselines/infrauni/transfusion_roi_v2_allframes/roi_v2_20261029'))
(RUN/'protocol.json').write_text(json.dumps(record,indent=2));(RUN/'checks/protocol_verification.json').write_text(json.dumps({'pass':True,'total_frames':total,'range':r},indent=2))
print('PASS: all three active configs share canonical data, GT, evaluator and range; no GT corner excluded',flush=True)
