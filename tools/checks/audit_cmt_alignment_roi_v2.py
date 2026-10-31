"""Inspect clocks, canonical geometry and per-class gap without changing data."""
import json,pickle,numpy as np,cv2
from pathlib import Path
from collections import defaultdict
R=Path('/home/kemove/devdata1/zrl/InfraUni');D=R/'data/resolve_multimodal';O=R/'experiments/resolve_roi_v2_20261029/checks/cmt_gap';O.mkdir(exist_ok=True)
infos={s:pickle.load((D/'infos'/f'{s}.pkl').open('rb')) for s in ['train','val','test']};groups=defaultdict(list)
for ds in infos.values():
 for d in ds:groups[d['scene_token']].append(d)
res={};offsets={};raw=[];pair_errors=0;matrix_error=0.;visible=0;total=0
for scene,rows in groups.items():
 dt=np.array([(d['image_timestamp_ns']-d['timestamp_ns'])/1e9 for d in rows]);off=np.median(dt,axis=0);offsets[scene]=off.tolist()
 for d,dd in zip(rows,dt):res[d['token']]=(dd-off).tolist()
 raw.extend(dt.ravel().tolist())
for d in infos['val']:
 b=d['gt_boxes'];p=np.column_stack([b[:,:3],np.ones(len(b))]);hit=np.zeros(len(b),bool)
 for v in range(4):
  P=np.eye(4);P[:3]=d['K'][v]@d['lidar2cam'][v][:3];matrix_error=max(matrix_error,float(abs(P-d['projection_mat'][v]).max()));q=p@P.T;depth=q[:,2];uv=q[:,:2]/np.maximum(depth[:,None],1e-6);hit|=(depth>0)&(uv[:,0]>=0)&(uv[:,0]<1920)&(uv[:,1]>=0)&(uv[:,1]<1080)
  name=Path(d['calibration_paths'][v]).name;pair_errors+=not name.startswith(Path(d['images'][v]).stem+'__'+Path(d['lidar_path']).stem)
 visible+=int(hit.sum());total+=len(b)
a=np.abs(np.array(list(res.values())));tf=json.load((R/'experiments/baselines/infrauni/transfusion_roi_v2_allframes/roi_v2_20261029/eval/eval_with_train/epoch_20/val/final_result/data/metrics_resolve.json').open());cm=json.load((R/'experiments/resolve_roi_v2_20261029/cmt/val_metrics.json').open());gap={k:dict(cmt=cm['mean_dist_aps'][k],transfusion=tf['mean_dist_aps'][k],difference=tf['mean_dist_aps'][k]-cm['mean_dist_aps'][k]) for k in cm['mean_dist_aps']}
report=dict(frames={k:len(v) for k,v in infos.items()},clock_method='Per-sequence/per-camera median offset removal; estimated jitter only, not independently verified physical latency. User confirms raw 38-89s is fixed clock offset.',raw_offset_seconds_quantiles=np.quantile(raw,[0,.5,1]).tolist(),median_centered_abs_residual_ms_quantiles=np.quantile(a,[.5,.9,.95,.99,1]).tolist(),median_centered_fraction_above_100ms=float((a>.1).mean()),pair_filename_errors=pair_errors,K_times_T_max_abs_error=matrix_error,val_GT_centers_in_any_raw_camera=dict(visible=visible,total=total,fraction=visible/total),per_class=gap)
report['median_centered_abs_residual_ms_quantiles']=[v*1000 for v in report['median_centered_abs_residual_ms_quantiles']];(O/'clock_offsets.json').write_text(json.dumps(offsets));(O/'frame_clock_jitter.json').write_text(json.dumps(res));(O/'alignment_audit.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
