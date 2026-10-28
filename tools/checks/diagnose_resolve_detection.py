"""Evidence-based RESOLVE detection audit; does not train or alter weights."""
from __future__ import annotations
import os,sys,json,pickle,logging,re
from pathlib import Path
import numpy as np
import cv2
import torch
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT),str(ROOT/'third_party/RESOLVE'),str(ROOT/'third_party/MC-DeTra')]
sys.path.append('/home/kemove/anaconda3/envs/bevfusion/lib/python3.8/site-packages')
from pcdet.config import cfg,cfg_from_yaml_file
from pcdet.models import build_network,load_data_to_gpu
from pcdet.utils.box_utils import boxes_to_corners_3d as official_corners
from infrauni.resolve.dataset import ResolveSequenceDataset
from detra_repro.data.resolve_camera import boxes_to_corners_3d,project_points,parse_calibration,BOX_EDGES

OUT=ROOT/'experiments/diagnostics/transfusion_mcdetra_20261028'
OUT.mkdir(parents=True,exist_ok=True)
NAMES=('car','truck','construction_vehicle','bus','trailer','barrier','motorcycle','bicycle','pedestrian','traffic_cone')
def read_infos(part):
    with (ROOT/'data/resolve_protocol'/(part+'.pkl')).open('rb') as f:return pickle.load(f)
def inside(boxes):
    return ((boxes[:,:3]>=np.array([-60,-60,-8]))&(boxes[:,:3]<np.array([88.8,88.8,0]))).all(axis=1)
def match(info,pred,threshold,class_aware=True):
    gt=info['gt_boxes'];taken=set();pairs=[]
    order=np.argsort(-np.asarray(pred['scores']),kind='stable')
    for j in order:
        if pred['scores'][j]<threshold:continue
        allowed=[i for i in range(len(gt)) if i not in taken and (not class_aware or info['gt_names'][i]==pred['names'][j])]
        if not allowed:continue
        dist=np.linalg.norm(gt[allowed,:2]-np.asarray(pred['boxes'][j])[:2],axis=1)
        k=int(dist.argmin())
        if dist[k]<2.:
            idx=allowed[k];taken.add(idx);pairs.append((idx,int(j)))
    return pairs
def summary(infos,predictions):
    out={}
    for threshold in [0.,.05,.1,.2,.3,.5]:
        n=hit=preds=agnostic=roi_n=roi_hit=0;per={k:dict(gt=0,tp=0) for k in NAMES}
        for info in infos:
            p=predictions[info['token']];pairs=match(info,p,threshold);ids={i for i,j in pairs};roi=inside(info['gt_boxes'])
            n+=len(info['gt_boxes']);hit+=len(ids);preds+=sum(s>=threshold for s in p['scores']);agnostic+=len(match(info,p,threshold,False));roi_n+=int(roi.sum());roi_hit+=sum(bool(roi[i]) for i in ids)
            for i,name in enumerate(info['gt_names']):per[str(name)]['gt']+=1;per[str(name)]['tp']+=int(i in ids)
        out[str(threshold)]={'gt':n,'tp':hit,'predictions':preds,'recall':hit/max(n,1),'precision':hit/max(preds,1),'class_agnostic_tp':agnostic,'inside_roi_gt':roi_n,'inside_roi_tp':roi_hit,'inside_roi_recall':roi_hit/max(roi_n,1),'per_class':per}
    return out
def normalize(rows):
    return {r['token']:dict(boxes=r.get('all_predicted_boxes',r.get('predicted_boxes')),scores=r.get('all_scores',r.get('scores')),names=r.get('all_names',r.get('predicted_names'))) for r in rows}
def latest(prefix):
    paths=[p for p in (ROOT/'experiments/visualization').glob(prefix+'*/manifest.json')]
    return max(paths,key=lambda p:p.stat().st_mtime)

train,val,test=[read_infos(k) for k in ('train','val','test')]
original=json.loads((ROOT/'experiments/visualization/transfusion_l_20261028_105558/manifest.json').read_text())
valby={i['token']:i for i in val};selected=[valby[r['token']] for r in original['samples']]
tfpath=latest('transfusion_l_');mcpath=latest('mcdetra_')
tfmeta=json.loads(tfpath.read_text());mcmeta=json.loads(mcpath.read_text())
assert [x['token'] for x in tfmeta['samples']]==[x['token'] for x in original['samples']]==[x['token'] for x in mcmeta['samples']]
tfpred=normalize(tfmeta['samples']);mcpred=normalize(mcmeta['samples'])
report={'definitions':{'matching':'confidence-ordered same-class xy distance <2 m; recall diagnostics, not AP','selected':'the fixed 10 crowded frames, not a random validation sample','training_probe':'32 random train and 32 random validation frames, seed20261028; limited diagnostic only'},'visualization':{'transfusion_manifest':str(tfpath),'mcdetra_manifest':str(mcpath),'same_order':True,'object_text':False,'threshold':.1},'selected_transfusion':summary(selected,tfpred),'selected_mcdetra':summary(selected,mcpred)}
with (ROOT/'experiments/baselines/infrauni/transfusion_sequence/resolve_sequence_20261027/eval/eval_with_train/epoch_20/val/result.pkl').open('rb') as f:full=pickle.load(f)
fullpred={a['metadata']['token']:dict(boxes=a['boxes_lidar'].tolist(),scores=a['score'].tolist(),names=a['name'].tolist()) for a in full}
assert set(fullpred)==set(valby)
report['full_validation']=summary(val,fullpred)
matched_scores={n:[] for n in NAMES}
for info in val:
    p=fullpred[info['token']]
    for gi,pi in match(info,p,0.):matched_scores[str(info['gt_names'][gi])].append(p['scores'][pi])
report['matched_score_quantiles']={n:{'matched':len(a),'p05_p25_median_p75_p95_max':np.quantile(a,[.05,.25,.5,.75,.95,1.]).tolist()} for n,a in matched_scores.items() if a}

diff=0.
for info in selected:
    a=np.asarray(tfpred[info['token']]['boxes']);b=np.asarray(fullpred[info['token']]['boxes'])
    assert a.shape==b.shape
    diff=max(diff,float(np.max(np.abs(a-b))) if a.size else 0.)
report['raw_indexwise_box_difference_not_a_set_comparison']=diff
(OUT/'partial.json').write_text(json.dumps(report,indent=2))
report['split']={};sets=[];bags=[];raw=[]
for part,infos in [('train',train),('val',val),('test',test)]:
    seq={Path(i['cache_path']).parent.name for i in infos};sets.append(seq)
    bags.append({str(i['token']).rsplit('_',1)[0] for i in infos});raw.append({i['lidar_path'] for i in infos})
    counts={n:sum(sum(str(v)==n for v in i['gt_names']) for i in infos) for n in NAMES}
    report['split'][part]={'frames':len(infos),'sequences':len(seq),'classes':counts,'gt_outside_roi':sum(int((~inside(i['gt_boxes'])).sum()) for i in infos),'gt':sum(len(i['gt_boxes']) for i in infos),'gt_with_1_to_5_lidar_points':sum(int(((i['num_lidar_pts']>=1)&(i['num_lidar_pts']<=5)).sum()) for i in infos)}
report['split']['overlaps']={str((a,b)):{'sequence':len(sets[a]&sets[b]),'rosbag':len(bags[a]&bags[b]),'raw_lidar':len(raw[a]&raw[b])} for a,b in [(0,1),(0,2),(1,2)]}

# Verify corner convention, projection mathematics and the actual calibration.
cammanifest=json.loads((ROOT/'data/resolve_cache/cameras_manifest.json').read_text())['samples'];cammap={(r['rosbag'],r['frame_index']):r for r in cammanifest if r['split']=='val'}
sync=[];maxcorner=maxcv=maxcalib=0.;corner_rejected_visible=0
for row,info in zip(original['samples'],selected):
    item=cammap[(row['rosbag'],row['frame_index'])]
    pred=tfpred[info['token']];boxes=np.asarray(pred['boxes']);keep=np.asarray(pred['scores'])>=.1;boxes=boxes[keep,:7]
    maxcorner=max(maxcorner,float(np.max(np.abs(boxes_to_corners_3d(boxes)-np.asarray(official_corners(boxes))))) if len(boxes) else 0.)
    with np.load(ROOT/'data/resolve_cache'/item['camera_npz']) as c:
        for j,cam in enumerate(c['camera_names']):
            k,d,t=parse_calibration(Path(str(c['source_calibration'][j])))
            maxcalib=max(maxcalib,float(np.max(np.abs(k-c['K'][j]))),float(np.max(np.abs(t-c['lidar2cam'][j]))))
            pts=info['gt_boxes'][:,:3].astype(np.float64);uv,z=project_points(pts,t,k,d);camera_points=pts@t[:3,:3].T+t[:3,3];uv2,_=cv2.projectPoints(camera_points,np.zeros(3),np.zeros(3),k,d);valid=z>0
            if valid.any():maxcv=max(maxcv,float(np.max(np.abs(uv[valid]-uv2[:,0][valid]))))
            src=Path(str(c['source_image'][j]));ref=str(c['json_image_stem'][j]);time=lambda s:int(s.split('-')[0])+int(s.split('-')[1])*1e-9
            sync.append({'sample':row['order'],'camera':str(cam),'delta_from_annotation_image_seconds':time(src.stem)-time(ref),'same_annotation_image':src.stem==ref,'annotation_image_available':src.with_name(ref+'.jpg').exists(),'source_image':str(src),'annotation_image_stem':ref})
            h,w=c['image_hw'][j]
            for corners in boxes_to_corners_3d(boxes):
                v,z=project_points(corners,t,k,d)
                inside_image=(z>0)&(v[:,0]>=0)&(v[:,0]<w)&(v[:,1]>=0)&(v[:,1]<h)
                if inside_image.any():continue
                for a,b in BOX_EDGES:
                    if z[a]<=0 or z[b]<=0:continue
                    p1=tuple(np.clip(np.round(v[a]),-1e8,1e8).astype(int));p2=tuple(np.clip(np.round(v[b]),-1e8,1e8).astype(int))
                    if cv2.clipLine((0,0,int(w),int(h)),p1,p2)[0]:corner_rejected_visible+=1;break
report['projection']={'max_corner_difference_vs_openpcdet_m':maxcorner,'max_pixel_difference_vs_opencv':maxcv,'max_calibration_difference':maxcalib,'boxes_with_visible_edges_but_no_visible_corner':corner_rejected_visible,'image_pairing':sync}

# Epoch snapshots and independent train/validation probes, without optimization.
os.chdir(ROOT/'third_party/RESOLVE/tools');cfg_from_yaml_file(str(ROOT/'configs/resolve/transfusion_sequence.yaml'),cfg)
logger=logging.getLogger('resolve-diagnostic');logger.setLevel(logging.INFO);logger.addHandler(logging.StreamHandler(sys.stdout))
dataset=ResolveSequenceDataset(cfg.DATA_CONFIG,cfg.CLASS_NAMES,training=False,logger=logger)
model=build_network(model_cfg=cfg.MODEL,num_class=len(cfg.CLASS_NAMES),dataset=dataset).cuda().eval()
def infer(infos):
    dataset.infos=infos;out={}
    for j,info in enumerate(infos):
        with torch.no_grad():
            b=dataset.collate_batch([dataset[j]]);load_data_to_gpu(b);pred,_=model(b)
        p=pred[0];out[info['token']]=dict(boxes=p['pred_boxes'].cpu().numpy().tolist(),scores=p['pred_scores'].cpu().numpy().tolist(),names=[cfg.CLASS_NAMES[int(v)-1] for v in p['pred_labels'].cpu().numpy()])
    return out
report['epochs_selected']={}
for epoch in [5,10,15,20]:
    cp=ROOT/'experiments/baselines/infrauni/transfusion_sequence/resolve_sequence_20261027/ckpt'/('checkpoint_epoch_%d.pth'%epoch)
    model.load_params_from_file(str(cp),logger=logger,to_cpu=True);p=infer(selected)
    report['epochs_selected'][str(epoch)]=summary(selected,p)
    print('EPOCH',epoch,json.dumps(report['epochs_selected'][str(epoch)]['0.1']),flush=True)
rng=np.random.default_rng(20261028)
for part,infos in [('train',train),('val',val)]:
    subset=[infos[int(i)] for i in rng.choice(len(infos),32,replace=False)]
    report['probe_'+part]=summary(subset,infer(subset))
    print('PROBE',part,json.dumps(report['probe_'+part]['0.1']),flush=True)
log=(ROOT/'experiments/resolve_transfusion_sequence_20261027/console.log').read_text()
lines={}
for line in log.splitlines():
    m=re.search(r'Train:\s+(\d+)/20.*Loss:\s+([\d.]+)\s+\(([\d.e+\-]+)\)',line)
    if m:lines[m.group(1)]={'last_logged_loss':float(m.group(2)),'running_epoch_loss':float(m.group(3))}
report['training_epoch_loss']=lines
(OUT/'report.json').write_text(json.dumps(report,indent=2))
print('REPORT',OUT/'report.json',flush=True)

