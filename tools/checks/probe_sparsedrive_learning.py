"""Inspect a fixed image-only subset: matched box errors and temporal drift."""
import sys,importlib.util,json,pickle
from pathlib import Path
from collections import defaultdict
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
spec=importlib.util.spec_from_file_location('_train_probe',ROOT/'tools/train/train_sparsedrive_resolve.py')
t=importlib.util.module_from_spec(spec);spec.loader.exec_module(t)
cfg,model=t.build_model(ROOT/'configs/resolve/sparsedrive_camera.py',1)
run=ROOT/'experiments/sparsedrive_camera_resolve_20261028'
state=torch.load(run/'diagnostics/loss_audit_epoch4/checkpoint_epoch004.pth',map_location='cpu');model.load_state_dict(state['state_dict']);model.eval()
dataset=t.ResolveCameraDataset(ROOT/'data/resolve_sparsedrive/train.pkl',cfg.input_size,False)
groups=defaultdict(list)
for i,d in enumerate(dataset.infos):groups[d['rosbag']].append(i)
names=['rosbag2_2025_12_21-15_53_23','rosbag2_2026_02_02-12_48_12']
indices=[]
for name in names:indices+=groups[name][:80]
assert len(indices)>=100
captured={}
def hook(module,args,result):captured['det']=result[0]
handle=model.head.register_forward_hook(hook)
report={}
for mode in ['temporal','single_frame']:
    model.reset_memory();totals=[];rows=[]
    for k,index in enumerate(indices):
        if mode=='single_frame':model.reset_memory()
        batch=t.to_device(t.collate([dataset[index]]));d=dataset.infos[index]
        with torch.no_grad():
            result=model(**batch);det=captured['det'];head=model.head.det_head
            target_labels=[torch.tensor(d['gt_labels'],device='cuda')];target_boxes=[torch.tensor(d['gt_boxes'],device='cuda')]
            weighted=torch.zeros(10,device='cuda');entropy_sum=0.;cns_sum=0.;final={}
            for layer,(cls,raw) in enumerate(zip(det['classification'],det['prediction'])):
                raw=raw[...,:10]
                ct,rt,rw=head.sampler.sample(cls,raw,target_labels,target_boxes)
                positive=(rt!=0).any(-1);num=positive.sum().clamp_min(1)
                mask=positive&(cls.max(-1).values.sigmoid()>.05)
                errors=(raw-rt).abs();weights=rw*raw.new_tensor(head.reg_weights)
                term=(errors*weights)[mask].sum(0)/num*.25
                weighted+=term
                target=torch.exp(-torch.linalg.vector_norm((raw-rt)[...,:3],dim=-1))[mask].clamp(1e-7,1-1e-7)
                entropy_sum+=float((-(target*target.log()+(1-target)*(1-target).log())).sum()/num)
                quality=det['quality'][layer][...,0][mask]
                cns_sum+=float(torch.nn.functional.binary_cross_entropy_with_logits(quality,target,reduction='sum')/num)
                if layer==5:
                    unmatched=~positive;score=cls.sigmoid().max(-1).values
                    final=dict(matched=int(positive.sum()),regressed=int(mask.sum()),matched_abs_error_mean=errors[positive].mean(0).cpu().tolist() if positive.any() else [0.]*10,all_log_size_max=float(raw[...,3:6].max()),unmatched_log_size_max=float(raw[...,3:6][unmatched].max()) if unmatched.any() else None,positive_log_size_max=float(raw[...,3:6][positive].max()) if positive.any() else None,score_ge_01_bad_size_count=int(((score>=.1)&((raw[...,3:6].abs()>10).any(-1))).sum()))
            totals.append(weighted.cpu().numpy());rows.append(dict(token=d['token'],box_loss_components=weighted.cpu().tolist(),centerness_loss=cns_sum,centerness_entropy_floor=entropy_sum,**final))
        if (k+1)%40==0:print('PROBE',mode,k+1,'/',len(indices),flush=True)
    a=np.mean(totals,axis=0)
    report[mode]=dict(frames=len(rows),weighted_box_components=a.tolist(),groups=dict(xyz=float(a[:3].sum()),size=float(a[3:6].sum()),yaw=float(a[6:8].sum()),velocity=float(a[8:10].sum())),centerness_loss_mean=float(np.mean([r['centerness_loss'] for r in rows])),centerness_entropy_floor_mean=float(np.mean([r['centerness_entropy_floor'] for r in rows])),rows=rows)
    print('PROBE_SUMMARY',mode,report[mode]['groups'],flush=True)
handle.remove()
out=run/'diagnostics/loss_audit_epoch4/component_probe.json';out.write_text(json.dumps(report,indent=2))
print('PROBE_COMPLETED',out,flush=True)
