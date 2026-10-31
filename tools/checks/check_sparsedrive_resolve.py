"""Actual data, forward/backward and inference-protocol integration check."""
import sys,json,pickle,logging
from pathlib import Path
import numpy as np
import torch
logging.disable(logging.INFO)
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from infrauni.resolve.sparsedrive.runtime import setup
setup()
from mmcv import Config
from mmdet.models import build_detector
import projects.mmdet3d_plugin
from infrauni.resolve.sparsedrive import models
from infrauni.resolve.sparsedrive.data import ResolveCameraDataset,collate,to_device,SequenceBatchSampler
from infrauni.resolve.sparsedrive.evaluation import normalize

cfg=Config.fromfile(str(ROOT/'configs/resolve/sparsedrive_camera.py'))
dataset=ResolveCameraDataset(ROOT/'data/resolve_sparsedrive/train.pkl',cfg.input_size,True)
valid=next(j for j,i in enumerate(dataset.infos) if len(i['gt_boxes']) and i['gt_future_mask'].all(axis=1).any())
sample=dataset[valid]
assert sample['img'].shape==(4,3,384,704)
assert torch.isfinite(sample['projection_mat']).all()
target=sample['gt_agent_fut_trajs'].cumsum(1).numpy()+dataset.infos[valid]['gt_boxes'][:,None,:2]
mask=dataset.infos[valid]['gt_future_mask'];assert np.allclose(target[mask],dataset.infos[valid]['gt_future'][mask],atol=1e-4)
indices=[i for batch in SequenceBatchSampler(dataset.infos,2,cfg.seed) for i in batch]
assert len(indices)==len(dataset) and len(set(indices))==len(dataset)
report={'full_train_sampler_no_drop':True,'future_target_roundtrip':True,'input_is_camera_only':True,'stages':{}}
for stage in [1,2]:
    config=Config.fromfile(str(ROOT/'configs/resolve/sparsedrive_camera.py'))
    if stage==1:config.model.head.task_config.with_motion_plan=False
    model=build_detector(config.model);model.init_weights();model.cuda().train();model.reset_memory()
    model.head.collect_training_diagnostics=True
    batch=to_device(collate([dataset[valid],dataset[valid+1]]))
    with torch.cuda.amp.autocast(enabled=False):losses=model(**batch);loss=sum(v.mean() for k,v in losses.items() if 'loss' in k)
    assert torch.isfinite(loss),losses
    diagnostics=model.head.last_training_diagnostics
    assert diagnostics['matched_gt']>0 and diagnostics['matched_log_size_max']<10
    assert np.isfinite(diagnostics['cns_last_decoder_entropy_floor'])
    loss.backward()
    bad=[n for n,p in model.named_parameters() if p.grad is not None and not torch.isfinite(p.grad).all()]
    assert not bad,(bad[:10],{k:float(v.detach()) for k,v in losses.items()})
    # A temporal cache can retain unconstrained, unmatched proposals. Recover
    # corrupted proposals without losing frame coverage or instance-ID shape.
    model.zero_grad(set_to_none=True);model.numeric_watch={}
    bank=model.head.det_head.instance_bank
    with torch.no_grad():
        bank.cached_anchor[:,0,3]=100.
        bank.cached_anchor[:,1,0]=float('nan')
    later=dict(batch);later['timestamp']=batch['timestamp']+.1
    later['img_metas']=[dict(m,timestamp=float(later['timestamp'][i])) for i,m in enumerate(batch['img_metas'])]
    recovered=model(**later);sum(v for k,v in recovered.items() if 'loss' in k).backward()
    assert model.numeric_watch['invalid_cached_proposals_reset']==4
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    assert torch.isfinite(bank.cached_anchor).all()
    if stage==2:
        model.zero_grad(set_to_none=True);model.reset_memory()
        with torch.cuda.amp.autocast(enabled=False):losses=model(**batch);motion=sum(v for k,v in losses.items() if k.startswith('motion_loss'))
        motion.backward();assert model.img_backbone.conv1.weight.grad is not None and model.img_backbone.conv1.weight.grad.abs().sum()>0
        assert model.head.motion_plan_head.layers[-1].motion_reg_branch[-1].weight.grad.abs().sum()>0
    model.eval();model.reset_memory();evaldata=ResolveCameraDataset(ROOT/'data/resolve_sparsedrive/val.pkl',cfg.input_size,False);test=to_device(collate([evaldata[0]]));assert not any(k.startswith('gt_') for k in test)
    with torch.no_grad(),torch.cuda.amp.autocast(enabled=False):result=model(**test)[0]
    normalized=normalize(result);assert len(normalized)==300
    if stage==2:assert np.asarray(normalized[0]['future_xy']).shape==(6,10,2)
    report['stages'][str(stage)]={'finite_loss':float(loss.detach()),'all_gradients_finite':True,'invalid_temporal_cache_recovery':True,'inference_without_gt':True,'decoded_queries':len(normalized),'peak_gpu_GB':torch.cuda.max_memory_allocated()/1e9,'loss_names':list(losses)}
    print('PASS_STAGE',stage,report['stages'][str(stage)],flush=True)
    del model,batch,losses,loss,result;torch.cuda.empty_cache()
(ROOT/'data/resolve_sparsedrive/validation.json').write_text(json.dumps(report,indent=2));print('ALL_INTEGRATION_CHECKS_PASSED',flush=True)
