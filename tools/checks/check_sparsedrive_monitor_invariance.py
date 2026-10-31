"""Verify monitoring preserves the real two-stage objective and RNG stream."""
import importlib.util,sys,random,json
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
spec=importlib.util.spec_from_file_location('_monitor_check',ROOT/'tools/train/train_sparsedrive_resolve.py')
t=importlib.util.module_from_spec(spec);spec.loader.exec_module(t)
data=t.ResolveCameraDataset(ROOT/'data/resolve_sparsedrive/train.pkl',(704,384),True)
i=next(i for i,d in enumerate(data.infos) if len(d['gt_boxes']) and d['gt_future_mask'].all(1).any())
batch=t.to_device(t.collate([data[i],data[i+1]]));report={}
for stage in [1,2]:
    cfg,model=t.build_model(ROOT/'configs/resolve/sparsedrive_camera.py',stage);model.train()
    rng=(torch.get_rng_state(),torch.cuda.get_rng_state_all(),np.random.get_state(),random.getstate())
    buffers={n:v.clone() for n,v in model.named_buffers()};values=[];after=[]
    for enabled in [False,True]:
        model.reset_memory();model.zero_grad(set_to_none=True)
        for n,v in model.named_buffers():v.copy_(buffers[n])
        torch.set_rng_state(rng[0]);torch.cuda.set_rng_state_all(rng[1]);np.random.set_state(rng[2]);random.setstate(rng[3])
        model.head.collect_training_diagnostics=enabled
        losses=model(**batch);loss=sum(v.mean() for k,v in losses.items() if 'loss' in k);loss.backward()
        values.append({k:float(v.detach()) for k,v in losses.items()})
        after.append((torch.get_rng_state(),torch.cuda.get_rng_state_all(),np.random.get_state(),random.getstate()))
        assert abs(sum(t.loss_groups(values[-1]).values())-float(loss.detach()))<1e-4
    assert values[0].keys()==values[1].keys()
    delta=max(abs(values[0][k]-values[1][k]) for k in values[0])
    assert all(np.isclose(values[0][k],values[1][k],atol=1e-4,rtol=1e-5) for k in values[0]),delta
    assert torch.equal(after[0][0],after[1][0]) and all(torch.equal(a,b) for a,b in zip(after[0][1],after[1][1]))
    assert np.array_equal(after[0][2][1],after[1][2][1]) and after[0][2][2:]==after[1][2][2:] and after[0][3]==after[1][3]
    report[str(stage)]=dict(loss_keys_identical=True,max_absolute_loss_difference=delta,rng_identical=True,exclusive_loss_groups_sum_correct=True)
    print('MONITOR_INVARIANCE_STAGE',stage,report[str(stage)],flush=True)
    del model,losses,loss;torch.cuda.empty_cache()
out=ROOT/'experiments/sparsedrive_camera_resolve_20261028/diagnostics/loss_audit_epoch4/monitor_invariance.json';out.write_text(json.dumps(report,indent=2))
print('MONITOR_INVARIANCE_PASSED',flush=True)
