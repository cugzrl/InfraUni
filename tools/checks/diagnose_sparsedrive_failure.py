"""Replay a captured bad backward pass, optionally isolating one loss group."""
import argparse
import importlib.util
import random
from pathlib import Path
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[2]
spec=importlib.util.spec_from_file_location('_resolve_train',ROOT/'tools/train/train_sparsedrive_resolve.py')
train=importlib.util.module_from_spec(spec)
spec.loader.exec_module(train)

def device(value):
    if torch.is_tensor(value):return value.cuda()
    if isinstance(value,dict):return {k:device(v) for k,v in value.items()}
    if isinstance(value,list):return [device(v) for v in value]
    if isinstance(value,tuple):return tuple(device(v) for v in value)
    return value

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('snapshot',type=Path)
    parser.add_argument('--loss',default='loss')
    parser.add_argument('--mode',choices=['legacy','projection-only','fixed'],default='fixed')
    args=parser.parse_args()
    saved=torch.load(args.snapshot,map_location='cpu')
    cfg,model=train.build_model(ROOT/'configs/resolve/sparsedrive_camera.py',saved['stage'])
    model.load_state_dict(saved['state_dict']);model.train();model.reset_memory()
    if args.mode!='fixed':
        # Reproduce the previous runtime on the saved within-sequence frame;
        # the projection-only control changes just its unsafe division.
        assert saved['stage']==1
        def original_forward(img,**data):return super(train.models.ResolveSparseDrive,model).forward(img,**data)
        model.forward=original_forward
        for layer in model.head.det_head.layers:
            if isinstance(layer,train.models.DeformableFeatureAggregation):
                layer.kps_generator.__class__=train.models.SparseBox3DKeyPointsGenerator
                if args.mode=='legacy':layer.__class__=train.models.DeformableFeatureAggregation
    for key,value in saved['bank'].items():setattr(model.head.det_head.instance_bank,key,device(value))
    model._resolve_scenes=saved['scenes'];model._resolve_times=saved['times']
    torch.set_rng_state(saved['rng']['torch']);torch.cuda.set_rng_state_all(saved['rng']['cuda'])
    np.random.set_state(saved['rng']['numpy']);random.setstate(saved['rng']['python'])
    with torch.autograd.detect_anomaly():
        losses=model(**device(saved['batch']))
        print('REPLAY_LOSSES',{k:float(v.detach()) for k,v in losses.items()},flush=True)
        chosen=[v.mean() for k,v in losses.items() if args.loss in k]
        assert chosen,args.loss
        sum(chosen).backward()
    bad=[n for n,p in model.named_parameters() if p.grad is not None and not torch.isfinite(p.grad).all()]
    print('NONFINITE_PARAMETERS',bad,flush=True)
    assert not bad
    print('REPLAY_FINITE_BACKWARD_PASSED',args.mode,flush=True)

if __name__=='__main__':main()
