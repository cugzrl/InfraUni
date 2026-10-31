"""Single-GPU, two-stage SparseDrive training using the shared RESOLVE split."""
import os,sys,json,pickle,math,time,random,argparse,logging
from contextlib import nullcontext
from collections import deque
from pathlib import Path
import numpy as np
import torch
logging.disable(logging.INFO)
from torch.utils.data import DataLoader

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from infrauni.resolve.sparsedrive.runtime import setup
setup()
from mmcv import Config
from mmdet.models import build_detector
import projects.mmdet3d_plugin
from infrauni.resolve.sparsedrive import models
from infrauni.resolve.sparsedrive.data import ResolveCameraDataset,SequenceBatchSampler,collate,to_device
from infrauni.resolve.sparsedrive.evaluation import normalize,evaluate

def build_model(config,stage):
    cfg=Config.fromfile(str(config))
    if stage==1:cfg.model.head.task_config.with_motion_plan=False
    model=build_detector(cfg.model)
    model.init_weights()
    watch={};model.numeric_watch=watch
    def watch_keypoints(module,inputs):
        anchor=inputs[0].detach()
        size=anchor[...,3:6]
        lo,hi=float(size.min()),float(size.max())
        watch['log_size_min']=min(watch.get('log_size_min',lo),lo)
        watch['log_size_max']=max(watch.get('log_size_max',hi),hi)
        if not torch.isfinite(anchor).all():
            print('KEYPOINT_ANCHOR_NUMERIC_FAILURE',lo,hi,flush=True)
            raise FloatingPointError('Invalid or overflowing keypoint anchor geometry')
    for module in model.modules():
        if isinstance(module,models.SparseBox3DKeyPointsGenerator):module.register_forward_pre_hook(watch_keypoints)
    return cfg,model.cuda()

def checkpoint(path,state):
    temp=path.with_suffix('.tmp');torch.save(state,str(temp));os.replace(str(temp),str(path))


def loss_groups(values):
    groups=dict(cls=0.,box=0.,cns=0.,yns=0.,motion_loss=0.,other=0.)
    for key,value in values.items():
        if 'loss' not in key:continue
        kind=next((name for name in ['cls','box','cns','yns'] if key.startswith('det_loss_'+name+'_')),None)
        if kind is None:kind='motion_loss' if key.startswith('motion_loss') else 'other'
        groups[kind]+=value
    return groups

def validate(model,dataset,motion,out,prediction_cache=None,cache_tokens=None):
    was_training=model.training;model.eval();model.reset_memory();predictions={}
    loader=DataLoader(dataset,batch_size=1,shuffle=False,num_workers=2,collate_fn=collate,pin_memory=True)
    with torch.no_grad():
        for j,batch in enumerate(loader):
            token=batch['img_metas'][0]['token'];batch=to_device(batch)
            # No GT, track IDs or future targets are present at inference.
            assert not any(k.startswith('gt_') for k in batch)
            with torch.cuda.amp.autocast(enabled=False):result=model(**batch)
            predictions[token]=normalize(result[0])
            if (j+1)%500==0:print('VALIDATION',j+1,'/',len(dataset),flush=True)
    if prediction_cache is not None:
        saved=predictions if cache_tokens is None else {token:predictions[token] for token in cache_tokens}
        prediction_cache.parent.mkdir(parents=True,exist_ok=True)
        with prediction_cache.open('wb') as f:pickle.dump(saved,f,protocol=4)
        print('PREDICTION_CACHE',len(saved),prediction_cache,flush=True)
    result=evaluate(dataset.infos,predictions,motion)
    out.write_text(json.dumps(result,indent=2));model.reset_memory();model.train(was_training)
    return result

def train_stage(args,stage,initial=None):
    cfg,model=build_model(args.config,stage)
    train=ResolveCameraDataset(ROOT/'data/resolve_sparsedrive/train.pkl',cfg.input_size,True)
    val=ResolveCameraDataset(ROOT/'data/resolve_sparsedrive/val.pkl',cfg.input_size,False)
    sampler=SequenceBatchSampler(train.infos,cfg.batch_size,cfg.seed)
    loader=DataLoader(train,batch_sampler=sampler,num_workers=args.workers,collate_fn=collate,pin_memory=True)
    epochs=cfg.stage1_epochs if stage==1 else cfg.stage2_epochs
    folder=args.work_dir/('stage%d'%stage);folder.mkdir(parents=True,exist_ok=True)
    groups=[{'params':[p for n,p in model.named_parameters() if p.requires_grad and not n.startswith('img_backbone.')],'lr':cfg.learning_rate},{'params':[p for n,p in model.named_parameters() if p.requires_grad and n.startswith('img_backbone.')],'lr':cfg.learning_rate*.1}]
    optimizer=torch.optim.AdamW(groups,weight_decay=cfg.weight_decay)
    scaler=torch.cuda.amp.GradScaler(init_scale=32.,enabled=False)
    start=updates=resume_batch=0;best=-float('inf');latest=folder/'latest.pth';resumed_stats=None
    if latest.is_file():
        state=torch.load(str(latest),map_location='cpu');model.load_state_dict(state['state_dict'],strict=True);optimizer.load_state_dict(state['optimizer']);scaler.load_state_dict(state['scaler']);start=state['epoch'];updates=state['updates'];best=state.get('best',best)
        if 'rng_torch' in state:torch.set_rng_state(state['rng_torch']);np.random.set_state(state['rng_numpy']);random.setstate(state['rng_python'])
        if 'rng_cuda' in state:torch.cuda.set_rng_state_all(state['rng_cuda'])
        resume_batch=state.get('next_batch',0)
        resumed_stats=state.get('train_stats')
        print('RESUME',stage,'epoch',start,'next_batch',resume_batch,'updates',updates,flush=True)
    elif initial is not None:
        state=torch.load(str(initial),map_location='cpu');missing,extra=model.load_state_dict(state['state_dict'],strict=False)
        assert not extra
        assert all(n.startswith('head.motion_plan_head.') for n in missing),missing
        print('STAGE2_INITIALIZATION',str(initial),'new_motion_tensors',len(missing),flush=True)
    total_updates=0
    for planned_epoch in range(epochs):
        sampler.set_epoch(planned_epoch);total_updates+=math.ceil(len(sampler)/cfg.accumulation_steps)
    started=time.time();grad=None;coefficient=None;train_stats={};window=deque(maxlen=100);model.train();model.reset_memory()
    def snapshot(path,completed,next_batch=0):
        train_stats['window']=list(window)
        state=dict(stage=stage,epoch=completed,next_batch=next_batch,updates=updates,train_stats=train_stats,state_dict=model.state_dict(),optimizer=optimizer.state_dict(),scaler=scaler.state_dict(),best=best,config=cfg.to_dict(),rng_torch=torch.get_rng_state(),rng_cuda=torch.cuda.get_rng_state_all(),rng_numpy=np.random.get_state(),rng_python=random.getstate(),protocol_manifest=json.loads((ROOT/'data/resolve_sparsedrive/manifest.json').read_text()))
        checkpoint(path,state)
        link_temp=folder/'latest.linktmp'
        if link_temp.is_symlink() or link_temp.exists():link_temp.unlink()
        link_temp.symlink_to(path.name);os.replace(str(link_temp),str(latest))
    for epoch in range(start,epochs):
        sampler.set_epoch(epoch);sampler.start_batch=resume_batch if epoch==start else 0
        model.reset_memory();optimizer.zero_grad(set_to_none=True);num_batches=len(loader)+sampler.start_batch
        if epoch==start and resumed_stats and resumed_stats.get('epoch_index')==epoch:
            train_stats=resumed_stats
        else:train_stats=dict(epoch_index=epoch,first_batch=sampler.start_batch+1,batches=0,samples=0,loss_sum=0.,component_sums={})
        window=deque(train_stats.get('window',[]),maxlen=100)
        for j,batch in enumerate(loader,start=sampler.start_batch):
            batch=to_device(batch)
            model.numeric_watch.clear()
            log_now=(j==sampler.start_batch or (j+1)%20==0)
            model.head.collect_training_diagnostics=log_now
            bank=model.head.det_head.instance_bank
            memory_before={k:getattr(bank,k) for k in ['cached_feature','cached_anchor','metas','mask','confidence','temp_confidence','instance_id','prev_id']}
            rng_before=dict(torch=torch.get_rng_state(),cuda=torch.cuda.get_rng_state_all(),numpy=np.random.get_state(),python=random.getstate())
            scenes_before=getattr(model,'_resolve_scenes',None);times_before=getattr(model,'_resolve_times',None)
            def dump_failure(reason):
                checkpoint(folder/'diagnostic_failure.pth',dict(reason=reason,state_dict=model.state_dict(),batch=batch,bank=memory_before,scenes=scenes_before,times=times_before,rng=rng_before,stage=stage,epoch=epoch+1,batch_index=j,updates=updates,losses={k:float(v.detach()) for k,v in losses.items()},numeric_watch=model.numeric_watch))
            with (torch.autograd.detect_anomaly() if args.diagnose_after_batch>=0 and j>=args.diagnose_after_batch else nullcontext()),torch.cuda.amp.autocast(enabled=False):
                losses=model(**batch);terms=[v.mean() for k,v in losses.items() if 'loss' in k];loss=sum(terms)
                if not torch.isfinite(loss):raise FloatingPointError('Non-finite loss '+str({k:float(v.detach()) for k,v in losses.items()}))
                accumulation=min(cfg.accumulation_steps,num_batches-(j//cfg.accumulation_steps)*cfg.accumulation_steps)
                try:scaler.scale(loss/accumulation).backward()
                except RuntimeError:
                    print('BACKWARD_FAILURE',stage,epoch+1,j+1,[m['token'] for m in batch['img_metas']],{k:float(v.detach()) for k,v in losses.items()},flush=True)
                    dump_failure('backward exception')
                    raise
            values={k:float(v.detach().mean()) for k,v in losses.items()}
            loss_value=float(loss.detach());window.append(loss_value)
            train_stats['loss_sum']+=loss_value;train_stats['batches']+=1;train_stats['samples']+=len(batch['img_metas'])
            for key,value in values.items():train_stats['component_sums'][key]=train_stats['component_sums'].get(key,0.)+value
            if (j+1)%cfg.accumulation_steps==0 or j+1==num_batches:
                scaler.unscale_(optimizer)
                bad=[n for n,p in model.named_parameters() if p.grad is not None and not torch.isfinite(p.grad).all()]
                if bad:
                    print('BAD_GRADIENT',stage,epoch+1,j+1,bad[:30],[m['token'] for m in batch['img_metas']],flush=True)
                    dump_failure('non-finite gradient elements: '+str(bad[:30]))
                    raise FloatingPointError('Non-finite gradient elements')
                # Accumulate norms in float64 so finite large FP32 gradients do
                # not overflow merely while computing their clipping norm.
                norm=torch.stack([p.grad.detach().double().norm() for p in model.parameters() if p.grad is not None]).norm()
                grad=float(norm)
                fp32_norm=torch.stack([p.grad.detach().norm() for p in model.parameters() if p.grad is not None]).norm()
                if not torch.isfinite(fp32_norm):
                    print('FP32_CLIP_NORM_OVERFLOW',stage,epoch+1,j+1,'finite_element_norm64',grad,flush=True)
                if not math.isfinite(grad):raise FloatingPointError('Non-finite gradient norm')
                coefficient=min(1.,cfg.grad_clip/(grad+1e-6))
                for p in model.parameters():
                    if p.grad is not None:p.grad.mul_(coefficient)
                progress=min(updates/max(total_updates,1),1.)
                factor=(1/3+(2/3)*updates/cfg.warmup_steps) if updates<cfg.warmup_steps else .001+.999*.5*(1+math.cos(math.pi*progress))
                optimizer.param_groups[0]['lr']=cfg.learning_rate*factor;optimizer.param_groups[1]['lr']=cfg.learning_rate*.1*factor
                scaler.step(optimizer);scaler.update();optimizer.zero_grad(set_to_none=True);updates+=1
                if args.save_every_updates and updates%args.save_every_updates==0:
                    snapshot(folder/'recovery.pth',epoch,j+1)
                    print('RECOVERY_CHECKPOINT',stage,epoch+1,j+1,updates,flush=True)
            if log_now:
                mean=train_stats['loss_sum']/train_stats['batches'];rolling=sum(window)/len(window)
                print('TRAIN stage=%d epoch=%d/%d batch=%d/%d update=%d loss=%.5f epoch_avg=%.5f rolling100=%.5f stats_first_batch=%d lr=%.7f elapsed_min=%.1f'%(stage,epoch+1,epochs,j+1,num_batches,updates,loss_value,mean,rolling,train_stats['first_batch'],optimizer.param_groups[0]['lr'],(time.time()-started)/60),flush=True)
                diagnostics=model.head.last_training_diagnostics
                with (folder/'train.jsonl').open('a') as f:f.write(json.dumps(dict(epoch=epoch+1,batch=j+1,update=updates,loss=loss_value,epoch_avg=mean,rolling100=rolling,stats_first_batch=train_stats['first_batch'],grad_norm_before_clip=grad,clip_scale=coefficient,losses=values,matched_boxes=diagnostics))+'\n')
                print('LOSS_PARTS',json.dumps({k:round(v,5) for k,v in loss_groups(values).items()}),'tokens',[m['token'] for m in batch['img_metas']],flush=True)
                print('GRAD_CLIP','norm_before',grad,'max_norm',cfg.grad_clip,'scale',coefficient,flush=True)
                print('MATCHED_BOXES',json.dumps(diagnostics),flush=True)
                print('ALL_PROPOSALS_RAW_GEOMETRY',model.numeric_watch,'includes_low_score_unmatched_queries=True',flush=True)
        means={k:v/train_stats['batches'] for k,v in train_stats['component_sums'].items()} if train_stats['batches'] else {}
        summary=dict(stage=stage,epoch=epoch+1,stats_first_batch=train_stats['first_batch'],batches=train_stats['batches'],samples=train_stats['samples'],loss_mean=train_stats['loss_sum']/max(train_stats['batches'],1),groups=loss_groups(means))
        with (folder/'epoch_summary.jsonl').open('a') as f:f.write(json.dumps(summary)+'\n')
        print('EPOCH_SUMMARY',json.dumps(summary),flush=True)
        # Persist the completed epoch even if validation is interrupted.
        file=folder/('epoch_%03d.pth'%(epoch+1));snapshot(file,epoch+1)
        print('EPOCH_WEIGHTS_SAVED_BEFORE_VALIDATION',file,flush=True)
        metrics=None
        if epoch==0 or (epoch+1)%5==0 or epoch+1==epochs:
            metrics=validate(model,val,stage==2,folder/('metrics_epoch_%03d.json'%(epoch+1)))
            print('EVALUATION',stage,epoch+1,'mAP',metrics['detection']['mAP'],'NDS',metrics['detection']['NDS'],'forecast',metrics.get('forecast',{}).get('epa'),flush=True)
        current=metrics['detection']['mAP'] if metrics else None
        is_best=current is not None and current>best
        if is_best:best=current
        file=folder/('epoch_%03d.pth'%(epoch+1));snapshot(file,epoch+1)
        if is_best:
            link=folder/'best_map.pth'
            if link.is_symlink() or link.exists():link.unlink()
            link.symlink_to(file.name)
        # Keep periodic and final snapshots; discard only unused rolling epochs.
        for previous in folder.glob('epoch_*.pth'):
            number=int(previous.stem.split('_')[-1])
            protected={latest.resolve(),(folder/'best_map.pth').resolve()}
            if number<epoch and number%5!=0 and previous.resolve() not in protected:previous.unlink()
        print('CHECKPOINT',file,flush=True)
    final=latest.resolve();del model,optimizer,scaler;torch.cuda.empty_cache();return final

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,default=ROOT/'configs/resolve/sparsedrive_camera.py')
    parser.add_argument('--work-dir',type=Path,default=ROOT/'experiments/sparsedrive_camera_resolve_20261028')
    parser.add_argument('--workers',type=int,default=4)
    parser.add_argument('--save-every-updates',type=int,default=250)
    parser.add_argument('--diagnose-after-batch',type=int,default=int(os.environ.get('SPARSEDRIVE_DIAGNOSE_AFTER_BATCH','-1')))
    args=parser.parse_args();args.work_dir=args.work_dir.resolve();args.work_dir.mkdir(parents=True,exist_ok=True)
    cfg=Config.fromfile(str(args.config));random.seed(cfg.seed);np.random.seed(cfg.seed);torch.manual_seed(cfg.seed);torch.cuda.manual_seed_all(cfg.seed)
    manifest=json.loads((ROOT/'data/resolve_sparsedrive/manifest.json').read_text())
    assert all(v['prepared_frames']==v['expected_frames'] for v in manifest['splits'].values()) and not manifest['missing']
    print('FULL_SPLIT',manifest['splits'],'GPU',torch.cuda.get_device_name(0),'ENV',sys.executable,flush=True)
    final=train_stage(args,1);train_stage(args,2,final)
    print('PIPELINE_COMPLETED',flush=True)

if __name__=='__main__':main()
