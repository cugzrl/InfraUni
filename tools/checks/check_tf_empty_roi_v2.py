from pathlib import Path
import sys,numpy as np,torch
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT),str(ROOT/'third_party/RESOLVE')]
sys.path.append('/home/kemove/anaconda3/envs/bevfusion/lib/python3.8/site-packages')
from pcdet.config import cfg,cfg_from_yaml_file
from pcdet.utils.common_utils import create_logger
from pcdet.models import build_network,load_data_to_gpu
from infrauni.resolve.dataset import ResolveSequenceDataset
cfg_from_yaml_file(str(ROOT/'configs/resolve/transfusion_roi_v2.yaml'),cfg)
ds=ResolveSequenceDataset(cfg.DATA_CONFIG,cfg.CLASS_NAMES,training=True,logger=create_logger())
empty=next(i for i,d in enumerate(ds.infos) if not len(d['gt_boxes']));positive=next(i for i,d in enumerate(ds.infos) if len(d['gt_boxes']))
m=build_network(cfg.MODEL,num_class=10,dataset=ds).cuda().train()
for indices in [[empty],[empty,positive]]:
 samples=[ds[i] for i in indices];assert samples[0]['metadata']['token']==ds.infos[empty]['token']
 batch=ds.collate_batch(samples);load_data_to_gpu(batch);m.zero_grad(set_to_none=True);ret,_,_=m(batch);loss=ret['loss'].mean();assert torch.isfinite(loss);loss.backward()
 assert all(torch.isfinite(p.grad).all() for p in m.parameters() if p.grad is not None)
 print('TF_EMPTY_AND_MIXED_PASS',indices,float(loss),flush=True)
