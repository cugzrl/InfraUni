"""Run MC-DeTra against canonical infos and uncropped raw LiDAR sweeps."""
import sys,runpy,os
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT),str(ROOT/'third_party/MC-DeTra')]
import torch
_original_load=torch.load
def _trusted_load(*args,**kwargs):
    kwargs.setdefault('weights_only',False)
    return _original_load(*args,**kwargs)
torch.load=_trusted_load
if os.environ.get("MC_ANOMALY_CHECK")=="1":torch.autograd.set_detect_anomaly(True)
# Keep full precision and checkpoint the dense 1920x1920 encoder activations.
from detra_repro.models.bev import PaperLikeLidarEncoder
_encoder_forward=PaperLikeLidarEncoder.forward
def _fp32_tree(value):
    if isinstance(value,torch.Tensor) and value.dtype==torch.bfloat16:return value.float()
    if isinstance(value,dict):return {k:_fp32_tree(v) for k,v in value.items()}
    if isinstance(value,list):return [_fp32_tree(v) for v in value]
    if isinstance(value,tuple):return tuple(_fp32_tree(v) for v in value)
    return value
def encoder_checkpoint_forward(self,*args,**kwargs):
    self.activation_checkpointing=True
    with torch.autocast(device_type='cuda',dtype=torch.bfloat16,enabled=False):
        result=_encoder_forward(self,*args,**kwargs)
    return _fp32_tree(result)
PaperLikeLidarEncoder.forward=encoder_checkpoint_forward
print('MC_PRECISION: all modules=float32; dense encoder activation checkpointing=enabled',flush=True)
import random,json,numpy as np
if '--experiment-json' in sys.argv:
    config=json.loads(Path(sys.argv[sys.argv.index('--experiment-json')+1]).read_text())
    seed=int(config.get('deterministic_seed',0));random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    print('MC_TRAINING_SEED',seed,flush=True)
import atexit
@atexit.register
def _memory_report():
    if torch.cuda.is_initialized():print('MC_PEAK_MEMORY_MB',torch.cuda.max_memory_allocated()/2**20,'PEAK_RESERVED_MB',torch.cuda.max_memory_reserved()/2**20,flush=True)
_reserved_probe=None
if os.environ.get('MC_SIMULATE_GPU_RESERVATION_MB'):
    _reserved_probe=torch.empty(int(os.environ['MC_SIMULATE_GPU_RESERVATION_MB'])*2**20,dtype=torch.uint8,device='cuda')
    print('MC_MEMORY_SHARED_GPU_SIMULATION_MB',os.environ['MC_SIMULATE_GPU_RESERVATION_MB'],flush=True)
import detra_repro.data.dataset as dataset_module
from infrauni.resolve.mcdetra_dataset import ResolveRawDataset
dataset_module.CachedWaymoDataset=ResolveRawDataset
os.environ['INFRAUNI_SHARED_PROTOCOL']='1'
if __name__=='__main__':runpy.run_module('detra_repro.train_cached',run_name='__main__')
