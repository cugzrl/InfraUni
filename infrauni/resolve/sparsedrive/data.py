"""Camera-only inputs, contiguous sequence batches, unchanged shared targets."""
import pickle
import math
from pathlib import Path
from collections import defaultdict
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset,Sampler
cv2.setNumThreads(0)

class ResolveCameraDataset(Dataset):
    def __init__(self,path,image_size=(704,384),training=False):
        with Path(path).open('rb') as f:self.infos=pickle.load(f)
        self.image_size=tuple(image_size);self.training=training
    def __len__(self):return len(self.infos)
    def __getitem__(self,index):
        d=self.infos[index];images=[];projections=[];w,h=self.image_size
        brightness=np.random.uniform(.9,1.1) if self.training else 1.
        for path,p in zip(d['images'],d['projection_mat']):
            if path is None:
                images.append(np.zeros((3,h,w),np.float32));projections.append(np.zeros((4,4),np.float32));continue
            im=cv2.imread(path)
            if im is None:raise FileNotFoundError(path)
            ih,iw=im.shape[:2];im=cv2.resize(im,(w,h),interpolation=cv2.INTER_LINEAR)
            im=np.clip(im[...,::-1].astype(np.float32)*brightness,0,255)
            im=(im-np.array([123.675,116.28,103.53],np.float32))/np.array([58.395,57.12,57.375],np.float32)
            images.append(np.ascontiguousarray(im.transpose(2,0,1)))
            scale=np.eye(4,dtype=np.float32);scale[0,0]=w/iw;scale[1,1]=h/ih;projections.append(scale@p)
        meta=dict(token=d['token'],scene_token=d['scene_token'],T_global=np.eye(4),T_global_inv=np.eye(4),timestamp=d['timestamp'])
        inputs=dict(camera_valid=torch.from_numpy(d.get('camera_valid',np.ones(4,dtype=bool))),img=torch.from_numpy(np.stack(images)),timestamp=torch.tensor(d['timestamp'],dtype=torch.float64),projection_mat=torch.from_numpy(np.stack(projections)),image_wh=torch.tensor([[w,h]]*4,dtype=torch.float32),img_metas=meta)
        if self.training:
            meta['instance_id']=d['gt_track_ids']
            rel=d['gt_future']-d['gt_boxes'][:,None,:2]
            rel=np.where(d['gt_future_mask'][...,None],rel,0.)
            delta=np.diff(np.concatenate([np.zeros((len(rel),1,2),np.float32),rel],axis=1),axis=1)
            inputs.update(gt_bboxes_3d=torch.from_numpy(d['gt_boxes'].copy()),gt_labels_3d=torch.from_numpy(d['gt_labels'].copy()),gt_agent_fut_trajs=torch.from_numpy(delta.astype(np.float32)),gt_agent_fut_masks=torch.from_numpy(d['gt_future_mask'].astype(np.float32)))
        return inputs

def collate(batch):
    keys=batch[0].keys();out={}
    for k in keys:
        if k=='img_metas' or k.startswith('gt_'):out[k]=[d[k] for d in batch]
        else:out[k]=torch.stack([d[k] for d in batch])
    return out

def to_device(batch,device='cuda'):
    return {k:(v.to(device,non_blocking=True) if torch.is_tensor(v) else [x.to(device,non_blocking=True) if torch.is_tensor(x) else x for x in v]) for k,v in batch.items()}

class SequenceBatchSampler(Sampler):
    """Each batch slot progresses within a sequence, with no dropped frames."""
    def __init__(self,infos,batch_size=2,seed=20261028):
        self.groups=defaultdict(list)
        for i,d in enumerate(infos):self.groups[d['scene_token']].append(i)
        self.batch_size=batch_size;self.seed=seed;self.epoch=0;self.start_batch=0
    def set_epoch(self,epoch):self.epoch=epoch;self.start_batch=0
    def __len__(self):
        # Exact length can vary with the last underfilled batch; no samples drop.
        return sum(1 for _ in self.__iter__())
    def __iter__(self):
        groups=list(self.groups.values());rng=np.random.default_rng(self.seed+self.epoch);order=rng.permutation(len(groups)).tolist()
        queues=[[] for _ in range(self.batch_size)];cursor=0;batch_index=0
        while True:
            batch=[]
            for slot in range(self.batch_size):
                if not queues[slot] and cursor<len(order):queues[slot]=list(reversed(groups[order[cursor]]));cursor+=1
                if queues[slot]:batch.append(queues[slot].pop())
            if not batch:break
            if batch_index>=self.start_batch:yield batch
            batch_index+=1
