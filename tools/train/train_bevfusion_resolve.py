"""Train official RESOLVE BEVFusion with canonical multimodal inputs and mAP/NDS9."""
import argparse
from collections import deque
import hashlib
import json
import math
import os
from pathlib import Path
import pickle
import random
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
VENDOR = ROOT / 'third_party/RESOLVE'
sys.path[:0] = [str(ROOT), str(VENDOR), str(VENDOR/'tools')]
sys.path.append('/home/kemove/anaconda3/envs/bevfusion/lib/python3.8/site-packages')
import numpy as np
import torch
from torch.utils.data import DataLoader
from easydict import EasyDict
from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.models import build_network, load_data_to_gpu
from pcdet.utils import common_utils
from train_utils.optimization import build_optimizer, build_scheduler
from infrauni.resolve.bevfusion_dataset import ResolveBEVFusionDataset


def atomic_save(state, path):
    temporary = path.with_suffix('.tmp')
    torch.save(state, temporary)
    temporary.replace(path)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', type=Path, default=ROOT/'configs/resolve/bevfusion_roi_v2.yaml')
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=True)
    logger = common_utils.create_logger(a.output/'train.log')
    common_utils.set_random_seed(666)
    torch.set_num_threads(4)
    cfg_from_yaml_file(str(a.config), cfg)
    opt = cfg.OPTIMIZATION
    (a.output/'config.yaml').write_text(a.config.read_text())
    data_root = Path(cfg.DATA_CONFIG.DATA_PATH)
    manifest = json.loads((data_root/'manifest.json').read_text())
    hashes = {s: hashlib.sha256((data_root/'infos'/(s+'.pkl')).read_bytes()).hexdigest() for s in ('train', 'val')}
    provenance = dict(dataset=manifest, info_sha256=hashes, point_cloud_range=cfg.DATA_CONFIG.POINT_CLOUD_RANGE,
                      model_source=str(VENDOR), environment=sys.executable, evaluator='infrauni.eval.resolve_detection.evaluate_boxes',
                      image_pretrained=cfg.MODEL.IMAGE_BACKBONE.INIT_CFG.checkpoint, camera_order='canonical axis1-axis4',
                      lidar_input='raw current scan xyzi, zero timestamp; no old cropped cache',
                      batch_size=opt.BATCH_SIZE_PER_GPU, accumulate=opt.ACCUMULATE, effective_batch=opt.BATCH_SIZE_PER_GPU*opt.ACCUMULATE)
    (a.output/'protocol.json').write_text(json.dumps(provenance, indent=2))
    train_set = ResolveBEVFusionDataset(cfg.DATA_CONFIG, cfg.CLASS_NAMES, True, logger=logger)
    val_set = ResolveBEVFusionDataset(cfg.DATA_CONFIG, cfg.CLASS_NAMES, False, logger=logger)
    if {r['scene_token'] for r in train_set.infos} & {r['scene_token'] for r in val_set.infos}:
        raise ValueError('Train/validation sequence overlap')
    if len(train_set) != manifest['splits']['train']['frames'] or len(val_set) != manifest['splits']['val']['frames']:
        raise ValueError('Canonical manifest count mismatch')
    model = build_network(cfg.MODEL, len(cfg.CLASS_NAMES), train_set).cuda()
    # Confirm that every Swin tensor was actually initialized from the supplied checkpoint.
    from pcdet.models.model_utils.swin_utils import swin_converter
    pre = torch.load(cfg.MODEL.IMAGE_BACKBONE.INIT_CFG.checkpoint, map_location='cpu')
    pre = pre.get('state_dict', pre.get('model', pre))
    if cfg.MODEL.IMAGE_BACKBONE.CONVERT_WEIGHTS:
        pre = swin_converter(pre)
    pre = {k[9:]: v for k, v in pre.items() if k.startswith('backbone.')}
    ms = model.image_backbone.state_dict()
    matched = [k for k, v in pre.items() if k in ms and v.shape == ms[k].shape]
    coverage = sum(ms[k].numel() for k in matched) / sum(v.numel() for v in ms.values())
    if coverage < .95 or not all(torch.equal(ms[k].cpu(), pre[k]) for k in matched):
        raise RuntimeError('Swin pretrained weights were not loaded correctly')
    logger.info('Swin pretrained tensor coverage %.5f; train=%d val=%d; effective batch=%d',
                coverage, len(train_set), len(val_set), opt.BATCH_SIZE_PER_GPU*opt.ACCUMULATE)
    optimizer = build_optimizer(model, opt)
    batches = math.ceil(len(train_set)/opt.BATCH_SIZE_PER_GPU)
    updates_per_epoch = math.ceil(batches/opt.ACCUMULATE)
    scheduler, _ = build_scheduler(optimizer, updates_per_epoch, opt.NUM_EPOCHS, -1, opt)
    ckpt_dir = a.output/'checkpoints'
    ckpt_dir.mkdir(exist_ok=True)
    epoch, next_batch, update, best_map = 0, 0, 0, -1.
    latest = ckpt_dir/'latest.pth'
    if latest.exists():
        state = torch.load(latest, map_location='cpu')
        if state['info_sha256'] != hashes:
            raise RuntimeError('Dataset changed since checkpoint')
        model.load_state_dict(state['model_state'])
        optimizer.load_state_dict(state['optimizer_state'])
        epoch, next_batch, update, best_map = state['epoch'], state['next_batch'], state['it'], state['best_map']
        random.setstate(state['python_rng']); np.random.set_state(state['numpy_rng'])
        torch.set_rng_state(state['torch_rng']); torch.cuda.set_rng_state_all(state['cuda_rng'])
        logger.info('RESUME epoch=%d batch=%d update=%d', epoch+1, next_batch, update)
    def save(path, ep, nb):
        atomic_save(dict(model_state=model.state_dict(), optimizer_state=optimizer.state_dict(), epoch=ep,
                         next_batch=nb, it=update, best_map=best_map, info_sha256=hashes,
                         python_rng=random.getstate(), numpy_rng=np.random.get_state(),
                         torch_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state_all()), path)
    val_loader = DataLoader(val_set, batch_size=1, shuffle=False, num_workers=4, pin_memory=True,
                            collate_fn=val_set.collate_batch, worker_init_fn=common_utils.worker_init_fn)
    # Exercise the real inference/decoder path once, without writing probe predictions or metrics.
    if update == 0:
        model.eval()
        with torch.no_grad():
            batch = next(iter(val_loader)); load_data_to_gpu(batch)
            predictions, _ = model(batch)
            if not all(torch.isfinite(row['pred_boxes']).all() and torch.isfinite(row['pred_scores']).all() for row in predictions):
                raise FloatingPointError('Nonfinite initial inference')
        logger.info('Canonical four-camera inference and decoder passed; starting full training')
        del batch, predictions
        torch.cuda.empty_cache()
    start = last_save = time.monotonic()
    while epoch < opt.NUM_EPOCHS:
        loader = DataLoader(train_set, batch_size=opt.BATCH_SIZE_PER_GPU, shuffle=True, drop_last=False,
                            num_workers=4, pin_memory=True, collate_fn=train_set.collate_batch,
                            worker_init_fn=common_utils.worker_init_fn,
                            generator=torch.Generator().manual_seed(666+epoch))
        model.train(); optimizer.zero_grad()
        rolling = deque(maxlen=100); total_loss = 0.; seen = 0
        for batch_idx, batch in enumerate(loader):
            if batch_idx < next_batch:
                continue
            if batch_idx % opt.ACCUMULATE == 0:
                scheduler.step(update, epoch + batch_idx/len(loader))
            group_start = (batch_idx//opt.ACCUMULATE)*opt.ACCUMULATE
            group_end = min(group_start+opt.ACCUMULATE, len(loader))
            group_samples = min(len(train_set), group_end*opt.BATCH_SIZE_PER_GPU) - group_start*opt.BATCH_SIZE_PER_GPU
            load_data_to_gpu(batch)
            ret, parts, _ = model(batch)
            loss = ret['loss'].mean()
            if not torch.isfinite(loss):
                raise FloatingPointError('Nonfinite loss at epoch %d batch %d' % (epoch+1, batch_idx))
            (loss * batch['batch_size']/group_samples).backward()
            value = float(loss.detach()); rolling.append(value); total_loss += value; seen += 1
            boundary = batch_idx+1 == group_end
            if boundary:
                grad = torch.nn.utils.clip_grad_norm_(model.parameters(), opt.GRAD_NORM_CLIP, error_if_nonfinite=True)
                optimizer.step(); optimizer.zero_grad(); update += 1
                if update == 1 or time.monotonic()-last_save >= 600:
                    save(latest, epoch, batch_idx+1); last_save = time.monotonic()
                if update <= 5 or update % 20 == 0:
                    scalars = {k: round(float(v), 5) for k, v in parts.items() if np.isscalar(v) or (torch.is_tensor(v) and v.numel()==1)}
                    logger.info('TRAIN epoch=%d/%d batch=%d/%d update=%d loss=%.5f rolling100=%.5f epoch_avg=%.5f lr=%.8f grad_preclip=%.4f gpu_peak_gb=%.2f elapsed_min=%.1f parts=%s',
                                epoch+1, opt.NUM_EPOCHS, batch_idx+1, len(loader), update, value, sum(rolling)/len(rolling), total_loss/seen,
                                optimizer.lr, float(grad), torch.cuda.max_memory_allocated()/2**30, (time.monotonic()-start)/60,
                                json.dumps(scalars))
            del loss, ret, parts, batch
        save(ckpt_dir/('epoch_%03d.pth' % (epoch+1)), epoch, len(loader))
        model.eval(); annos = []
        with torch.no_grad():
            for vi, batch in enumerate(val_loader):
                load_data_to_gpu(batch)
                pred, _ = model(batch)
                annos.extend(val_set.generate_prediction_dicts(batch, pred, cfg.CLASS_NAMES))
                if (vi+1) % 500 == 0:
                    logger.info('VAL epoch=%d frame=%d/%d', epoch+1, vi+1, len(val_set))
        out = a.output/'eval'/('epoch_%03d' % (epoch+1)); out.mkdir(parents=True, exist_ok=True)
        with (out/'predictions.pkl').open('wb') as f:
            pickle.dump(annos, f)
        message, metrics = val_set.evaluation(annos, cfg.CLASS_NAMES, output_path=out)
        logger.info('EVAL epoch=%d %s', epoch+1, message)
        with (a.output/'metrics.jsonl').open('a') as f:
            f.write(json.dumps(dict(epoch=epoch+1, epoch_loss=total_loss/max(1,seen), **metrics))+'\n')
        if metrics['mAP'] > best_map:
            best_map = metrics['mAP']
            save(ckpt_dir/'best_map.pth', epoch+1, 0)
        epoch += 1; next_batch = 0
        save(latest, epoch, 0); last_save = time.monotonic()
        del annos
    logger.info('TRAINING_COMPLETED epochs=%d best_mAP=%.6f', epoch, best_map)


if __name__ == '__main__':
    main()
