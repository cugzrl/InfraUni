#!/usr/bin/env python3
"""Render crowded, diverse validation scenes using a TransFusion checkpoint.

The output directory contains only images: 4 camera panels + 1 overview per
sample. Reproduction metadata and selected predictions go in experiments/.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import logging
import os
import pickle
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
VENDOR = ROOT / 'third_party/RESOLVE'
sys.path[:0] = [str(ROOT), str(VENDOR), str(VENDOR / 'tools'), str(ROOT / 'third_party/MC-DeTra')]
sys.path.append('/home/kemove/anaconda3/envs/bevfusion/lib/python3.8/site-packages')

import cv2
import numpy as np
import torch
from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.models import build_network, load_data_to_gpu
from infrauni.resolve.dataset import ResolveSequenceDataset
from detra_repro.data.resolve_camera import CAMERAS, project_points, read_lidar_xyz
from tools.visualization.resolve_viz_common import fixed_samples, prepare_output, render_frame, verify_output


def choose_samples(cache_root, infos, count):
    manifest = json.loads((cache_root / 'cameras_manifest.json').read_text())
    by_path = {str(Path(i['cache_path']).resolve()): i for i in infos}
    groups = {}
    for item in manifest['samples']:
        if item['split'] != 'val' or int(item['cameras_present']) != 4:
            continue
        info = by_path.get(str((cache_root / item['point_npz']).resolve()))
        if info is None:
            continue
        image_paths = [cache_root / 'val' / item['batch_id'] / 'images' / cam /
                       ('sample_%06d.jpg' % int(item['frame_index'])) for cam in CAMERAS]
        if not all(p.is_file() for p in image_paths):
            continue
        key = (len(info['gt_boxes']), len(set(info['gt_names'])), int(item['json_matches']), -int(item['frame_index']))
        candidate = (key, item, info)
        if item['rosbag'] not in groups or key > groups[item['rosbag']][0]:
            groups[item['rosbag']] = candidate
    if len(groups) < count:
        raise RuntimeError('Only %d distinct validation bags with four cameras' % len(groups))
    remaining = list(groups.values())
    selected, covered = [], set()
    while len(selected) < count:
        # Density dominates; prefer complementary classes among similarly busy scenes.
        best = max(remaining, key=lambda row: (row[0][0] + 2 * len(set(row[2]['gt_names']) - covered), row[0], row[1]['rosbag']))
        remaining.remove(best)
        selected.append((best[1], best[2]))
        covered.update(best[2]['gt_names'])
    selected.sort(key=lambda row: (-len(row[1]['gt_boxes']), row[0]['rosbag'], int(row[0]['frame_index'])))
    return selected


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    run = ROOT / 'experiments/baselines/infrauni/transfusion_sequence/resolve_sequence_20261027'
    parser.add_argument('--checkpoint', type=Path, default=run / 'ckpt/checkpoint_epoch_20.pth')
    parser.add_argument('--config', type=Path, default=ROOT / 'configs/resolve/transfusion_sequence.yaml')
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'visual/transfusion-l')
    parser.add_argument('--num-samples', type=int, default=10)
    parser.add_argument('--score-thresh', type=float, default=.1)
    parser.add_argument('--sample-manifest', type=Path)
    parser.add_argument('--overwrite', action='store_true')
    args = parser.parse_args()
    args.checkpoint = args.checkpoint.resolve()
    args.config = args.config.resolve()
    if args.sample_manifest: args.sample_manifest = args.sample_manifest.resolve()
    args.output_dir = args.output_dir.resolve()
    record_dir = ROOT / 'experiments/visualization' / ('transfusion_l_' + datetime.now().strftime('%Y%m%d_%H%M%S'))
    prepare_output(args.output_dir,record_dir,args.overwrite)
    logger = logging.getLogger('transfusion-viz');logger.setLevel(logging.INFO)
    logger.addHandler(logging.StreamHandler(sys.stdout))
    os.chdir(VENDOR / 'tools')
    cfg_from_yaml_file(str(args.config), cfg)
    dataset = ResolveSequenceDataset(cfg.DATA_CONFIG, cfg.CLASS_NAMES, training=False, logger=logger)
    picked = fixed_samples(ROOT/'data/resolve_cache',dataset.infos,args.sample_manifest) if args.sample_manifest else choose_samples(ROOT / 'data/resolve_cache', dataset.infos, args.num_samples)
    assert len(picked)==args.num_samples
    dataset.infos = [copy.deepcopy(info) for item, info in picked]
    for rank, (item, info) in enumerate(picked, 1):
        print('SELECT %02d GT=%d classes=%s %s frame=%d' % (rank, len(info['gt_boxes']), ','.join(sorted(set(info['gt_names']))), item['rosbag'], int(item['frame_index'])), flush=True)
    model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=dataset)
    model.load_params_from_file(filename=str(args.checkpoint), logger=logger, to_cpu=True)
    model.cuda().eval()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    records = []
    cache = ROOT / 'data/resolve_cache'
    for index, (item, info) in enumerate(picked):
        with torch.no_grad():
            batch = dataset.collate_batch([dataset[index]])
            load_data_to_gpu(batch)
            preds, _ = model(batch)
        pred = preds[0]
        keep = pred['pred_scores'] >= args.score_thresh
        boxes = pred['pred_boxes'][keep].detach().cpu().numpy().astype(np.float64)
        scores = pred['pred_scores'][keep].detach().cpu().numpy()
        names = [cfg.CLASS_NAMES[int(label) - 1] for label in pred['pred_labels'][keep].detach().cpu().numpy()]
        if not np.isfinite(boxes).all():
            raise ValueError('Non-finite predicted boxes')
        cameras, overview = render_frame(cache,item,info,boxes,names,args.output_dir,index+1,'TransFusion-L',args.score_thresh)
        record = dict(order=index + 1, token=info['token'], rosbag=item['rosbag'], frame_index=int(item['frame_index']), num_gt=len(info['gt_boxes']), classes=sorted(set(info['gt_names'])), num_predictions=len(boxes), cameras=cameras, overview=overview,
                      predicted_boxes=boxes.tolist(), scores=scores.tolist(), predicted_names=names, all_predicted_boxes=pred['pred_boxes'].detach().cpu().numpy().tolist(), all_scores=pred['pred_scores'].detach().cpu().numpy().tolist(), all_names=[cfg.CLASS_NAMES[int(v)-1] for v in pred['pred_labels'].detach().cpu().numpy()])
        records.append(record)
        print('RENDER %02d detections=%d drawn=%s' % (index + 1, len(boxes), [c['boxes_drawn'] for c in cameras]), flush=True)
    metadata = dict(checkpoint=str(args.checkpoint), checkpoint_sha256=hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(), config=str(args.config), split='val', object_text=False, source_sample_manifest=str(args.sample_manifest), score_threshold=args.score_thresh, selection='one crowded frame per distinct rosbag, with preference for class diversity; final ordering by GT density', samples=records)
    (record_dir / 'manifest.json').write_text(json.dumps(metadata, indent=2))
    verify_output(args.output_dir,args.num_samples)
    print('DONE %d images; metadata outside visual directory: %s' % (5*args.num_samples, record_dir), flush=True)


if __name__ == '__main__':
    main()
