#!/usr/bin/env python3
"""Render cached ground truth on exactly the same frames as model outputs."""
from __future__ import annotations
import argparse
import json
import pickle
import sys
from datetime import datetime
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT),str(ROOT/'third_party/MC-DeTra')]
from detra_repro.data.resolve_camera import CLASS_NAMES
from tools.visualization.resolve_viz_common import fixed_samples,prepare_output,render_frame,verify_output

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sample-manifest',type=Path,default=ROOT/'experiments/visualization/transfusion_l_20261028_105558/manifest.json')
    parser.add_argument('--output-dir',type=Path,default=ROOT/'visual/gt')
    parser.add_argument('--overwrite',action='store_true')
    args=parser.parse_args()
    cache=ROOT/'data/resolve_cache'
    with (ROOT/'data/resolve_protocol/val.pkl').open('rb') as f:infos=pickle.load(f)
    picked=fixed_samples(cache,infos,args.sample_manifest.resolve())
    out=args.output_dir.resolve()
    records=ROOT/'experiments/visualization'/('gt_'+datetime.now().strftime('%Y%m%d_%H%M%S'))
    prepare_output(out,records,args.overwrite)
    samples=[]
    for order,(item,info) in enumerate(picked,1):
        boxes=info['gt_boxes'][:,:7].astype(np.float64)
        names=[str(n) for n in info['gt_names']]
        with np.load(cache/item['point_npz'],allow_pickle=True) as raw:
            assert np.array_equal(boxes,raw['gt_boxes'])
            assert names==[CLASS_NAMES[int(v)] for v in raw['gt_types']]
        cameras,overview=render_frame(cache,item,info,boxes,names,out,order,'Ground truth',None)
        samples.append(dict(order=order,token=info['token'],rosbag=item['rosbag'],frame_index=int(item['frame_index']),num_gt=len(boxes),cameras=cameras,overview=overview))
        print('GT %02d objects=%d drawn=%s'%(order,len(boxes),[c['boxes_drawn'] for c in cameras]),flush=True)
    verify_output(out,len(picked))
    reference=ROOT/'visual/transfusion-l'
    assert {p.name for p in out.iterdir()}=={p.name for p in reference.iterdir()}
    metadata=dict(source='shared validation ground truth, verified against original cache',source_sample_manifest=str(args.sample_manifest.resolve()),object_text=False,score_threshold=None,samples=samples)
    (records/'manifest.json').write_text(json.dumps(metadata,indent=2))
    print('DONE %d images; records outside output: %s'%(len(picked)*5,records),flush=True)

if __name__=='__main__':main()
