"""Render cached full-sequence SparseDrive inference on the fixed comparison frames."""
from __future__ import annotations
import argparse,json,pickle,sys,hashlib,os
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from infrauni.resolve.sparsedrive.runtime import setup
setup()
from tools.visualization.resolve_viz_common import fixed_samples,prepare_output,render_frame,verify_output

def sha256_file(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):digest.update(block)
    return digest.hexdigest()

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--prediction-cache',type=Path,required=True)
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--sample-manifest',type=Path,default=ROOT/'experiments/visualization/transfusion_l_20261028_111049/manifest.json')
    p.add_argument('--output-dir',type=Path,default=ROOT/'visual/sparsedrive')
    p.add_argument('--score-thresh',type=float,default=.1)
    p.add_argument('--overwrite',action='store_true')
    a=p.parse_args();records=a.prediction_cache.resolve().parent
    with a.prediction_cache.open('rb') as f:predictions=pickle.load(f)
    with (ROOT/'data/resolve_protocol/val.pkl').open('rb') as f:infos=pickle.load(f)
    cache=ROOT/'data/resolve_cache'
    # The shared infos can point to the reorganized multimodal dataset. Verify
    # content identity before supplying legacy renderer paths on shallow copies.
    selected=json.loads(a.sample_manifest.read_text())['samples']
    selection={s['token']:s for s in selected}
    old=json.loads((cache/'cameras_manifest.json').read_text())['samples']
    frames={(s['rosbag'],int(s['frame_index'])):s for s in old if s['split']=='val'}
    aligned=[]
    for info in infos:
        if info['token'] not in selection:aligned.append(info);continue
        row=selection[info['token']];item=frames[(row['rosbag'],int(row['frame_index']))]
        point_path=cache/item['point_npz']
        with np.load(point_path,allow_pickle=True) as point:
            assert np.array_equal(info['gt_boxes'][:,:7],point['gt_boxes'])
            assert np.array_equal(info['gt_labels'],point['gt_types'])
        with np.load(cache/item['camera_npz'],allow_pickle=False) as camera:source=str(camera['source_lidar'])
        assert os.path.samefile(source,info['lidar_path']) or sha256_file(source)==sha256_file(info['lidar_path'])
        aligned.append(dict(info,cache_path=str(point_path),lidar_path=source))
    picked=fixed_samples(cache,aligned,a.sample_manifest)
    assert len(picked)==10 and set(predictions)=={info['token'] for _,info in picked}
    prepare_output(a.output_dir,records,a.overwrite);samples=[]
    for order,(item,info) in enumerate(picked,1):
        selected=[d for d in predictions[info['token']] if d['score']>=a.score_thresh]
        boxes=np.asarray([d['translation']+d['size']+[d['yaw']] for d in selected],dtype=np.float64).reshape(-1,7)
        assert np.isfinite(boxes).all() and (boxes[:,3:6]>0).all()
        names=[d['name'] for d in selected]
        cameras,overview=render_frame(cache,item,info,boxes,names,a.output_dir,order,'SparseDrive Camera-only',a.score_thresh)
        samples.append(dict(order=order,token=info['token'],rosbag=item['rosbag'],frame_index=int(item['frame_index']),num_gt=len(info['gt_boxes']),num_predictions=len(boxes),predicted_boxes=boxes.tolist(),predicted_names=names,scores=[d['score'] for d in selected],cameras=cameras,overview=overview))
        print('SPARSEDRIVE_RENDER',order,'predictions',len(boxes),'drawn',[x['boxes_drawn'] for x in cameras],flush=True)
    verify_output(a.output_dir,len(picked))
    reference=ROOT/'visual/transfusion-l'
    assert {p.name for p in a.output_dir.iterdir()}=={p.name for p in reference.iterdir()}
    metadata=dict(checkpoint=str(a.checkpoint.resolve()),checkpoint_sha256=sha256_file(a.checkpoint),prediction_cache=str(a.prediction_cache.resolve()),shared_info_path=str((ROOT/'data/resolve_protocol/val.pkl').resolve()),shared_info_frames=len(infos),split='val',full_sequence_inference=True,source_sample_manifest=str(a.sample_manifest),path_alias_content_verified=True,object_text=False,score_threshold=a.score_thresh,samples=samples)
    (records/'manifest.json').write_text(json.dumps(metadata,indent=2))
    print('DONE',len(picked)*5,'images',a.output_dir,flush=True)

if __name__=='__main__':main()
