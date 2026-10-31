"""Shared fixed-frame, image-only rendering for RESOLVE model comparisons."""
from __future__ import annotations
import json
import shutil
from pathlib import Path
import cv2
import numpy as np
from detra_repro.data.resolve_camera import CAMERAS, read_lidar_xyz
from tools.visualization.visualize_resolve_cameras import draw_projection, _mosaic

ROOT=Path(__file__).resolve().parents[2]

def fixed_samples(cache_root, infos, manifest_path):
    selected=json.loads(Path(manifest_path).read_text())['samples']
    cameras=json.loads((cache_root/'cameras_manifest.json').read_text())['samples']
    by_frame={(r['rosbag'],int(r['frame_index'])):r for r in cameras if r['split']=='val'}
    by_token={str(r['token']):r for r in infos}
    picked=[]
    for row in selected:
        item=by_frame[(row['rosbag'],int(row['frame_index']))]
        info=by_token[row['token']]
        assert Path(info['cache_path']).resolve()==(cache_root/item['point_npz']).resolve()
        picked.append((item,info))
    return picked

def prepare_output(output, record_dir, overwrite=False):
    output=Path(output).resolve()
    if (ROOT/'visual').resolve() not in output.parents:
        raise ValueError('Comparison output must be a subdirectory of InfraUni/visual')
    record_dir.mkdir(parents=True,exist_ok=True)
    if output.exists() and any(output.iterdir()):
        if not overwrite:raise RuntimeError('Output is nonempty; pass --overwrite to archive old images')
        shutil.move(str(output),str(record_dir/'previous_images'))
    output.mkdir(parents=True,exist_ok=True)

def render_frame(cache,item,info,boxes,names,out,index,model_name,threshold):
    stem='%02d_%s_f%06d'%(index,item['rosbag'],int(item['frame_index']))
    panels=[];stats=[]
    with np.load(cache/item['camera_npz'],allow_pickle=False) as camera:
        assert Path(str(camera['source_lidar'])).resolve()==Path(info['lidar_path']).resolve()
        points=read_lidar_xyz(str(camera['source_lidar']))
        for ci in range(len(CAMERAS)):
            name=str(camera['camera_names'][ci])
            path=cache/'val'/item['batch_id']/'images'/name/('sample_%06d.jpg'%int(item['frame_index']))
            image=cv2.imread(str(path))
            if image is None:raise FileNotFoundError(path)
            canvas,counts=draw_projection(image,points,boxes[:,:7],names,camera['lidar2cam'][ci],camera['K'][ci],camera['distortion'][ci],80.)
            filename=stem+'_'+name+'.jpg'
            if not cv2.imwrite(str(out/filename),canvas,[cv2.IMWRITE_JPEG_QUALITY,95]):raise IOError(filename)
            panels.append(canvas);stats.append(dict(camera=name,file=filename,**counts))
        filename=stem+'_00_overview.jpg'
        if not cv2.imwrite(str(out/filename),_mosaic(panels),[cv2.IMWRITE_JPEG_QUALITY,95]):raise IOError(filename)
    return stats,filename

def verify_output(output,n):
    paths=sorted(output.iterdir())
    assert len(paths)==5*n and all(p.is_file() and p.suffix=='.jpg' for p in paths)
    for p in paths:
        assert cv2.imread(str(p)) is not None,p
