"""Inspect exact four-camera RESOLVE pairs from the unified dataset only."""
import argparse,sys
from pathlib import Path
import cv2,numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'third_party/MC-DeTra'))
from infrauni.resolve.multimodal import load_infos
from tools.visualization.visualize_resolve_cameras import draw_projection,_mosaic

def main():
    p=argparse.ArgumentParser();p.add_argument('--split',choices=['train','val','test'],default='val');p.add_argument('--sequence');p.add_argument('--frame-index',type=int);p.add_argument('--output',type=Path,default=ROOT/'visual/resolve_multimodal');a=p.parse_args()
    infos=load_infos(a.split)
    if a.sequence:infos=[d for d in infos if d['scene_token']==a.sequence]
    if a.frame_index is not None:infos=[d for d in infos if d['frame_index']==a.frame_index]
    if not infos:raise ValueError('No eligible frame matches this selection')
    info=infos[0] if a.frame_index is not None else max(infos,key=lambda d:len(d['gt_boxes']))
    points=np.fromfile(info['lidar_path'],np.float32).reshape(-1,4)[:,:3]
    a.output.mkdir(parents=True,exist_ok=True);panels=[];stem=a.split+'_'+info['scene_token']+'_f%06d'%info['frame_index']
    for j,name in enumerate(info['camera_names']):
        image=cv2.imread(info['images'][j]);assert image is not None
        canvas,counts=draw_projection(image,points,info['gt_boxes'][:,:7],[str(x) for x in info['gt_names']],info['lidar2cam'][j],info['K'][j],info['distortion'][j],80.)
        path=a.output/(stem+'_'+name+'.jpg');assert cv2.imwrite(str(path),canvas);panels.append(canvas)
    overview=a.output/(stem+'_overview.jpg');assert cv2.imwrite(str(overview),_mosaic(panels))
    print('SAVED',str(overview));print('LIDAR',info['lidar_path']);print('IMAGES',info['images'])

if __name__=='__main__':main()
