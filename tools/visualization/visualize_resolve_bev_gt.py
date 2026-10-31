#!/usr/bin/env python3
"""Clean RESOLVE BEV: white background, height-colored points, red GT outlines.

Audits all train/val/test GT corners before selecting a compact covering ROI.
Renders the same ten comparison samples. Only axes, points and box outlines
are drawn. This script does not edit any model training configuration.
"""
from __future__ import annotations
import argparse,json,math,pickle
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.patches import Polygon
from PIL import Image,ImageOps
ROOT=Path(__file__).resolve().parents[2]

def corners(boxes):
    b=np.asarray(boxes,dtype=np.float64)
    local=b[:,None,3:5]*np.array([[-.5,-.5],[.5,-.5],[.5,.5],[-.5,.5]])
    c,s=np.cos(b[:,6]),np.sin(b[:,6])
    rot=np.stack([c,-s,s,c],1).reshape(-1,2,2)
    return np.einsum('nij,nkj->nki',rot,local)+b[:,None,:2]

def bounds(boxes):
    b=np.asarray(boxes,dtype=np.float64);xy=corners(b)
    return np.r_[xy.min((0,1)),np.min(b[:,2]-b[:,5]/2),xy.max((0,1)),np.max(b[:,2]+b[:,5]/2)]

def coverage(boxes,roi):
    b=np.asarray(boxes);xy=corners(b)
    center=((b[:,:2]>=roi[:2])&(b[:,:2]<=roi[3:5])).all(1)
    footprint=((xy>=roi[:2])&(xy<=roi[3:5])).all((1,2))
    height=(b[:,2]-b[:,5]/2>=roi[2])&(b[:,2]+b[:,5]/2<=roi[5])
    return dict(boxes=len(b),centers_outside_xy=int((~center).sum()),footprints_outside_xy=int((~footprint).sum()),boxes_outside_xyz=int((~(footprint&height)).sum()))

def infos(root,split):
    with (root/'infos'/(split+'.pkl')).open('rb') as f:return pickle.load(f)

def recommend_roi(extent,xy_margin=5.,voxel_xy=.3):
    # A square multiple of 16 voxels supports stride-8 features and a further
    # stride-2 BEV block without shape mismatches; add at least 5 m in XY.
    lo,hi=extent[:3],extent[3:];unit=16*voxel_xy
    side=math.ceil(float(np.max(hi[:2]-lo[:2]+2*xy_margin))/unit)*unit
    for _ in range(100):
        lower=np.floor(((lo[:2]+hi[:2])/2-side/2)/unit+1e-9)*unit
        upper=lower+side
        if (lower<=lo[:2]-xy_margin).all() and (upper>=hi[:2]+xy_margin).all():break
        side+=unit
    else:raise RuntimeError('Cannot find covering voxel-aligned square')
    zlo=math.floor((lo[2]-1.)/2)*2
    zhi=math.ceil((hi[2]+.5)/2)*2
    return np.round(np.r_[lower,zlo,upper,zhi],6)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-root',type=Path,default=ROOT/'data/resolve_multimodal')
    p.add_argument('--sample-manifest',type=Path,default=ROOT/'experiments/visualization/transfusion_l_20261028_105558/manifest.json')
    p.add_argument('--output-dir',type=Path,default=ROOT/'visual/bev_gt')
    p.add_argument('--point-cloud-range',nargs=6,type=float,default=None,metavar=('XMIN','YMIN','ZMIN','XMAX','YMAX','ZMAX'))
    p.add_argument('--dpi',type=int,default=180)
    p.add_argument('--point-size',type=float,default=.9)
    p.add_argument('--alpha',type=float,default=.9)
    p.add_argument('--z-color-min',type=float,default=None)
    p.add_argument('--z-color-max',type=float,default=None)
    a=p.parse_args();out=a.output_dir.resolve();out.mkdir(parents=True,exist_ok=True)
    splits={};extents=[];count=0
    for split in ['train','val','test']:
        ds=infos(a.data_root,split);splits[split]=ds
        b=np.concatenate([d['gt_boxes'] for d in ds]);extent=bounds(b)
        assert np.isfinite(b).all() and (b[:,3:6]>0).all()
        extents.append(extent);count+=len(b)
        print('GT_BOUNDS',split,len(ds),len(b),extent.tolist(),flush=True)
    extent=np.r_[np.min([e[:3] for e in extents],0),np.max([e[3:] for e in extents],0)]
    recommended=recommend_roi(extent)
    roi=np.array(a.point_cloud_range,dtype=float) if a.point_cloud_range else recommended
    if not ((roi[:3]<roi[3:]).all() and (extent[:3]>=roi[:3]).all() and (extent[3:]<=roi[3:]).all()):
        raise ValueError('Requested range does not contain all dataset GT corners')
    print('ALL_GT',sum(len(ds) for ds in splits.values()),count,extent.tolist(),flush=True)
    print('RECOMMENDED_RANGE',recommended.tolist(),'ACTIVE_RANGE',roi.tolist(),flush=True)
    print('XY_GRID_AT_0.3M',((roi[3:5]-roi[:2])/.3).tolist(),flush=True)
    selected=json.loads(a.sample_manifest.read_text())['samples']
    assert len(selected)==10 and len({s['token'] for s in selected})==10
    lookup={d['token']:d for d in splits['val']};picked=[lookup[s['token']] for s in selected]
    assert all(d['scene_token']==s['rosbag'] and int(d['frame_index'])==int(s['frame_index']) for s,d in zip(selected,picked))
    clouds=[]
    for d in picked:
        pts=np.fromfile(d['lidar_path'],dtype=np.float32).reshape(-1,4)
        keep=np.isfinite(pts[:,:3]).all(1)&((pts[:,:3]>=roi[:3])&(pts[:,:3]<roi[3:])).all(1)
        clouds.append(pts[keep])
    z=np.concatenate([pts[:,2] for pts in clouds])
    vmin=float(np.percentile(z,15)) if a.z_color_min is None else a.z_color_min
    vmax=float(np.percentile(z,99)) if a.z_color_max is None else a.z_color_max
    if vmax<=vmin:raise ValueError('Invalid height color limits')
    norm=Normalize(vmin=vmin,vmax=vmax,clip=True)
    print('SHARED_HEIGHT_COLOR_LIMITS',vmin,vmax,flush=True)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':13,'figure.facecolor':'white','axes.facecolor':'white','axes.edgecolor':'black','xtick.color':'black','ytick.color':'black'})
    paths=[]
    for order,(row,d,pts) in enumerate(zip(selected,picked,clouds),1):
        boxes=np.asarray(d['gt_boxes']);xy=corners(boxes);stat=coverage(boxes,roi)
        assert stat['boxes_outside_xyz']==0
        pts=pts[np.argsort(pts[:,2],kind='stable')]
        fig,ax=plt.subplots(figsize=(8,8),facecolor='white')
        ax.scatter(pts[:,0],pts[:,1],c=pts[:,2],cmap='viridis',norm=norm,
                   s=a.point_size,alpha=a.alpha,marker='.',linewidths=0,rasterized=True)
        for polygon in xy:
            ax.add_patch(Polygon(polygon,closed=True,fill=False,edgecolor='red',linewidth=1.,zorder=3))
        ax.set_xlim(roi[0],roi[3]);ax.set_ylim(roi[1],roi[4]);ax.set_aspect('equal',adjustable='box')
        xt=np.arange(math.ceil(roi[0]/20)*20,roi[3]+1e-6,20)
        yt=np.arange(math.ceil(roi[1]/20)*20,roi[4]+1e-6,20)
        ax.set_xticks(np.unique(np.r_[roi[0],xt,roi[3]]))
        ax.set_yticks(np.unique(np.r_[roi[1],yt,roi[4]]))
        ax.tick_params(direction='out',length=4,width=.8,pad=3)
        for spine in ax.spines.values():spine.set_linewidth(1.)
        ax.grid(False)
        # Deliberately no title, legend, ROI border, origin, heading line,
        # object text, colorbar, axis-label text, or footer.
        assert len(ax.patches)==len(boxes) and len(ax.lines)==0
        assert not ax.texts and ax.get_legend() is None
        assert not ax.get_title() and not ax.get_xlabel() and not ax.get_ylabel()
        fig.subplots_adjust(left=.095,right=.975,bottom=.085,top=.965)
        path=out/('%02d_%s_f%06d_bev_gt.png'%(order,row['rosbag'],int(row['frame_index'])))
        fig.savefig(path,dpi=a.dpi,facecolor='white');plt.close(fig)
        with Image.open(path) as check:check.verify()
        paths.append(path)
        print('SAMPLE',order,row['token'],'POINTS',len(pts),'GT',len(boxes),'OUTSIDE',stat['boxes_outside_xyz'],flush=True)
    # Contact sheet contains only the ten rendered panels, without labels.
    tw=600;sheet=Image.new('RGB',(tw*2,tw*5),'white')
    for i,path in enumerate(paths):
        with Image.open(path) as im:sheet.paste(ImageOps.contain(im.convert('RGB'),(tw,tw),Image.Resampling.LANCZOS),((i%2)*tw,(i//2)*tw))
    sheet.save(out/'00_overview.jpg',quality=95,subsampling=0)
    print('DONE: 10 PNGs + overview in '+str(out),flush=True)

if __name__=='__main__':main()
