"""Actual image overlays and independent projection arithmetic checks."""
from pathlib import Path
import pickle,json,cv2,numpy as np
R=Path('/home/kemove/devdata1/zrl/InfraUni');O=R/'experiments/resolve_roi_v2_20261029/checks/cmt_gap/overlays';O.mkdir(exist_ok=True)
a=pickle.load(open(R/'data/resolve_multimodal/infos/val.pkl','rb'));ids=np.linspace(0,len(a)-1,10,dtype=int);edges=[(0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),(0,4),(1,5),(2,6),(3,7)];pix=[]
sign=np.array([[-1,-1,-1],[1,-1,-1],[1,1,-1],[-1,1,-1],[-1,-1,1],[1,-1,1],[1,1,1],[-1,1,1]])/2
for j,i in enumerate(ids):
 d=a[i];b=d['gt_boxes'];corners=[]
 for v in b:
  p=sign*v[3:6];s,c=np.sin(v[6]),np.cos(v[6]);rot=np.array([[c,-s,0],[s,c,0],[0,0,1]]);corners.append(p@rot.T+v[:3])
 canv=[]
 for v in range(4):
  im=cv2.imread(d['images'][v]);assert im.shape[:2]==(1080,1920);p=np.concatenate(corners) if corners else np.empty((0,3));ph=np.column_stack([p,np.ones(len(p))]);q=ph@d['projection_mat'][v].T;q2=(ph@d['lidar2cam'][v].T)[:,:3]@d['K'][v].T;good=(q[:,2]>.5)&(q2[:,2]>.5)
  if good.any():pix.extend(np.linalg.norm(q[good,:2]/q[good,2:3]-q2[good,:2]/q2[good,2:3],axis=1).tolist())
  for p in corners:
   q=np.column_stack([p,np.ones(8)])@d['projection_mat'][v].T;uv=q[:,:2]/np.maximum(q[:,2:3],1e-6)
   for x,y in edges:
    if q[x,2]>.5 and q[y,2]>.5:
     p0=tuple(np.clip(uv[x],-10000,10000).astype(int));p1=tuple(np.clip(uv[y],-10000,10000).astype(int));ok,p0,p1=cv2.clipLine((0,0,1920,1080),p0,p1)
     if ok:cv2.line(im,p0,p1,(0,0,255),2,cv2.LINE_AA)
  cv2.putText(im,d['camera_names'][v],(20,40),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,0),2);canv.append(cv2.resize(im,(960,540)))
 out=np.vstack([np.hstack(canv[:2]),np.hstack(canv[2:])]);cv2.imwrite(str(O/f'pair_{j:02d}_idx{i}.jpg'),out,[cv2.IMWRITE_JPEG_QUALITY,85])
r={'frames':ids.tolist(),'corners_projection_pixel_error_quantiles':np.quantile(pix,[.5,.95,1]).tolist(),'note':'Arithmetic consistency does not establish physical calibration truth. Inspect actual red GT corners on released images.'};(O.parent/'projection_overlay_audit.json').write_text(json.dumps(r,indent=2));print(r)
