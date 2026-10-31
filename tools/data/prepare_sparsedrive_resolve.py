"""Prepare full split camera manifests and train-only SparseDrive anchors."""
import sys,json,pickle,hashlib,bisect
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from infrauni.resolve.sparsedrive.runtime import setup
setup()
from sklearn.cluster import MiniBatchKMeans,KMeans
from detra_repro.data.resolve_camera import parse_calibration

RAW=Path('/home/kemove/devdata1/zrl/dataset/RESOLVE')
CACHE=ROOT/'data/resolve_cache'
OUT=ROOT/'data/resolve_sparsedrive'
NAMES=('car','truck','construction_vehicle','bus','trailer','barrier','motorcycle','bicycle','pedestrian','traffic_cone')
CAMERAS=('axis1','axis2','axis3','axis4')

def timestamp(stem):
    sec,ns=stem.split('-');return int(sec)+int(ns)*1e-9

def main():
    OUT.mkdir(exist_ok=True)
    splitfile=ROOT/'third_party/MC-DeTra/splits/resolve_forecast_split.json'
    split=json.loads(splitfile.read_text())
    report={'split_sha256':hashlib.sha256(splitfile.read_bytes()).hexdigest(),'camera_order':CAMERAS,'splits':{},'image_selection_counts':{},'max_image_delta_s':0.,'missing_views':[],'calibration_pair_counts':{},'varying_camera_calibrations':[],'missing':[],'input_modalities':['camera'],'forecast_steps':10,'forecast_hz':2,'forecast_modes':6,'anchor_source':'train only'}
    image_index={};calib_cache={};all_tracks={};train_centers=[];train_sizes=[];motions=[[] for _ in NAMES]
    for part in ('train','val','test'):
        with (ROOT/'data/resolve_protocol'/(part+'.pkl')).open('rb') as f:source=pickle.load(f)
        prepared=[]
        def load_record(info):
            sample=Path(info['cache_path'])
            with np.load(sample.parent/'cameras'/sample.name,allow_pickle=False) as z:camera={k:z[k] for k in z.files}
            with np.load(sample,allow_pickle=True) as z:data={k:z[k] for k in ['gt_boxes','gt_future','gt_future_mask','gt_track_ids','frame_timestamp_micros']}
            return camera,data
        pool=ThreadPoolExecutor(max_workers=8)
        for index,(info,loaded) in enumerate(zip(source,pool.map(load_record,source))):
            path=Path(info['cache_path']);scene=path.parent.name
            assert scene in split[part],path
            frame_index=int(path.stem.split('_')[-1]);camfile=path.parent/'cameras'/path.name
            bag=info['token'].rsplit('_',1)[0]
            images=[];projections=[];selection=[];deltas=[]
            with nullcontext(loaded[0]) as camera:
                assert str(camera['source_lidar'])==info['lidar_path']
                for j,name in enumerate(CAMERAS):
                    assert str(camera['camera_names'][j])==name
                    folder=RAW/bag/(name+'_sync');reference=str(camera['json_image_stem'][j])
                    exact=folder/(reference+'.jpg')
                    mode='annotation_exact';delta=0.
                    # The raw calibration filename explicitly pairs an image
                    # with this lidar frame. Prefer that supplied synchronization
                    # over conflicting annotation image timestamps.
                    provided=folder/(str(camera['image_stem'][j])+'.jpg')
                    if provided.is_file():
                        exact=provided;mode='provided_image_lidar_pair';delta=timestamp(exact.stem)-timestamp(reference)
                    elif not str(camera['source_calibration'][j]):
                        candidates=list((RAW/bag/'calibration_txt'/(name+'__ouster128b')).glob('*_calibration.txt'))
                        if candidates:
                            target_lidar=timestamp(str(camera['lidar_stem']))
                            nearest_pair=min(candidates,key=lambda p:abs(timestamp(p.stem.split('__')[1][:-12])-target_lidar))
                            lidar_gap=abs(timestamp(nearest_pair.stem.split('__')[1][:-12])-target_lidar)
                            paired=folder/(nearest_pair.stem.split('__')[0]+'.jpg')
                            if lidar_gap<=.201 and paired.is_file():
                                exact=paired;mode='provided_nearest_lidar_pair';delta=timestamp(exact.stem)-timestamp(reference)
                    if not exact.is_file():
                        key=(bag,name)
                        if key not in image_index:
                            paths=sorted(folder.glob('*.jpg'),key=lambda p:timestamp(p.stem))
                            image_index[key]=([timestamp(p.stem) for p in paths],paths)
                        times,paths=image_index[key]
                        if not paths:
                            report['missing'].append({'token':info['token'],'camera':name,'reason':'no images'})
                            continue
                        target=timestamp(reference);slot=bisect.bisect_left(times,target)
                        candidates=[k for k in (slot-1,slot) if 0<=k<len(times)]
                        chosen=min(candidates,key=lambda k:abs(times[k]-target));exact=paths[chosen];delta=times[chosen]-target;mode='nearest_annotation_time'
                        if abs(delta)>.101:
                            report['missing_views'].append({'token':info['token'],'camera':name,'reason':'no synchronized image; view explicitly masked','delta_from_annotation_s':delta})
                            images.append(None);projections.append(np.zeros((4,4),np.float32));selection.append('missing_view_masked');deltas.append(None)
                            continue
                    # Extrinsics are associated with this lidar frame. Verify the
                    # chosen image's calibration agrees whenever it is provided.
                    k=camera['K'][j].astype(np.float64);t=camera['lidar2cam'][j].astype(np.float64)
                    cp=Path(str(camera['source_calibration'][j]))
                    folder_calib=RAW/bag/'calibration_txt'/(name+'__ouster128b')
                    if folder_calib not in calib_cache:
                        fs=sorted(folder_calib.glob('*_calibration.txt'))
                        if not fs:
                            calib_cache[folder_calib]=(fs,False)
                        else:
                            first=parse_calibration(fs[0]);last=parse_calibration(fs[-1])
                            calib_cache[folder_calib]=(fs,all(np.allclose(a,b) for a,b in zip(first,last)))
                    fs,constant=calib_cache[folder_calib]
                    if not fs:
                        report['missing'].append({'token':info['token'],'camera':name,'reason':'no calibration'})
                        continue
                    if not constant and str(folder_calib) not in report['varying_camera_calibrations']:
                        report['varying_camera_calibrations'].append(str(folder_calib))
                    exact_pair=folder_calib/(exact.stem+'__'+str(camera['lidar_stem'])+'_calibration.txt')
                    pair_mode='current_lidar_cached_calibration'
                    if exact_pair.is_file():
                        k,distortion,t=parse_calibration(exact_pair);pair_mode='exact_image_lidar_pair'
                    elif not cp.is_file():
                        target_lidar=timestamp(str(camera['lidar_stem']))
                        nearest=min(fs,key=lambda p:abs(timestamp(p.stem.split('__')[1][:-12])-target_lidar))
                        lidar_delta=abs(timestamp(nearest.stem.split('__')[1][:-12])-target_lidar)
                        if lidar_delta>.201:
                            report['missing'].append({'token':info['token'],'camera':name,'reason':'no nearby lidar calibration','delta_s':lidar_delta})
                            continue
                        k,distortion,t=parse_calibration(nearest);pair_mode='nearest_lidar_calibration_missing_cache'
                    report['calibration_pair_counts'][pair_mode]=report['calibration_pair_counts'].get(pair_mode,0)+1
                    if np.any(np.abs(camera['distortion'][j])>1e-10):
                        raise RuntimeError('Nonzero lens distortion requires image undistortion')
                    p=np.eye(4,dtype=np.float64);p[:3,:]=k@t[:3,:]
                    images.append(str(exact));projections.append(p.astype(np.float32));selection.append(mode);deltas.append(delta)
                    report['image_selection_counts'][mode]=report['image_selection_counts'].get(mode,0)+1
                    report['max_image_delta_s']=max(report['max_image_delta_s'],abs(delta))
            if len(images)!=4 or not any(v is not None for v in images):
                report['missing'].append({'token':info['token'],'reason':'invalid camera slot count or no valid camera'})
                continue
            with nullcontext(loaded[1]) as d:
                boxes=info['gt_boxes'].copy().astype(np.float32)
                assert np.array_equal(boxes[:,:7],d['gt_boxes'])
                futures=d['gt_future'].astype(np.float32);masks=d['gt_future_mask'].astype(bool)
                labels=np.asarray([NAMES.index(str(n)) for n in info['gt_names']],np.int64)
                ids=[]
                for name in d['gt_track_ids']:
                    name=str(name)
                    if name not in all_tracks:all_tracks[name]=len(all_tracks)
                    ids.append(all_tracks[name])
                entry=dict(token=info['token'],scene_token=scene,rosbag=bag,frame_index=frame_index,timestamp=float(d['frame_timestamp_micros'])/1e6,images=images,camera_valid=np.asarray([v is not None for v in images],dtype=bool),projection_mat=np.stack(projections),gt_boxes=boxes,gt_labels=labels,gt_track_ids=np.asarray(ids,np.int64),gt_future=futures,gt_future_mask=masks,image_selection=selection,image_delta_s=deltas)
            prepared.append(entry)
            if part=='train' and len(boxes):
                train_centers.append(boxes[:,:3]);train_sizes.append(boxes[:,3:6])
                full=masks.all(axis=1)
                for c in range(len(NAMES)):
                    select=full&(labels==c)
                    if not select.any():continue
                    rel=futures[select]-boxes[select,None,:2]
                    yaw=boxes[select,6];co=np.cos(yaw)[:,None];si=np.sin(yaw)[:,None]
                    agent=np.stack([rel[...,0]*co+rel[...,1]*si,-rel[...,0]*si+rel[...,1]*co],axis=-1)
                    motions[c].append(agent)
            if (index+1)%2000==0:print(part,index+1,flush=True)
        pool.shutdown()
        prepared.sort(key=lambda d:(d['scene_token'],d['frame_index']))
        report['splits'][part]={'expected_frames':len(source),'prepared_frames':len(prepared),'sequences':len({i['scene_token'] for i in prepared})}
        with (OUT/(part+'.pkl')).open('wb') as f:pickle.dump(prepared,f,protocol=4)
        print('SPLIT',part,report['splits'][part],flush=True)
    (OUT/'manifest.json').write_text(json.dumps(report,indent=2))
    if report['missing'] or any(v['expected_frames']!=v['prepared_frames'] for v in report['splits'].values()):
        raise RuntimeError('Camera coverage audit failed; full split is required. See '+str(OUT/'manifest.json'))
    centers=np.concatenate(train_centers)
    km=MiniBatchKMeans(n_clusters=900,random_state=20261028,n_init=3,batch_size=4096,max_iter=100).fit(centers)
    anchors=np.zeros((900,11),np.float32);anchors[:,:3]=km.cluster_centers_;anchors[:,3:6]=np.log(np.median(np.concatenate(train_sizes),axis=0));anchors[:,7]=1.
    np.save(OUT/'det_anchors.npy',anchors)
    clusters=[];counts=[]
    for c in range(len(NAMES)):
        data=np.concatenate(motions[c]) if motions[c] else np.zeros((0,10,2),np.float32);counts.append(len(data))
        if len(data)==0:raise RuntimeError('No full-horizon training trajectories for '+NAMES[c])
        k=min(6,len(data));model=KMeans(n_clusters=k,random_state=20261028,n_init=10).fit(data.reshape(len(data),20))
        cluster=model.cluster_centers_.reshape(k,10,2)
        if k<6:cluster=cluster[np.arange(6)%k]
        clusters.append(cluster)
    np.save(OUT/'motion_anchors.npy',np.asarray(clusters,np.float32))
    report['anchor_full_future_agents_per_class']=dict(zip(NAMES,counts));report['det_anchor_count']=900
    (OUT/'manifest.json').write_text(json.dumps(report,indent=2));print('PREPARATION_COMPLETE',flush=True)

if __name__=='__main__':main()
