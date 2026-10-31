"""Independently validate the published assets, exact pairs and model infos."""
import argparse,hashlib,json,pickle,re,time,zipfile
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import cv2
import numpy as np

CAMERAS=('axis1','axis2','axis3','axis4')

def parse_calibration(path):
    text=path.read_text()
    rr=re.search(r'R:\s*(.*?)\s*t:',text,re.S).group(1)
    tt=re.search(r't:\s*(.*?)\n',text).group(1)
    kk=re.search(r'Intrinsic:\s*\n([^\n]+)\n([^\n]+)\n([^\n]+)',text)
    dd=re.search(r'Distortion:\s*\n\[(.*?)\]',text,re.S)
    t=np.eye(4);t[:3,:3]=np.asarray([list(map(float,r.split())) for r in rr.strip().splitlines()]);t[:3,3]=list(map(float,tt.split()))
    k=np.asarray([list(map(float,r.split())) for r in kk.groups()]);d=np.zeros(4)
    if dd:
        v=np.fromstring(dd.group(1),sep=',');d[:min(4,len(v))]=v[:4]
    return k,d,t

def ns(stem):
    a,b=stem.split('-');return int(a)*10**9+int(b)

def hash_file(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(8<<20),b''):h.update(b)
    return h.hexdigest()

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--data-root',type=Path,required=True);ap.add_argument('--full-hash',action='store_true');args=ap.parse_args()
    root=args.data_root.resolve();manifest=json.loads((root/'manifest.json').read_text());split=json.loads((root/'splits.json').read_text());counts=Counter();started=time.time()
    def internal(path):
        p=Path(path);assert p.is_file(),str(p);assert p.resolve().is_relative_to(root),str(p);return p
    all_scenes=[];tokens=set();selected_images=set();box_lookup={};annotation_lookup={}
    for part in ('train','val','test'):
        scenes=split[part];all_scenes.extend(scenes)
        assert {p.name for p in (root/part).iterdir() if p.is_dir()}==set(scenes)
        for scene in scenes:
            assert scene.startswith('rosbag2_'),scene
            frames=json.loads((root/part/scene/'annotations.json').read_text())
            annotation_lookup[scene]={int(f['frameIndex']):f for f in frames}
            box_lookup[scene]={int(f['frameIndex']):{str(x['id']):np.asarray(x['points'][:2],np.float32) for x in f.get('labels',[]) if x.get('drawType')=='box3d'} for f in frames}
        with (root/'infos'/(part+'.pkl')).open('rb') as f:infos=pickle.load(f)
        assert len(infos)==manifest['splits'][part]['frames']
        actual_scene_frames=Counter()
        for index,info in enumerate(infos):
            scene=info['scene_token'];fi=int(info['frame_index']);actual_scene_frames[scene]+=1
            assert scene in scenes and info['token'] not in tokens;tokens.add(info['token'])
            lidar=internal(info['lidar_path']);cache=internal(info['cache_path']);camera=internal(info['camera_path'])
            assert lidar.parent.parent.name==scene and cache.parent.name==scene
            assert lidar.stat().st_size>0 and lidar.stat().st_size%16==0
            assert ns(lidar.stem)==int(info['timestamp_ns'])
            ann=annotation_lookup[scene][fi]
            assert ann['multimodal_eligible'] and ann['info']==str(lidar) and ann['imgInfo']==info['images']
            assert len(info['images'])==4 and bool(info['camera_valid'].all())
            with np.load(cache,allow_pickle=False) as z:
                assert str(z['source_tfrecord'])==str(lidar) and str(z['source_lidar'])==str(lidar)
                assert str(z['context_name'])==scene and int(z['frame_index'])==fi
                assert np.array_equal(z['gt_boxes'],info['gt_boxes'][:,:7])
                assert np.array_equal(z['gt_velocity'],info['gt_boxes'][:,7:])
                assert np.array_equal(z['gt_track_ids'],info['gt_track_ids'])
                for key in ('gt_future','gt_future_mask','gt_history','gt_history_mask'):
                    assert np.array_equal(z[key],info[key])
                for obj,tid in enumerate(z['gt_track_ids']):
                    track=str(tid).split('_',1)[1]
                    for xy,mask,offsets in ((info['gt_future'],info['gt_future_mask'],[(j+1)*5 for j in range(10)]),(info['gt_history'],info['gt_history_mask'],[(j-4)*5 for j in range(5)])):
                        for j,offset in enumerate(offsets):
                            if mask[obj,j]:assert np.array_equal(xy[obj,j],box_lookup[scene][fi+offset][track])
                counts['gt_boxes']+=len(z['gt_boxes'])
            with np.load(camera,allow_pickle=False) as z:
                assert tuple(z['camera_names'])==CAMERAS and bool(z['image_present'].all())
                assert str(z['lidar_stem'])==lidar.stem and str(z['source_lidar'])==str(lidar)
                assert list(z['source_image'])==info['images'] and list(z['source_calibration'])==info['calibration_paths']
                assert bool(z['json_matches_sync'].all()) and list(z['json_image_stem'])==list(z['image_stem'])
                for j,cam in enumerate(CAMERAS):
                    image=internal(info['images'][j]);cal=internal(info['calibration_paths'][j]);selected_images.add(image)
                    assert image.parent.name==cam+'_sync' and image.parent.parent.name==scene
                    assert cal.parent.name==cam+'__ouster128b'
                    assert cal.name==image.stem+'__'+lidar.stem+'_calibration.txt'
                    assert ns(image.stem)==int(z['image_timestamp_ns'][j])==int(info['image_timestamp_ns'][j])
                    k,d,t=parse_calibration(cal)
                    for a,b in ((k,z['K'][j]),(t,z['lidar2cam'][j]),(d,z['distortion'][j]),(k,info['K'][j]),(t,info['lidar2cam'][j])):assert np.array_equal(a,b),(camera,cam)
                    proj=np.eye(4);proj[:3,:]=k@t[:3,:]
                    assert np.array_equal(proj.astype(np.float32),info['projection_mat'][j])
                    counts['verified_exact_camera_pairs']+=1
            counts['frames']+=1
            if (index+1)%3000==0:print('CHECK_METADATA',part,index+1,flush=True)
        assert set(actual_scene_frames)==set(scenes),'A split contains a scene with no eligible examples'
    assert len(all_scenes)==len(set(all_scenes))==202
    assets=json.loads((root/'audit/raw_files.json').read_text());assert len(assets)==manifest['copied_raw_files']
    def check_asset(record):
        p=internal(root/record['path']);assert p.stat().st_size==record['bytes']
        assert not p.is_symlink(),'Raw assets must be independent files'
        if args.full_hash:assert hash_file(p)==record['sha256'],str(p)
        if p.suffix.lower()=='.jpg':
            im=cv2.imread(str(p));assert im is not None and im.shape[:2]==(1080,1920),str(p)
            return 'image'
        return 'raw'
    with ThreadPoolExecutor(max_workers=2) as pool:
        for i,result in enumerate(pool.map(check_asset,assets),1):
            counts['decoded_images' if result=='image' else 'other_raw_assets']+=1
            if i%20000==0:print('CHECK_RAW_ASSETS',i,'/',len(assets),flush=True)
    integrity=json.loads((root/'audit/point_cache_integrity.json').read_text())
    for i,record in enumerate(integrity,1):
        p=internal(root/record['path'])
        if args.full_hash:
            with zipfile.ZipFile(p) as z:
                member=z.getinfo('points.npy');z.fp.seek(member.header_offset);h=hashlib.sha256();left=record['points_member_span']
                while left:
                    block=z.fp.read(min(8<<20,left));assert block;h.update(block);left-=len(block)
                assert h.hexdigest()==record['points_member_sha256'],str(p)
        counts['point_cache_members_verified']+=1
        if i%3000==0:print('CHECK_POINT_MEMBERS',i,'/',len(integrity),flush=True)
    coverage=json.loads((root/'audit/image_coverage.json').read_text())
    assert sum(r['copied_images'] for r in coverage)==counts['decoded_images']
    assert all(r['silently_missing_images']==0 for r in coverage)
    report=dict(status='passed',dataset_version=manifest['version'],whole_scene_split_disjoint=True,all_training_paths_inside_dataset=True,all_four_views_present=True,calibration_matrix_mismatches=0,canonical_annotation_pair_mismatches=0,physical_synchronization_verified=False,full_raw_hashes_checked=args.full_hash,**counts,elapsed_s=round(time.time()-started))
    (root/'audit/validation.json').write_text(json.dumps(report,indent=2))
    print('VALIDATED',json.dumps(report),flush=True)

if __name__=='__main__':main()
