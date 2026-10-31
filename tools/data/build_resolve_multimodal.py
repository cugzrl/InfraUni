"""Publish a self-contained RESOLVE dataset using exact released pair keys.

Raw data and running SparseDrive inputs are immutable. Whole-scene split and
shared detection/forecast targets are retained. All raw assets are copied;
supervised common-modality indices admit only frames with four exact pairs.
"""
import argparse,copy,csv,hashlib,io,json,os,pickle,shutil,struct,subprocess,sys,time,zipfile
from collections import Counter,defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np

CAMERAS=('axis1','axis2','axis3','axis4')
CLASSES=('car','truck','construction_vehicle','bus','trailer','barrier','motorcycle','bicycle','pedestrian','traffic_cone')
CLASS_MAP={'car':'car','truck':'truck','construction vehicle':'construction_vehicle','construction_vehicle':'construction_vehicle','bus':'bus','trailer':'trailer','van':'barrier','barrier':'barrier','motorcycle':'motorcycle','bicycle':'bicycle','pedestrian':'pedestrian','golf cart':'traffic_cone','traffic_cone':'traffic_cone','traffic cone':'traffic_cone'}
VERSION='resolve_multimodal_exact4_v1'

def timestamp_ns(stem):
    sec,nano=stem.split('-');return int(sec)*10**9+int(nano)

def sha256(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(8<<20),b''):h.update(chunk)
    return h.hexdigest()

def write_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+'.partial');tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2));os.replace(tmp,path)

def npy_bytes(value):
    b=io.BytesIO();np.lib.format.write_array(b,np.asarray(value),allow_pickle=False);return b.getvalue()

def rebase_npz(src,dst,new_lidar):
    """Copy unchanged ZIP members verbatim, replacing only provenance fields.

    Compressed points are retained bit-for-bit, avoiding lossy conversion and
    hours of recompression. ZipFile writes the final central directory.
    """
    patches={'source_tfrecord.npy':npy_bytes(str(new_lidar)),
             'source_lidar.npy':npy_bytes(str(new_lidar)),
             'dataset_version.npy':npy_bytes(VERSION)}
    dst.parent.mkdir(parents=True,exist_ok=True);tmp=dst.with_name(dst.name+'.partial')
    points_digest=None;points_span=None
    with zipfile.ZipFile(src,'r') as zin,zipfile.ZipFile(tmp,'w',allowZip64=True) as zout:
        entries=sorted(zin.infolist(),key=lambda z:z.header_offset)
        for k,entry in enumerate(entries):
            if entry.filename in patches:
                zout.writestr(entry.filename,patches.pop(entry.filename),compress_type=zipfile.ZIP_DEFLATED)
                continue
            end=entries[k+1].header_offset if k+1<len(entries) else zin.start_dir
            span=end-entry.header_offset;zin.fp.seek(entry.header_offset)
            cloned=copy.copy(entry);cloned.header_offset=zout.fp.tell()
            digest=hashlib.sha256() if entry.filename=='points.npy' else None
            remaining=span
            while remaining:
                block=zin.fp.read(min(8<<20,remaining));assert block,'Unexpected source ZIP EOF'
                zout.fp.write(block);remaining-=len(block)
                if digest:digest.update(block)
            zout.filelist.append(cloned);zout.NameToInfo[cloned.filename]=cloned
            zout.start_dir=zout.fp.tell();zout._didModify=True
            if digest:points_digest=digest.hexdigest();points_span=span
        for name,value in patches.items():zout.writestr(name,value,compress_type=zipfile.ZIP_DEFLATED)
    with zipfile.ZipFile(tmp) as z:
        assert len(z.namelist())==len(set(z.namelist()))
        assert 'points.npy' in z.namelist()
    os.replace(tmp,dst)
    return dict(points_member_sha256=points_digest,points_member_span=points_span)

def check_zip_copy_helper(directory):
    """Exercise mixed stored/compressed entries and Unicode scalar rebasing."""
    a=directory/'.zip_source_test.npz';b=directory/'.zip_target_test.npz'
    original=dict(points=np.arange(400,dtype=np.float32).reshape(100,4),source_tfrecord=np.asarray('old'),gt_boxes=np.zeros((0,7),np.float32),context_name=np.asarray('sequence'))
    np.savez_compressed(a,**original);rebase_npz(a,b,Path('/new/点云.bin'))
    with np.load(b,allow_pickle=False) as z:
        assert np.array_equal(z['points'],original['points'])
        assert str(z['source_tfrecord'])==str(Path('/new/点云.bin'))
        assert np.array_equal(z['gt_boxes'],original['gt_boxes'])
    with zipfile.ZipFile(b) as z:assert z.testzip() is None
    a.unlink();b.unlink()

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root',type=Path,default=Path('/home/kemove/devdata1/zrl/InfraUni'))
    ap.add_argument('--raw',type=Path,default=Path('/home/kemove/devdata1/zrl/dataset/RESOLVE'))
    ap.add_argument('--publish-protocol',action='store_true')
    args=ap.parse_args();root=args.root.resolve();raw=args.raw.resolve();out=root/'data/resolve_multimodal'
    assert out.parent.resolve()==root/'data'
    out.mkdir(exist_ok=True)
    assert not any(out.iterdir()),'Output must be empty; existing publications are never overwritten'
    for folder in ('infos','audit','provenance'):(out/folder).mkdir()
    started=time.time();write_json(out/'BUILD_STATUS.json',dict(status='building',phase='preflight',version=VERSION))
    check_zip_copy_helper(out)
    parser={};exec((root/'third_party/MC-DeTra/detra_repro/data/resolve_camera.py').read_text(),parser)
    parse_calibration=parser['parse_calibration']
    source_annotation=raw/'HSQJN_3.json';frames=json.loads(source_annotation.read_text())
    source_split=root/'third_party/MC-DeTra/splits/resolve_forecast_split.json';split=json.loads(source_split.read_text())
    assignment={batch:part for part in ('train','val','test') for batch in split[part]}
    assert len(assignment)==sum(len(split[p]) for p in ('train','val','test'))==202
    grouped=defaultdict(list);batch_by_bag={}
    for f in frames:
        bag=f['info'].split('/')[-3];batch=str(f['batchId'])
        assert bag not in batch_by_bag or batch_by_bag[bag]==batch
        batch_by_bag[bag]=batch;grouped[bag].append(f)
    assert set(batch_by_bag.values())==set(assignment)
    readable_split={part:sorted(b for b,batch in batch_by_bag.items() if assignment[batch]==part) for part in ('train','val','test')}
    write_json(out/'splits.json',dict(version=VERSION,unit='whole rosbag sequence',seed=split['seed'],source_split_sha256=sha256(source_split),**readable_split))
    shutil.copy2(source_split,out/'provenance/original_sequence_split.json')
    write_json(out/'provenance/scene_ids.json',batch_by_bag)
    original_protocol=root/'data/resolve_protocol'
    protocol_before={p.name:sha256(p) for p in original_protocol.iterdir() if p.is_file()}
    write_json(out/'provenance/previous_protocol_sha256.json',protocol_before)
    raw_records=[];point_records=[];excluded=[];conflicts=[];coverage=[]
    infos={p:[] for p in ('train','val','test')};counters=Counter();per_split={p:Counter() for p in infos};first_example={}
    old_cache=root/'data/resolve_cache'

    def copy_asset(job):
        src,dst=job;dst.parent.mkdir(parents=True,exist_ok=True)
        assert not dst.exists();h=hashlib.sha256();size=0
        with src.open('rb') as fi,dst.open('xb') as fo:
            for data in iter(lambda:fi.read(8<<20),b''):fo.write(data);h.update(data);size+=len(data)
        shutil.copystat(src,dst)
        assert src.stat().st_ino!=dst.stat().st_ino or src.stat().st_dev!=dst.stat().st_dev
        return dict(path=str(dst.relative_to(out)),source=str(src),bytes=size,sha256=h.hexdigest())

    for bag_no,bag in enumerate(sorted(grouped),1):
        batch=batch_by_bag[bag];part=assignment[batch];bd=out/part/bag;bd.mkdir(parents=True)
        original=raw/bag;jobs=[]
        for sub in ['128_128b_sync',*[c+'_sync' for c in CAMERAS],'calibration_txt']:
            for src in sorted((original/sub).rglob('*')):
                if src.is_file():jobs.append((src,bd/src.relative_to(original)))
        with ThreadPoolExecutor(max_workers=2) as pool:raw_records.extend(pool.map(copy_asset,jobs))
        pairs={};calib_records={}
        for cam in CAMERAS:
            files=sorted((bd/'calibration_txt'/(cam+'__ouster128b')).glob('*_calibration.txt'))
            for p in files:
                image,lidar=p.name[:-len('_calibration.txt')].split('__')
                key=(cam,lidar);assert key not in pairs,('Ambiguous released pair',bag,key)
                assert (bd/(cam+'_sync')/(image+'.jpg')).is_file()
                pairs[key]=(image,p)
        canonical_annotations=[];frame_rows=[];selected_images={c:set() for c in CAMERAS};seen_lidar=set()
        for f in sorted(grouped[bag],key=lambda d:int(d['frameIndex'])):
            idx=int(f['frameIndex']);li=Path(f['info']).stem;seen_lidar.add(li);lidar=bd/'128_128b_sync'/(li+'.bin')
            image_paths=[];cal_paths=[];image_stems=[];matrix=[];ks=[];ts=[];ds=[];missing=[]
            legacy={u.split('/')[-2].replace('_label',''):Path(u).stem for u in f.get('imgInfo',[])}
            for cam in CAMERAS:
                pair=pairs.get((cam,li))
                if pair is None:
                    missing.append(cam);image_paths.append(None);cal_paths.append(None);image_stems.append(None);continue
                im,cp=pair;image=bd/(cam+'_sync')/(im+'.jpg')
                assert image.is_file();image_paths.append(str(image));cal_paths.append(str(cp));image_stems.append(im)
                k,d,t=parse_calibration(cp);projection=np.eye(4);projection[:3,:]=k@t[:3,:]
                ks.append(k);ts.append(t);ds.append(d);matrix.append(projection.astype(np.float32))
                if legacy.get(cam)!=im:
                    conflicts.append(dict(sequence=bag,frame_index=idx,camera=cam,original_json_image_stem=legacy.get(cam),canonical_image_stem=im,source='released exact calibration pair'))
            reason='missing_lidar' if not lidar.is_file() else 'missing_exact_camera_pair' if missing else None
            row=dict(sequence=bag,frame_index=idx,lidar_stem=li,lidar_path=str(lidar) if lidar.is_file() else None,images=image_paths,calibrations=cal_paths,eligible=reason is None,excluded_reason=reason,missing_cameras=missing)
            frame_rows.append(row)
            canonical=copy.deepcopy(f);canonical['info']=str(lidar);canonical['imgInfo']=image_paths;canonical['sequence']=bag;canonical['multimodal_eligible']=reason is None
            canonical_annotations.append(canonical)
            if reason:
                excluded.append(row);per_split[part][reason]+=1;continue
            src=old_cache/part/batch/('sample_%06d.npz'%idx);assert src.is_file()
            dest=bd/src.name;point_integrity=rebase_npz(src,dest,lidar)
            point_records.append(dict(path=str(dest.relative_to(out)),**point_integrity))
            with np.load(dest,allow_pickle=False) as z:
                assert str(z['context_name'])==bag and int(z['frame_index'])==idx
                boxes=z['gt_boxes'];types=z['gt_types'];velocity=z['gt_velocity'];ids=z['gt_track_ids']
                labels={str(l['id']):l for l in f.get('labels',[]) if l.get('drawType')=='box3d'}
                for j,tid in enumerate(ids):
                    label=labels[str(tid)[len(batch)+1:]]
                    assert np.array_equal(boxes[j],np.asarray(label['points'],np.float32)[[0,1,2,6,7,8,5]])
                    assert CLASSES[int(types[j])]==CLASS_MAP[label['label']]
                gt_future=z['gt_future'];gt_future_mask=z['gt_future_mask'];gt_history=z['gt_history'];gt_history_mask=z['gt_history_mask'];point_counts=z['gt_num_lidar_points']
            cp=bd/'cameras'/src.name;cp.parent.mkdir(exist_ok=True)
            arrays=dict(camera_names=np.asarray(CAMERAS),image_present=np.ones(4,dtype=bool),image_stem=np.asarray(image_stems),image_timestamp_s=np.asarray([timestamp_ns(s)/1e9 for s in image_stems],np.float64),image_timestamp_ns=np.asarray([timestamp_ns(s) for s in image_stems],np.int64),lidar_stem=np.asarray(li),lidar_timestamp_s=np.asarray(timestamp_ns(li)/1e9,np.float64),lidar_timestamp_ns=np.asarray(timestamp_ns(li),np.int64),K=np.stack(ks),distortion=np.stack(ds),lidar2cam=np.stack(ts),image_hw=np.asarray([[1080,1920]]*4,np.int32),json_image_stem=np.asarray(image_stems),json_matches_sync=np.ones(4,dtype=bool),frame_index=np.asarray(idx,np.int32),batch_id=np.asarray(batch),rosbag=np.asarray(bag),split=np.asarray(part),source_image=np.asarray(image_paths),source_calibration=np.asarray(cal_paths),source_lidar=np.asarray(str(lidar)),extrinsic_frame=np.asarray('128_128b_sync unrotated'),pairing_policy=np.asarray('released_calibration_exact'),physical_time_residual_verified=np.asarray(False))
            np.savez(cp,**arrays)
            entry=dict(token=bag+'_'+str(idx),scene_token=bag,batch_id=batch,frame_index=idx,timestamp=timestamp_ns(li)/1e9,timestamp_ns=timestamp_ns(li),lidar_path=str(lidar),cache_path=str(dest),camera_path=str(cp),images=image_paths,camera_names=CAMERAS,camera_valid=np.ones(4,dtype=bool),projection_mat=np.stack(matrix),K=np.stack(ks),lidar2cam=np.stack(ts),distortion=np.stack(ds),calibration_paths=cal_paths,image_timestamp_ns=np.asarray([timestamp_ns(s) for s in image_stems],np.int64),gt_boxes=np.column_stack([boxes,velocity]).astype(np.float32),gt_names=np.asarray([CLASSES[int(v)] for v in types]),gt_labels=types,gt_track_ids=ids,num_lidar_pts=point_counts,gt_future=gt_future,gt_future_mask=gt_future_mask,gt_history=gt_history,gt_history_mask=gt_history_mask,pairing='released_calibration_exact',dataset_version=VERSION)
            infos[part].append(entry);per_split[part]['frames']+=1;per_split[part]['boxes']+=len(boxes)
            for cam,stem in zip(CAMERAS,image_stems):selected_images[cam].add(stem)
            first_example.setdefault(part,dict(sequence=bag,frame_index=idx))
        for lidar in sorted((bd/'128_128b_sync').glob('*.bin')):
            if lidar.stem not in seen_lidar:
                frame_rows.append(dict(sequence=bag,lidar_stem=lidar.stem,lidar_path=str(lidar),eligible=False,excluded_reason='unlabelled_lidar',images=[str(bd/(c+'_sync')/(pairs[(c,lidar.stem)][0]+'.jpg')) if (c,lidar.stem) in pairs else None for c in CAMERAS]))
                per_split[part]['unlabelled_lidar']+=1
        write_json(bd/'annotations.json',canonical_annotations)
        (bd/'frames.jsonl').write_text(''.join(json.dumps(r,separators=(',',':'))+'\n' for r in frame_rows))
        with (bd/'paired_frames.csv').open('w',newline='',encoding='utf-8-sig') as handle:
            writer=csv.writer(handle);writer.writerow(['frame_index','lidar','axis1','axis2','axis3','axis4','calibration_axis1','calibration_axis2','calibration_axis3','calibration_axis4','sample_cache'])
            for row in frame_rows:
                if row['eligible']:writer.writerow([row['frame_index'],str(Path(row['lidar_path']).relative_to(bd)),*[str(Path(p).relative_to(bd)) for p in row['images']],*[str(Path(p).relative_to(bd)) for p in row['calibrations']],'sample_%06d.npz'%row['frame_index']])
        for cam in CAMERAS:
            all_images={p.stem for p in (bd/(cam+'_sync')).glob('*.jpg')}
            coverage.append(dict(sequence=bag,camera=cam,copied_images=len(all_images),selected_unique_images=len(selected_images[cam]),preserved_unselected_images=len(all_images-selected_images[cam]),silently_missing_images=0))
        if bag_no%10==0 or bag_no==len(grouped):
            print('BUILD_SEQUENCE',bag_no,'/',len(grouped),'samples',sum(len(x) for x in infos.values()),'elapsed_s',round(time.time()-started),flush=True)
            write_json(out/'BUILD_STATUS.json',dict(status='building',phase='copy_and_pair',sequences=bag_no,total_sequences=len(grouped)))
    for part,records in infos.items():
        records.sort(key=lambda d:(d['scene_token'],d['frame_index']))
        with (out/'infos'/(part+'.pkl')).open('wb') as f:pickle.dump(records,f,protocol=4)
        write_json(out/part/'manifest.json',dict(cache_schema_version=2,dataset=VERSION,split=part,samples=len(records),class_names=CLASSES,scene_names=readable_split[part],dataset_root=str(out),sample_pattern='*/sample_*.npz'))
    write_json(out/'audit/raw_files.json',raw_records)
    write_json(out/'audit/point_cache_integrity.json',point_records)
    write_json(out/'audit/excluded_annotated_frames.json',excluded)
    write_json(out/'audit/image_coverage.json',coverage)
    (out/'audit/original_json_pair_conflicts.jsonl').write_text(''.join(json.dumps(v,separators=(',',':'))+'\n' for v in conflicts))
    manifest=dict(version=VERSION,dataset_root=str(out),modalities=['lidar','axis1','axis2','axis3','axis4'],splits={p:dict(sequences=len(readable_split[p]),**per_split[p]) for p in infos},source_annotation_sha256=sha256(source_annotation),source_split_sha256=sha256(source_split),pairing_policy='Only exact released image__lidar calibration pairs; all four views required; no nearest-neighbor fallback',canonical_annotations='Per-sequence annotations.json uses exactly the same images as infos; original JSON alternatives are audit-only',physical_synchronization_verified=False,timestamp_note='Original filename timestamps retained in int64 ns. Clock-domain offsets remain uncalibrated; filename deltas must not be used as physical motion-compensation delays.',point_features_raw=['x','y','z','intensity'],point_features_cache=['x','y','z','relative_time_s'],point_cache_history_sweeps=5,point_frame='unrotated 128_128b_sync',forecast=dict(steps=10,hz=2,seconds=5),class_names=CLASSES,class_aliases={'van':'barrier','golf cart':'traffic_cone'},gt_policy='Preserved verified shared GT: nonempty lidar boxes; train centers in ROI; val/test range unchanged',copied_raw_files=len(raw_records),copied_raw_bytes=sum(r['bytes'] for r in raw_records),cache_frames=len(point_records),legacy_json_conflict_records=len(conflicts),old_training_inputs_unchanged=['data/resolve_cache','data/resolve_sparsedrive',str(raw)],examples=first_example)
    write_json(out/'manifest.json',manifest);write_json(out/'infos/manifest.json',manifest)
    readme='RESOLVE multimodal exact-pair dataset\n\nTrain/val/test split by whole rosbag sequence, preserving the existing split.\nRead infos/train.pkl, val.pkl and test.pkl for identical frames across modalities.\nEach scene has 128_128b_sync/, axis1_sync/ ... axis4_sync/, calibration_txt/,\nannotations.json, frames.jsonl, sample_XXXXXX.npz and cameras/sample_XXXXXX.npz.\nRaw assets are independent copies. All filenames and original timestamps are retained.\nOnly labeled frames with lidar plus four exact released camera/calibration pairs\nenter the common benchmark. audit/excluded_annotated_frames.json lists exclusions.\nAll other images and lidar files remain available in the scene folders.\nCalibration arrays are parsed per frame. No nearest-frame substitution is used.\nThe published pairs are reproducible associations; physical exposure/scan\nsynchronization is unverified. Do not use filename clock offsets as physical delays.\nThe currently running SparseDrive remains on its frozen, older manifest.\nNew validation samples differ slightly; reevaluate old models on this version\nbefore comparing with newly trained models.\n'
    (out/'README.txt').write_text(readme)
    write_json(out/'BUILD_STATUS.json',dict(status='validating',phase='full_integrity'))
    subprocess.run([sys.executable,'-B',str(root/'tools/checks/check_resolve_multimodal.py'),'--data-root',str(out),'--full-hash'],check=True)
    if args.publish_protocol:
        assert {p.name:sha256(p) for p in original_protocol.iterdir() if p.is_file()}==protocol_before,'Protocol changed during build; refusing to replace it'
        stamp=time.strftime('%Y%m%d_%H%M%S');backup=root/'data'/('resolve_protocol_backup_'+stamp)
        temp_link=root/'data'/('.resolve_protocol_'+stamp)
        assert not backup.exists() and not temp_link.exists();temp_link.symlink_to(out/'infos',target_is_directory=True)
        os.replace(original_protocol,backup)
        try:os.replace(temp_link,original_protocol)
        except BaseException:
            os.replace(backup,original_protocol);raise
        manifest['previous_protocol_backup']=str(backup);manifest['active_protocol']=str(original_protocol)
        write_json(out/'manifest.json',manifest);write_json(out/'infos/manifest.json',manifest)
    write_json(out/'BUILD_STATUS.json',dict(status='ready',elapsed_s=round(time.time()-started),frames=sum(len(x) for x in infos.values())))
    (out/'_READY').write_text(VERSION+'\n')
    print('READY',json.dumps(manifest['splits']),flush=True)

if __name__=='__main__':main()
