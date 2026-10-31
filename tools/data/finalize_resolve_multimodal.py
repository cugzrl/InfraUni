"""Finalize human-readable pair tables and future baseline entry points."""
import csv,hashlib,json,pickle,shutil
from collections import defaultdict
from pathlib import Path

def main():
    root=Path(__file__).resolve().parents[2];data=root/'data/resolve_multimodal'
    assert (data/'_READY').is_file()
    validation=json.loads((data/'audit/validation.json').read_text());assert validation['status']=='passed'
    manifest=json.loads((data/'manifest.json').read_text())
    assert (root/'data/resolve_protocol').resolve()==data/'infos'
    totals={}
    for part in ('train','val','test'):
        with (data/'infos'/(part+'.pkl')).open('rb') as f:infos=pickle.load(f)
        groups=defaultdict(list)
        for d in infos:groups[d['scene_token']].append(d)
        cache_manifest=dict(cache_schema_version=2,dataset=manifest['version'],split=part,samples=len(infos),class_names=manifest['class_names'],scene_names=sorted(groups),dataset_root=str(data),sample_pattern='*/sample_*.npz')
        (data/part/'manifest.json').write_text(json.dumps(cache_manifest,indent=2))
        for scene,records in groups.items():
            folder=data/part/scene;path=folder/'paired_frames.csv'
            with path.open('w',newline='',encoding='utf-8-sig') as handle:
                writer=csv.writer(handle);writer.writerow(['frame_index','lidar','axis1','axis2','axis3','axis4','calibration_axis1','calibration_axis2','calibration_axis3','calibration_axis4','sample_cache'])
                for d in records:writer.writerow([d['frame_index'],str(Path(d['lidar_path']).relative_to(folder)),*[str(Path(p).relative_to(folder)) for p in d['images']],*[str(Path(p).relative_to(folder)) for p in d['calibration_paths']],Path(d['cache_path']).name])
            with path.open(encoding='utf-8-sig',newline='') as handle:rows=list(csv.DictReader(handle))
            assert len(rows)==len(records)
            for row,d in zip(rows,records):
                assert int(row['frame_index'])==d['frame_index']
                assert folder/row['lidar']==Path(d['lidar_path'])
                for j,cam in enumerate(('axis1','axis2','axis3','axis4')):
                    assert folder/row[cam]==Path(d['images'][j])
                    assert folder/row['calibration_'+cam]==Path(d['calibration_paths'][j])
        totals[part]=dict(frames=len(infos),sequences=len(groups))
    provenance=data/'provenance'
    for relative in ['tools/data/prepare_resolve.py','tools/data/configure_resolve.py','tools/eval/evaluate_resolve.py','third_party/MC-DeTra/configs/resolve_mc_detra.json','configs/resolve/mcdetra.json']:
        src=root/relative
        if src.is_file():
            dst=provenance/(relative.replace('/','__')+'.before')
            if not dst.exists():shutil.copy2(src,dst)
    shutil.copy2(root/'tools/data/prepare_resolve_multimodal.py',root/'tools/data/prepare_resolve.py')
    shutil.copy2(root/'tools/eval/evaluate_resolve_multimodal.py',root/'tools/eval/evaluate_resolve.py')
    p=root/'tools/data/configure_resolve.py';text=p.read_text()
    text=text.replace('data/resolve_cache/','data/resolve_multimodal/').replace("experiments/mcdetra/resolve'","experiments/mcdetra/resolve_multimodal_v1'").replace('experiments/mcdetra/resolve/model.pt','experiments/mcdetra/resolve_multimodal_v1/model.pt');p.write_text(text)
    for relative in ['third_party/MC-DeTra/configs/resolve_mc_detra.json','configs/resolve/mcdetra.json']:
        p=root/relative
        if not p.is_file():continue
        cfg=json.loads(p.read_text());cfg.update(train_cache=str(data/'train'),val_cache=str(data/'val'),run_dir=str(root/'experiments/mcdetra/resolve_multimodal_v1'),checkpoint_path=str(root/'experiments/mcdetra/resolve_multimodal_v1/model.pt'))
        p.write_text(json.dumps(cfg,indent=2)+'\n')
    frozen=root/'experiments/data_preparation/resolve_multimodal_20261028/frozen_sparse_inputs.json'
    original=json.loads(frozen.read_text())
    for name,value in original.items():
        p=Path(name)
        assert p.stat().st_size==value['size'] and p.stat().st_mtime_ns==value['mtime_ns']
        assert hashlib.sha256(p.read_bytes()).hexdigest()==value['sha256']
    readme=data/'README.txt'
    note='\nHuman inspection: each scene/paired_frames.csv is the exact four-view table\nused by infos; frames.jsonl also lists excluded and unlabelled raw frames.\nPython access: infrauni.resolve.multimodal.load_infos/read_frame.\nMC-DeTra cache roots: resolve_multimodal/train and resolve_multimodal/val.\nresolve_protocol is a pointer to resolve_multimodal/infos.\n'
    if 'Human inspection:' not in readme.read_text():readme.write_text(readme.read_text()+note)
    report=dict(status='passed',splits=totals,paired_csv_matches_infos=True,sparsedrive_frozen_inputs_unchanged=True,protocol_target=str((root/'data/resolve_protocol').resolve()),future_mcdetra_defaults_updated=True)
    (data/'audit/final_entrypoints.json').write_text(json.dumps(report,indent=2))
    print('FINAL_ENTRYPOINTS',json.dumps(report))

if __name__=='__main__':main()
