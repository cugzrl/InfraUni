"""Generate lightweight detection infos from the existing MC-DeTra cache."""
import sys,json,pickle,hashlib
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
NAMES=('car','truck','construction_vehicle','bus','trailer','barrier','motorcycle','bicycle','pedestrian','traffic_cone')
cache=ROOT/'data/resolve_cache'
splitfile=ROOT/'third_party/MC-DeTra/splits/resolve_forecast_split.json'
split=json.loads(splitfile.read_text())
out=ROOT/'data/resolve_protocol';out.mkdir(exist_ok=True)
seen=set();report={'split_sha256':hashlib.sha256(splitfile.read_bytes()).hexdigest(),'splits':{}}
for part in ('train','val','test'):
    infos=[];scenes=set();tokens=set()
    for path in sorted((cache/part).glob('*/sample_*.npz')):
        assert path.parent.name in split[part],path
        scenes.add(path.parent.name)
        with np.load(path,allow_pickle=True) as d:
            token=str(d['context_name'])+'_'+str(int(d['frame_index']))
            assert token not in tokens;tokens.add(token)
            source=str(d['source_tfrecord'])
            raw=Path('/home/kemove/devdata1/zrl/dataset/RESOLVE')/source.split('data/RESOLVE/',1)[-1]
            assert raw.is_file(),raw
            boxes=np.concatenate((d['gt_boxes'].reshape(-1,7),d['gt_velocity'].reshape(-1,2)),axis=1).astype(np.float32)
            infos.append(dict(token=token,lidar_path=str(raw),gt_boxes=boxes,gt_names=np.asarray([NAMES[int(i)] for i in d['gt_types']]),num_lidar_pts=d['gt_num_lidar_points'],cache_path=str(path)))
    assert infos and not (seen & scenes);seen.update(scenes)
    with (out/(part+'.pkl')).open('wb') as f:pickle.dump(infos,f)
    report['splits'][part]={'frames':len(infos),'scenes':sorted(scenes),'boxes':sum(len(i['gt_boxes']) for i in infos)}
    print(part,len(infos),len(scenes),flush=True)
(out/'manifest.json').write_text(json.dumps(report,indent=2))
