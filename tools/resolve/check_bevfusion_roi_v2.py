#!/home/kemove/anaconda3/envs/ccf/bin/python
"""Read-only BEVFusion training health for the shared RESOLVE monitor."""
from pathlib import Path
import json,datetime,re,time
ROOT=Path(__file__).resolve().parents[2]
RUN=ROOT/'experiments/bevfusion_resolve_roi_v2_20261030'
def read_json(path):
    try:return json.loads(Path(path).read_text())
    except (FileNotFoundError,ValueError):return None
def status():
    s={'work_dir':str(RUN),'expected_gpu':1,'protocol':read_json(RUN/'protocol.json')}
    try:
        pid=int((RUN/'train.pid').read_text());p=Path('/proc')/str(pid);cmd=(p/'cmdline').read_bytes().replace(b'\0',b' ').decode();env=(p/'environ').read_bytes().decode().split('\0');s.update(pid=pid,alive=(p/'cwd').resolve()==ROOT and 'tools/train/train_bevfusion_resolve.py' in cmd,command=cmd,python=str((p/'exe').resolve()),cuda_visible_devices=next((v.split('=',1)[1] for v in env if v.startswith('CUDA_VISIBLE_DEVICES=')),None))
    except (FileNotFoundError,ProcessLookupError,ValueError):s['alive']=False
    start=None
    try:s['started_at']=(RUN/'started_at').read_text().strip();start=datetime.datetime.fromisoformat(s['started_at']).timestamp()
    except (FileNotFoundError,ValueError):pass
    try:s['exit_code']=(RUN/'exit_code').read_text().strip()
    except FileNotFoundError:pass
    log=RUN/'train.log';s['log']=str(log);lines=[]
    if log.exists():
        with log.open('rb') as f:f.seek(max(0,log.stat().st_size-500000));lines=f.read().decode(errors='replace').splitlines()
        s['log_age_seconds']=round(time.time()-log.stat().st_mtime,1)
    current=[]
    for line in lines:
        if start and re.match(r'\d{4}-\d{2}-\d{2} ',line):
            try:
                if datetime.datetime.strptime(line[:23],'%Y-%m-%d %H:%M:%S,%f').timestamp()<start:continue
            except ValueError:pass
        current.append(line)
    s['last_lines']=current[-5:];s['numerical_error_lines']=[l for l in current if re.search(r'(?:loss|grad_preclip|grad_norm)\s*[:=]\s*(?:nan|inf)\b|numerical_failure',l,re.I)][-5:]
    checkpoints=sorted((RUN/'checkpoints').glob('*.pth'),key=lambda p:p.stat().st_mtime);s['checkpoints']=[dict(path=str(p),bytes=p.stat().st_size,mtime=p.stat().st_mtime) for p in checkpoints[-5:]]
    metrics=sorted(RUN.rglob('*metrics*.json'),key=lambda p:p.stat().st_mtime);s['metrics_file']=str(metrics[-1]) if metrics else None;s['metrics']=read_json(metrics[-1]) if metrics else None
    shared=read_json(ROOT/'experiments/resolve_roi_v2_20261029/protocol.json');proto=s['protocol'] or {};s['shared_protocol_alignment']=bool(shared and proto.get('point_cloud_range')==shared['range'] and proto.get('dataset',{}).get('dataset_root')==shared['dataset_root'] and all(proto.get('info_sha256',{}).get(k)==shared['splits'][k]['sha256'] for k in ['train','val']))
    return s
if __name__=='__main__':print(json.dumps(status(),indent=2))
