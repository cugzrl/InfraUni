"""Read-only UniBEV process/log/checkpoint monitor; no CUDA allocation."""
from pathlib import Path
import json,time,os,subprocess
ROOT=Path(__file__).resolve().parents[2]
RUN=ROOT/'experiments/unibev_resolve_roi_v2_20261103'
def read(name):
 p=RUN/name
 return p.read_text().strip() if p.exists() else None
pid=read('train.pid');proc=Path('/proc')/(pid or '0')
cmd=(proc/'cmdline').read_bytes().replace(b'\0',b' ').decode() if (proc/'cmdline').exists() else None
progress=read('progress.json')
payload=dict(run=str(RUN),pid=pid,alive=bool(cmd and 'UniBEV' in cmd or cmd and 'train_resolve.py' in cmd),command=cmd,
 exit_code=read('exit_code'),progress=json.loads(progress) if progress else None,
 log_age_seconds=time.time()-(RUN/'console.log').stat().st_mtime if (RUN/'console.log').exists() else None,
 checkpoints=[str(p) for p in RUN.glob('*.pth')],validation=read('val_metrics.json'))
print(json.dumps(payload,indent=2))
