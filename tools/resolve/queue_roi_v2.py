#!/home/kemove/anaconda3/envs/ccf/bin/python
"""Run MC-DeTra on GPU0 only after TransFusion passes completion/health checks."""
from pathlib import Path
import json,time,subprocess,os,fcntl
from check_roi_v2 import ROOT,RUN,alive,tf_gate,read_json

def save(phase,**kwargs):
    p=RUN/'queue_status.json';tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(dict(phase=phase,time=time.time(),**kwargs),indent=2));tmp.replace(p)

lock=(RUN/'queue.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
(RUN/'queue.pid').write_text(str(os.getpid()))
while alive(RUN/'transfusion')['alive']:
    save('transfusion_training');time.sleep(60)
gate=tf_gate()
if not gate['ok']:
    save('transfusion_needs_review',gate=gate);raise SystemExit(0)
if alive(RUN/'mc-detra')['alive']:
    save('mcdetra_training_already_running');raise SystemExit(0)
code=RUN/'mc-detra/exit_code'
if code.exists() and code.read_text().strip()=='0':
    save('mcdetra_finished_needs_convergence_review');raise SystemExit(0)
save('mcdetra_starting',transfusion_gate=gate)
with (RUN/'mc-detra/console.log').open('ab') as log:
    process=subprocess.Popen(['bash',str(ROOT/'tools/resolve/train_mcdetra_roi_v2.sh')],stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL)
    save('mcdetra_training',launcher_pid=process.pid,transfusion_gate=gate);rc=process.wait()
save('mcdetra_finished_needs_convergence_review' if rc==0 else 'mcdetra_failed',exit_code=rc)
