#!/home/kemove/anaconda3/envs/ccf/bin/python
"""Read-only status for the three canonical RESOLVE retraining jobs."""
from pathlib import Path
import json,os,re,subprocess,pickle,math
ROOT=Path(__file__).resolve().parents[2]
RUN=ROOT/'experiments/resolve_roi_v2_20261029'
TF=ROOT/'experiments/baselines/infrauni/transfusion_roi_v2_allframes/roi_v2_20261029'

def read_json(path):
    try:return json.loads(Path(path).read_text())
    except (FileNotFoundError,ValueError):return None

def alive(folder):
    p=folder/'train.pid'
    try:
        pid=int(p.read_text());q=Path('/proc')/str(pid)
        command=(q/'cmdline').read_bytes().replace(b'\0',b' ').decode()
        cwd=(q/'cwd').resolve()
        return {'pid':pid,'alive':str(ROOT) in str(cwd) and ('train' in command),'command':command}
    except (FileNotFoundError,ProcessLookupError,ValueError):return {'alive':False}

def text_tail(path,n=2000000):
    try:
        with Path(path).open('rb') as f:f.seek(max(0,Path(path).stat().st_size-n));return f.read().decode(errors='replace')
    except FileNotFoundError:return ''

def tf_gate():
    logs=sorted(TF.glob('train_*.log'));text=text_tail(logs[-1]) if logs else ''
    by_epoch={}
    for line in text.splitlines():
        m=re.search(r'Train:\s+(\d+)/(\d+).*Loss:\s+([\deE.+-]+)\s+\(([\deE.+-]+)\)',line)
        if m:by_epoch[int(m[1])]=float(m[4])
    means=[by_epoch[e] for e in sorted(by_epoch)[-3:]]
    metrics=list(TF.glob('eval/**/metrics_resolve.json'));metrics.sort(key=lambda p:p.stat().st_mtime)
    report=read_json(metrics[-1]) if metrics else None
    code=(RUN/'transfusion/exit_code').read_text().strip() if (RUN/'transfusion/exit_code').exists() else None
    problems=[]
    if code!='0':problems.append('training/evaluation has not completed successfully')
    if len(means)<3 or max(means)-min(means)>.2*max(1e-8,sum(means)/len(means)):problems.append('last three epoch losses require convergence review')
    if not report or report.get('protocol')!='resolve_official_curves_paper_nds9_v1' or report.get('mAP',0)<.05:problems.append('full shared evaluation absent or very low AP')
    if report:
        result=metrics[-1].parents[2]/'result.pkl'
        if not result.exists():
            candidates=list(metrics[-1].parents[2].glob('**/result.pkl'));result=candidates[0] if candidates else result
        try:
            predictions=pickle.load(result.open('rb'));truth=pickle.load((ROOT/'data/resolve_multimodal/infos/val.pkl').open('rb'))
            if len(predictions)!=len(truth) or {p['metadata']['token'] for p in predictions}!={p['token'] for p in truth}:problems.append('prediction tokens do not match complete canonical validation split')
        except (FileNotFoundError,KeyError):problems.append('full validation prediction coverage not verified')
    return {'ok':not problems,'problems':problems,'last_epoch_losses':means,'mAP':report.get('mAP') if report else None,'NDS':report.get('NDS') if report else None}

def status():
    result={'run':str(RUN),'protocol':read_json(RUN/'protocol.json'),'queue':read_json(RUN/'queue_status.json')}
    for name in ['transfusion','mc-detra','cmt']:
        folder=Path((read_json(RUN/'cmt_active_run.json') or {}).get('work_dir',str(RUN/name))) if name=='cmt' else RUN/name;item=alive(folder);item['work_dir']=str(folder);console=folder/('train.log' if name=='cmt' else 'console.log');text=text_tail(console,150000)
        item.update(log=str(console),mtime=console.stat().st_mtime if console.exists() else None,last_lines=text.replace('\r','\n').splitlines()[-8:])
        if name=='cmt':item['metrics']=read_json(folder/'val_metrics.json')
        if name=='mc-detra':item['metrics']=read_json(folder/'val/latest_metrics.json')
        if item.get('metrics'):item['metrics']={k:item['metrics'].get(k) for k in ['mAP','NDS','frames','protocol']}
        result[name]=item
    result['cmt_final_selection']=read_json(RUN/'cmt_final_selection.json')
    result['tf_transition_gate']=tf_gate()
    from check_bevfusion_roi_v2 import status as bevfusion_status
    result['bevfusion']=bevfusion_status()
    result['gpu']=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used,memory.total,utilization.gpu','--format=csv'],text=True)
    return result
if __name__=='__main__':print(json.dumps(status(),indent=2))
