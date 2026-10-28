"""Check exact official curve parity and shared end-to-end behavior."""
import sys, importlib.util
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT),str(ROOT/'third_party/RESOLVE')]
sys.path.append('/home/kemove/anaconda3/envs/bevfusion/lib/python3.8/site-packages')
from infrauni.eval import resolve_detection as d
from infrauni.eval.resolve_forecast import evaluate_end_to_end
source=(ROOT/'third_party/RESOLVE/pcdet/datasets/carla/sunlakes_utils.py').read_text()
scope=dict(d.__dict__)
exec(source[source.index('def accumulate('):source.index('def load_prediction(')],scope)
gt={};pred={}
for i,name in enumerate(d.CLASS_NAMES):
    token=str(i)
    b=dict(translation=[0.,0.,-2.],size=[4.,2.,1.5],yaw=.3,velocity=[1.,0.],name=name,score=.9,future_xy=np.ones((10,2)),future_mask=np.ones(10,dtype=bool),mode_probs=[1.])
    gt[token]=[b];p=dict(b);p['translation']=[.2,0.,-2.];p['future_xy']=np.ones((1,10,2));pred[token]=[p]
    if i%2==0:
        fp=dict(p);fp['translation']=[20.,20.,-2.];fp['score']=.8;pred[token].append(fp)
a=d.evaluate_boxes(gt,pred)
saved=d.cal_metrics
d.cal_metrics=scope['cal_metrics']
b=d.evaluate_boxes(gt,pred);d.cal_metrics=saved
assert a['label_aps']==b['label_aps'] and a['label_tp_errors']==b['label_tp_errors']
assert abs(a['NDS']*9-a['NDS_legacy10']*10)<1e-10
f=evaluate_end_to_end(gt,pred)
assert abs(f['epa']-.75)<1e-10 and f['minade']==0. and f['minfde']==0.
pred['bad_token']=[]
try:d.evaluate_boxes(gt,pred)
except ValueError:pass
else:raise AssertionError('unknown frame must fail')
print('PASS: exact official AP/TP parity, NDS9/10 relation, EPA false positives, perfect futures, unknown-frame rejection')
