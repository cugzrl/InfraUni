"""Evaluate saved canonical JSON or OpenPCDet result.pkl on the shared split.

JSON: {token: [{translation:[x,y,z],size:[dx,dy,dz],yaw:radians,
velocity:[vx,vy],name:class,score:float,future_xy:[K,T,2],mode_probs:[K]}]}.
Future fields are required only with --forecast. Coordinates are raw lidar.
"""
import sys,json,pickle,argparse
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
sys.path.append('/home/kemove/anaconda3/envs/bevfusion/lib/python3.8/site-packages')
from infrauni.resolve.cache import ground_truth
from infrauni.eval.resolve_detection import evaluate_boxes
from infrauni.eval.resolve_forecast import evaluate_end_to_end

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--predictions',type=Path,required=True)
    p.add_argument('--split',choices=['val','test'],default='val')
    p.add_argument('--forecast',action='store_true')
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();gt=ground_truth(ROOT/'data/resolve_cache'/a.split)
    if a.predictions.suffix=='.pkl':
        with a.predictions.open('rb') as f:annos=pickle.load(f)
        pred={}
        for row in annos:
            token=row['metadata']['token'];pred[token]=[]
            for b,n,s in zip(row['boxes_lidar'],row['name'],row['score']):
                pred[token].append(dict(translation=b[:3].tolist(),size=b[3:6].tolist(),yaw=float(b[6]),velocity=b[7:9].tolist(),name=str(n),score=float(s)))
    else:pred=json.loads(a.predictions.read_text())
    if set(gt)!=set(pred):raise ValueError('Predictions must contain exactly all split tokens, including empty frames')
    result={'split':a.split,'detection':evaluate_boxes(gt,pred)}
    if a.forecast:result['forecast']=evaluate_end_to_end(gt,pred)
    result['split_manifest']=json.loads((ROOT/'data/resolve_protocol/manifest.json').read_text())
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,indent=2))
    print(json.dumps(result['detection'],indent=2))
if __name__=='__main__':main()
