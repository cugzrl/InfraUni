"""Evaluate predictions against the unified RESOLVE data version."""
import sys,json,pickle,argparse
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
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
    p.add_argument('--data-root',type=Path,default=ROOT/'data/resolve_multimodal')
    p.add_argument('--manifest',type=Path)
    a=p.parse_args();data=a.data_root.resolve()
    if data.name=='resolve_multimodal' and not (data/'_READY').is_file():raise RuntimeError('Dataset publication is incomplete')
    mp=a.manifest or data/'manifest.json';manifest=json.loads(mp.read_text())
    gt=ground_truth(data/a.split)
    if a.predictions.suffix=='.pkl':
        with a.predictions.open('rb') as f:annos=pickle.load(f)
        pred={}
        for row in annos:
            token=row['metadata']['token'];pred[token]=[]
            for b,n,s in zip(row['boxes_lidar'],row['name'],row['score']):pred[token].append(dict(translation=b[:3].tolist(),size=b[3:6].tolist(),yaw=float(b[6]),velocity=b[7:9].tolist(),name=str(n),score=float(s)))
    else:pred=json.loads(a.predictions.read_text())
    if set(gt)!=set(pred):raise ValueError('Predictions must cover exactly this data version and split, including empty frames')
    result=dict(split=a.split,dataset_root=str(data),detection=evaluate_boxes(gt,pred),split_manifest=manifest)
    if a.forecast:result['forecast']=evaluate_end_to_end(gt,pred)
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,indent=2))
    print(json.dumps(result['detection'],indent=2))

if __name__=='__main__':main()
