"""Evaluate a SparseDrive checkpoint on the complete shared val/test split."""
import sys,argparse,json,logging,importlib.util
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from infrauni.resolve.sparsedrive.runtime import setup
setup()
spec=importlib.util.spec_from_file_location('_infrauni_sparsedrive_train_helpers',ROOT/'tools/train/train_sparsedrive_resolve.py')
helpers=importlib.util.module_from_spec(spec);spec.loader.exec_module(helpers)
build_model,validate=helpers.build_model,helpers.validate
from infrauni.resolve.sparsedrive.data import ResolveCameraDataset

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--split',choices=['val','test'],default='val')
    p.add_argument('--infos',type=Path,help='Optional explicit info pickle; default is the saved SparseDrive training protocol')
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--config',type=Path,default=ROOT/'configs/resolve/sparsedrive_camera.py')
    p.add_argument('--prediction-cache',type=Path,help='Optionally save canonical predictions for visualization')
    p.add_argument('--sample-manifest',type=Path,help='Save only these sample tokens in the optional prediction cache')
    a=p.parse_args();state=torch.load(str(a.checkpoint),map_location='cpu');stage=int(state['stage'])
    cfg,model=build_model(a.config,stage);model.load_state_dict(state['state_dict'],strict=True)
    info_path=a.infos if a.infos is not None else ROOT/'data/resolve_sparsedrive'/(a.split+'.pkl')
    data=ResolveCameraDataset(info_path,cfg.input_size,False)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    tokens=None if a.sample_manifest is None else {s['token'] for s in json.loads(a.sample_manifest.read_text())['samples']}
    result=validate(model,data,stage==2,a.output,a.prediction_cache,tokens)
    print(json.dumps({'split':a.split,'infos':str(info_path),'stage':stage,'frames':len(data),'mAP':result['detection']['mAP'],'NDS':result['detection']['NDS'],'forecast':result.get('forecast')},indent=2))

if __name__=='__main__':main()
