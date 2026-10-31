"""Check the unified RESOLVE protocol entry point without rebuilding legacy data."""
import json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from infrauni.resolve.multimodal import load_infos

def main():
    data=ROOT/'data/resolve_multimodal';protocol=ROOT/'data/resolve_protocol'
    if protocol.resolve()!=(data/'infos').resolve():raise RuntimeError('resolve_protocol must point to the validated resolve_multimodal/infos publication')
    for split in ('train','val','test'):
        infos=load_infos(split,data)
        assert all(Path(d['lidar_path']).is_relative_to(data) and Path(d['cache_path']).is_relative_to(data) for d in infos)
        print(split,len(infos),'frames',len({d['scene_token'] for d in infos}),'sequences')
    print('Protocol:',protocol,'->',protocol.resolve())

if __name__=='__main__':main()
