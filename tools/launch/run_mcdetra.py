"""Run any MC-DeTra module with the shared InfraUni evaluation on sys.path."""
import sys,os,runpy
from pathlib import Path
root=Path(__file__).resolve().parents[2];vendor=root/'third_party/MC-DeTra'
sys.path[:0]=[str(root),str(vendor)]
sys.path.append('/home/kemove/anaconda3/envs/bevfusion/lib/python3.8/site-packages')
os.chdir(vendor)
module=sys.argv.pop(1)
runpy.run_module(module,run_name='__main__')
