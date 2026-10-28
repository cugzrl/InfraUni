"""Launch the vendored official TransFusion with InfraUni data and evaluation."""
import os,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
VENDOR=ROOT/'third_party/RESOLVE'
sys.path[:0]=[str(ROOT),str(VENDOR),str(VENDOR/'tools')]
sys.path.append('/home/kemove/anaconda3/envs/bevfusion/lib/python3.8/site-packages')
os.chdir(VENDOR/'tools')
import pcdet.datasets
from infrauni.resolve.dataset import ResolveSequenceDataset
pcdet.datasets.__all__['ResolveSequenceDataset']=ResolveSequenceDataset
import train
if __name__=='__main__': train.main()
