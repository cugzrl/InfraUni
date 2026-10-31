"""Load the vendored implementation using an existing conda environment."""
import os
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[3]
VENDOR=ROOT/'third_party/SparseDrive'

def setup():
    os.environ['INFRAUNI_SPARSEDRIVE_MINIMAL']='1'
    for p in [ROOT,VENDOR,ROOT/'third_party/MC-DeTra',VENDOR/'.runtime']:
        if str(p) in sys.path:sys.path.remove(str(p))
        sys.path.insert(0,str(p))

