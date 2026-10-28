"""RESOLVE detector/forecaster components for InfraUni experiments.

Baseline implementations remain vendored; new experiment compositions belong
in this namespace. No parameter renaming or checkpoint conversion is applied.
"""
import sys
from pathlib import Path
VENDOR=Path(__file__).resolve().parents[2]/'third_party/MC-DeTra'
if str(VENDOR) not in sys.path:sys.path.insert(0,str(VENDOR))
from detra_repro.models.bev import PaperLikeLidarEncoder
from detra_repro.models.proposals import PaperProposalHead, ProposalHead
from detra_repro.models.refinement import DeTraMini, build_map_encoder

__all__=['PaperLikeLidarEncoder','PaperProposalHead','ProposalHead','DeTraMini','build_map_encoder']
