"""Regression checks for extreme, unused projection/size derivatives."""
import sys,json
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from infrauni.resolve.sparsedrive.runtime import setup
setup()
from infrauni.resolve.sparsedrive.models import ResolveFeatureAggregation,ResolveBox3DKeyPointsGenerator
from projects.mmdet3d_plugin.models.blocks import DeformableFeatureAggregation
from projects.mmdet3d_plugin.models.detection3d.detection3d_blocks import SparseBox3DKeyPointsGenerator

p=torch.eye(4).reshape(1,1,4,4)
wh=torch.ones(1,1,2)
normal=torch.tensor([[[[.2,.3,1.]]]],requires_grad=True)
native=DeformableFeatureAggregation.project_points(normal,p,wh)
fixed=ResolveFeatureAggregation.project_points(normal,p,wh)
assert torch.equal(native,fixed)
gn=torch.autograd.grad(native.sum(),normal,retain_graph=True)[0]
gf=torch.autograd.grad(fixed.sum(),normal)[0]
assert torch.equal(gn,gf)

extreme=torch.tensor([[[[1e32,1e32,1e-4]]]],requires_grad=True)
native=DeformableFeatureAggregation.project_points(extreme,p,wh)
assert torch.isfinite(native).all()
# CUDA ignores outside-image samples, represented here by a zero output.
valid=torch.isfinite(native)&(native>0)&(native<1)
zero=torch.where(valid,native,torch.zeros_like(native)).sum()
gn=torch.autograd.grad(zero,extreme)[0]
fixed=ResolveFeatureAggregation.project_points(extreme,p,wh)
gf=torch.autograd.grad(fixed.sum(),extreme)[0]
assert torch.isfinite(zero) and not torch.isfinite(gn).all()
assert torch.isfinite(fixed).all() and torch.isfinite(gf).all() and (fixed==-1).all()

bad=torch.zeros(1,1,11,requires_grad=True)
with torch.no_grad():bad[...,3:6]=100.;bad[...,7]=1.
old=SparseBox3DKeyPointsGenerator(fix_scale=[[1.,0.,0.]])
new=ResolveBox3DKeyPointsGenerator(fix_scale=[[1.,0.,0.]])
out=old(bad);old_zero=torch.where(torch.isfinite(out),out,torch.zeros_like(out)).sum()
g_old=torch.autograd.grad(old_zero,bad)[0]
out=new(bad);g_new=torch.autograd.grad(out.sum(),bad)[0]
assert not torch.isfinite(g_old).all()
assert torch.isfinite(out).all() and torch.isfinite(g_new).all()

report=dict(visible_projection_and_gradient_unchanged=True,old_unused_projection_finite_loss_bad_gradient=True,fixed_unused_projection_finite_gradient=True,old_expansion_bad_gradient=True,fixed_expansion_finite_gradient=True)
target=ROOT/'experiments/sparsedrive_camera_resolve_20261028/diagnostics/numerical_regression.json'
target.parent.mkdir(parents=True,exist_ok=True);target.write_text(json.dumps(report,indent=2))
print('NUMERICAL_REGRESSION_PASSED',report,flush=True)
