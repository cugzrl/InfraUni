"""Materialize a documented SparseDrive-S RESOLVE camera/motion configuration."""
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from infrauni.resolve.sparsedrive.runtime import setup
setup()
from mmcv import Config

def replace_flash(obj):
    if isinstance(obj,dict):
        if obj.get('type')=='MultiheadFlashAttention':obj['type']='MultiheadAttention'
        for value in obj.values():replace_flash(value)
    elif isinstance(obj,(tuple,list)):
        for value in obj:replace_flash(value)

def main():
    base=Config.fromfile(str(ROOT/'third_party/SparseDrive/projects/configs/sparsedrive_small_stage2.py'))
    model=base.model
    model.type='ResolveSparseDrive';model.depth_branch=None
    model.img_backbone.pretrained=str(ROOT/'third_party/SparseDrive/ckpt/resnet50-0676ba61.pth')
    model.head.type='ResolveSparseDriveHead'
    model.head.task_config=dict(with_det=True,with_map=False,with_motion_plan=True)
    model.head.pop('map_head',None)
    det=model.head.det_head
    det.instance_bank.anchor=str(ROOT/'data/resolve_sparsedrive/det_anchors.npy')
    det.instance_bank.default_time_interval=.1;det.instance_bank.max_time_interval=.51
    det.deformable_model.num_cams=4
    det.deformable_model.type='ResolveFeatureAggregation'
    det.deformable_model.kps_generator.type='ResolveBox3DKeyPointsGenerator'
    det.with_instance_id=True
    # RESOLVE golf carts and vans have meaningful orientation/velocity. Do not
    # inherit nuScenes traffic-cone/static-object regression exceptions.
    det.sampler.cls_wise_reg_weights={}
    det.loss_reg.cls_allow_reverse=[]
    motion=model.head.motion_plan_head
    motion.type='ResolveMotionHead';motion.fut_ts=10;motion.fut_mode=6
    motion.motion_anchor=str(ROOT/'data/resolve_sparsedrive/motion_anchors.npy')
    motion.operation_order=['temp_gnn','gnn','norm','ffn','norm']*3+['refine']
    motion.refine_layer.fut_ts=10;motion.refine_layer.fut_mode=6
    for key in ['plan_anchor','instance_queue','planning_sampler','planning_decoder','plan_loss_cls','plan_loss_reg','plan_loss_status','cross_graph_model','num_map']:
        motion.pop(key,None)
    motion.queue_length=4
    replace_flash(model)
    config=Config(dict(model=model,class_names=base.class_names,input_size=(704,384),batch_size=2,accumulation_steps=4,amp=False,stage1_epochs=20,stage2_epochs=10,learning_rate=1e-4,weight_decay=.001,grad_clip=25.,warmup_steps=500,seed=20261028,camera_order=['axis1','axis2','axis3','axis4'],protocol='resolve_official_curves_paper_nds9_v1',forecast_steps=10,forecast_hz=2,forecast_modes=6,split_file=str(ROOT/'third_party/MC-DeTra/splits/resolve_forecast_split.json')))
    target=ROOT/'configs/resolve/sparsedrive_camera.py';target.parent.mkdir(exist_ok=True)
    config.dump(str(target));print(target)

if __name__=='__main__':main()
