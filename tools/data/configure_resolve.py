import yaml,json
from pathlib import Path
root=Path(__file__).resolve().parents[2]
vendor=root/'third_party/RESOLVE'
model=yaml.safe_load((vendor/'tools/cfgs/sunlakes_models/transfusion.yaml').read_text())
data=yaml.safe_load((vendor/'tools/cfgs/dataset_configs/sunlakes_dataset.yaml').read_text())
data.update(model['DATA_CONFIG']);data.pop('_BASE_CONFIG_',None)
data.update(DATASET='ResolveSequenceDataset',DATA_PATH=str(root/'data/resolve_protocol'),VERSION='',FILTER_MIN_POINTS_IN_GT=False,INFO_PATH={'train':['train.pkl'],'test':['val.pkl']})
data.update(POINT_CLOUD_RANGE=[-120,-96,-10,72,96,2], DATA_PATH=str(root/'data/resolve_multimodal'), INFO_PATH={'train':['infos/train.pkl'],'test':['infos/val.pkl']}, FILTER_EMPTY_GT=False)

data['DATA_AUGMENTOR']={'DISABLE_AUG_LIST':[],'AUG_CONFIG_LIST':[]}
data['DATA_PROCESSOR'][1]['SHUFFLE_ENABLED']['test']=False
model['DATA_CONFIG']=data
model['MODEL']['BACKBONE_3D']['GRID_SIZE']=[640,640,60]
model['MODEL']['MAP_TO_BEV']['NUM_BEV_FEATURES']=384
model['MODEL']['DENSE_HEAD']['POST_PROCESSING']['POST_CENTER_RANGE']=[-120,-96,-10,72,96,2]
model.pop('HOOK',None)
model['MODEL']['POST_PROCESSING']['EVAL_METRIC']='resolve'
target=root/'configs/resolve/transfusion_sequence.yaml';target.parent.mkdir(parents=True,exist_ok=True)
target.write_text(yaml.safe_dump(model,sort_keys=False))
runtime=vendor/'tools/cfgs/infrauni/transfusion_sequence.yaml';runtime.parent.mkdir(exist_ok=True)
if not runtime.exists():runtime.symlink_to(target)
cfg=root/'third_party/MC-DeTra/configs/resolve_mc_detra.json'
payload=json.loads(cfg.read_text())
payload.update(train_cache=str(root/'data/resolve_multimodal/train'),val_cache=str(root/'data/resolve_multimodal/val'),run_dir=str(root/'experiments/mcdetra/resolve_multimodal_v1'),checkpoint_path=str(root/'experiments/mcdetra/resolve_multimodal_v1/model.pt'))
payload.update(roi=[-120,-96,72,96], z_range=[-10,2],val_frames_per_sequence=0,val_batches=None,save_best_metric='NDS',run_dir=str(root/'experiments/resolve_roi_v2_20261029/mc-detra'),checkpoint_path=str(root/'experiments/resolve_roi_v2_20261029/mc-detra/model.pt'))
cfg.write_text(json.dumps(payload,indent=2)+'\n')
(root/'configs/resolve/mcdetra.json').write_text(json.dumps(payload,indent=2)+'\n')
