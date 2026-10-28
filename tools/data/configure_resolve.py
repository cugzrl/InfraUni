import yaml,json
from pathlib import Path
root=Path(__file__).resolve().parents[2]
vendor=root/'third_party/RESOLVE'
model=yaml.safe_load((vendor/'tools/cfgs/sunlakes_models/transfusion.yaml').read_text())
data=yaml.safe_load((vendor/'tools/cfgs/dataset_configs/sunlakes_dataset.yaml').read_text())
data.update(model['DATA_CONFIG']);data.pop('_BASE_CONFIG_',None)
data.update(DATASET='ResolveSequenceDataset',DATA_PATH=str(root/'data/resolve_protocol'),VERSION='',FILTER_MIN_POINTS_IN_GT=False,INFO_PATH={'train':['train.pkl'],'test':['val.pkl']})
data['DATA_AUGMENTOR']={'DISABLE_AUG_LIST':[],'AUG_CONFIG_LIST':[]}
data['DATA_PROCESSOR'][1]['SHUFFLE_ENABLED']['test']=False
model['DATA_CONFIG']=data
model.pop('HOOK',None)
model['MODEL']['POST_PROCESSING']['EVAL_METRIC']='resolve'
target=root/'configs/resolve/transfusion_sequence.yaml';target.parent.mkdir(parents=True,exist_ok=True)
target.write_text(yaml.safe_dump(model,sort_keys=False))
runtime=vendor/'tools/cfgs/infrauni/transfusion_sequence.yaml';runtime.parent.mkdir(exist_ok=True)
if not runtime.exists():runtime.symlink_to(target)
cfg=root/'third_party/MC-DeTra/configs/resolve_mc_detra.json'
payload=json.loads(cfg.read_text())
payload.update(train_cache=str(root/'data/resolve_cache/train'),val_cache=str(root/'data/resolve_cache/val'),run_dir=str(root/'experiments/mcdetra/resolve'),checkpoint_path=str(root/'experiments/mcdetra/resolve/model.pt'))
cfg.write_text(json.dumps(payload,indent=2)+'\n')
(root/'configs/resolve/mcdetra.json').write_text(json.dumps(payload,indent=2)+'\n')
