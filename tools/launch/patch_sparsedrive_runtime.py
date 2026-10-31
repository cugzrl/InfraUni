"""Small, idempotent import patches; original behavior remains the default."""
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
r=ROOT/'third_party/SparseDrive'
for relative in ['projects/mmdet3d_plugin/__init__.py','projects/mmdet3d_plugin/datasets/__init__.py']:
    p=r/relative;t=p.read_text()
    if 'INFRAUNI_SPARSEDRIVE_MINIMAL' not in t:
        minimal='from .models import *' if '/datasets/' not in relative else 'from . import utils'
        p.write_text("import os\nif os.environ.get('INFRAUNI_SPARSEDRIVE_MINIMAL')=='1':\n    "+minimal+"\nelse:\n"+''.join('    '+line+'\n' for line in t.splitlines()))
p=r/'projects/mmdet3d_plugin/models/attention.py';t=p.read_text()
if 'flash_attn_unpadded_kvpacked_func = None' not in t:
    a=t.index('try:\n    from flash_attn');b=t.index('\n\ndef _in_projection_packed',a)
    t=t[:a]+'''try:
    from flash_attn.flash_attn_interface import flash_attn_varlen_kvpacked_func as flash_attn_unpadded_kvpacked_func
    from flash_attn.bert_padding import unpad_input, pad_input, index_first_axis
except ImportError:
    flash_attn_unpadded_kvpacked_func = None
    unpad_input = pad_input = index_first_axis = None
'''+t[b:]
    p.write_text(t)
print('NATIVE_IMPORT_PATCHES_DONE')

p=r/'projects/mmdet3d_plugin/models/blocks.py';t=p.read_text()
if 'InfraUni explicit missing-view mask' not in t:
    needle='        if self.training and self.attn_drop > 0:'
    replacement="""        # InfraUni explicit missing-view mask: all valid views preserve the
        # original attention; missing camera slots never contribute features.
        if metas is not None and 'camera_valid' in metas:
            valid=metas['camera_valid'][:, None, :, None, None, None].to(weights)
            weights=weights*valid
            weights=weights/weights.sum(dim=(2,3,4),keepdim=True).clamp_min(1e-8)
"""+needle
    assert needle in t;t=t.replace(needle,replacement);p.write_text(t)
