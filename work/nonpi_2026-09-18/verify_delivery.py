from pathlib import Path
import hashlib,json,sys,platform
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
OUT=Path(__file__).resolve().parent
import torch,numpy as np
from torch import nn
from src.v2.model import encoder as E
m=E.build_image_cnn(16)
m.fc=nn.Sequential(nn.AdaptiveAvgPool2d(1),nn.Flatten(),nn.Linear(64,64),nn.ReLU(True),nn.Dropout(.5),nn.Linear(64,16))
h=E.build_head();mac=[0]
def hook(layer,inputs,output):
    if isinstance(layer,nn.Conv2d):mac[0]+=output.numel()*layer.in_channels*layer.kernel_size[0]*layer.kernel_size[1]//layer.groups
    elif isinstance(layer,nn.Linear):mac[0]+=output.numel()*layer.in_features
hooks=[x.register_forward_hook(hook) for x in m.modules() if isinstance(x,(nn.Conv2d,nn.Linear))]
m.eval()
with torch.no_grad():m(torch.zeros(1,1,64,160))
for x in hooks:x.remove()
cost=dict(encoder_params=sum(x.numel() for x in m.parameters()),head_params=sum(x.numel() for x in h.parameters()),encoder_mmac=mac[0]/1e6,head_mmac=E.temporal_head_mmac())
assert cost['encoder_params']==33264 and abs(cost['encoder_mmac']-31.36832)<1e-9
records=json.loads((ROOT/'snapshots/nonpi_2026-09-18/manifest.json').read_text(encoding='utf8'))
changed=[];checked=0
for r in records:
    if 'error' in r:continue
    p=Path(r['path']);checked+=1
    if hashlib.sha256(p.read_bytes()).hexdigest()!=r['sha256']:changed.append(str(p))
assert not changed,changed
scripts={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in OUT.glob('*.py')}
report=dict(original_files_checked=checked,original_files_changed=changed,archive_unreadable_files=[r['path'] for r in records if 'error' in r],
            gap_cost=cost,scripts_sha256=scripts,environment=dict(python=platform.python_version(),numpy=np.__version__,torch=torch.__version__,cuda=torch.version.cuda,gpu=torch.cuda.get_device_name()),
            notes=['Compare completed training in gap_completion_status.json; no partial accuracy promoted.',
                   'PDF review draft clears unassigned publication metadata and labels the first-page footer as Review draft.'])
(OUT/'results/delivery_verification.json').write_text(json.dumps(report,indent=2),encoding='utf8')
print(json.dumps({k:v for k,v in report.items() if k!='scripts_sha256'},indent=2))
