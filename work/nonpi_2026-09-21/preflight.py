from pathlib import Path
import sys,json,hashlib
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
import numpy as np
from src.v2.train_encoder import Bundle
from src.v2.common import splits
from src.v2.model import encoder as E
b=Bundle(str(ROOT/'data/processed/v2'));checks=[]
for fold in range(5):
    m=splits.subject_masks(b.subject,splits.load_folds(),splits.fold_rotation(fold))
    te=np.flatnonzero(m['test'])
    d=np.load(ROOT/f'results/v2/train_encoder_final_scores/fold{fold}_seed0.npz')
    assert np.array_equal(b.y[te],d['y']) and np.array_equal(b.subject[te],d['subject'])
    checks.append(dict(fold=fold,n_test=len(te)))
old=json.loads((ROOT/'snapshots/nonpi_2026-09-18/manifest.json').read_text(encoding='utf8'))
changed=[]
for item in old:
    p=Path(item['path'])
    if 'error' not in item and p.suffix=='.py' and 'v2' in p.parts and 'src' in p.parts:
        if hashlib.sha256(p.read_bytes()).hexdigest()!=item['sha256']:changed.append(str(p))
assert not changed,changed
cost=dict(encoder_params=sum(p.numel() for p in E.build('vdrop').parameters()),head_params=sum(p.numel() for p in E.build_head().parameters()),
          total_mmac=E.analyse('vdrop')['total_mmac']+E.temporal_head_mmac())
out=dict(fold_alignment=checks,source_changed_since_sept18=changed,vdrop_cost=cost)
(Path(__file__).resolve().parent/'preflight.json').write_text(json.dumps(out,indent=2))
print(json.dumps(out,indent=2))
