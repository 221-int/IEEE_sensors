"""CPU smoke/regression checks using explicitly labelled legacy fixtures only."""
from pathlib import Path
import sys,json
import numpy as np,torch
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
import post_train as P
from src.v2.train_encoder import Bundle
torch.set_num_threads(2)
original_read=P.read
def fixture_read(path):
    if 'records' in path.parts:
        variant=path.parent.name
        if variant in ['vpres','image_head']:
            stem='train_encoder_final' if variant=='vpres' else 'train_image_cnn_head_final'
            data=json.loads((ROOT/f'results/v2/{stem}.json').read_text(encoding='utf8'))
            run=next(r for r in data['runs'] if r['fold']==0 and r['seed']==0)
            return {'models':str(ROOT/f'models/v2/{stem}/fold0_seed0'),'run':run}
        return original_read(ROOT/f'work/nonpi_2026-09-21/records/{variant}/fold0_seed0.json')
    return original_read(path)
P.read=fixture_read
b=Bundle(str(ROOT/'data/processed/v2'));ix=np.load(ROOT/'data/processed/v2/index.npz')
ids=np.flatnonzero(ix['e_valid']&(ix['e_fold']==0))[:2];rows=ix['e_rows'][ids]
crops=np.asarray(b.frames[np.maximum(rows,0)]).copy();mask=(rows>=0).astype(np.float32);mask[:,8]=0
altered=crops.copy();altered[:,8]=255
checks=[]
for v in ['vpres','image_head','vdrop','gap']:
    front,head,thr=P.model(v,0,0,'cpu')
    a=P.infer(front,head,crops,mask,'cpu');z=P.infer(front,head,altered,mask,'cpu')
    assert a.shape==(2,) and np.isfinite(a).all() and np.array_equal(a,z)
    checks.append({'variant':v,'finite':True,'masked_pixels_invariant':True})
(P.HERE/'post_train_smoke.json').write_text(json.dumps({'fixtures':'legacy models; not corrected experiment results','checks':checks},indent=2))
print(json.dumps(checks))
