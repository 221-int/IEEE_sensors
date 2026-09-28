"""Separate CPU numerical differences from landmark-provider interventions."""
from pathlib import Path
import sys,json
import numpy as np,torch
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from src.v2.model import encoder as E
from src.v2.dataset import crop as C
from src.v2.common import splits
OUT=Path(__file__).resolve().parent/'submission_audit'
torch.set_num_threads(2)
idx=np.load(ROOT/'data/processed/v2/index.npz');frames=np.load(ROOT/'data/processed/v2/frames_m22.npy',mmap_mode='r')
events=json.loads((OUT/'failure_events.json').read_text());events={e['event_index']:e for e in events}
sel=json.loads((OUT/'review_selection.json').read_text());reports=[]
for user,ids in sel.items():
    fold=splits.load_folds()[int(user)];rows=idx['e_rows'][ids];mask=(rows>=0).astype(np.float32)
    n,t=rows.shape;flat=np.maximum(rows,0).ravel()
    for variant,stem,key in [('ours','train_encoder_final','ours'),('image_head','train_image_cnn_head_final','ours'),('ear_head','train_encoder_final','ear_head')]:
        results=json.loads((ROOT/f'results/v2/{stem}.json').read_text(encoding='utf8'))
        for run in results['runs']:
            if run['fold']!=fold:continue
            seed=run['seed'];folder=ROOT/f'models/v2/{stem}/fold{fold}_seed{seed}'
            front=E.build() if variant=='ours' else E.build_image_cnn(16) if variant=='image_head' else E.build_ear_frontend();head=E.build_head()
            front.load_state_dict(torch.load(folder/('earhead_front.pt' if variant=='ear_head' else 'encoder.pt'),weights_only=True,map_location='cpu'));front.eval()
            head.load_state_dict(torch.load(folder/('earhead_head.pt' if variant=='ear_head' else 'head.pt'),weights_only=True,map_location='cpu'));head.eval()
            x=idx['f_ear'][flat].copy() if variant=='ear_head' else C.batch_input(np.asarray(frames[flat]))
            if variant=='ear_head':x[mask.ravel()==0]=0
            with torch.no_grad():
                zs=[front(torch.from_numpy(x[j:j+32])) for j in range(0,len(x),32)]
                z=torch.cat(zs).reshape(n,t,-1);logits=head(z,torch.from_numpy(mask)).numpy()
                zz=z.clone();zz[torch.from_numpy(mask)==0]=0
                mask_diff=float(torch.max(torch.abs(head(zz,torch.from_numpy(mask))-torch.from_numpy(logits))))
            ref=np.array([events[e]['seed_logits'][variant][seed] for e in ids]);thr=run[key]['thr']
            reports.append(dict(user=int(user),variant=variant,seed=seed,event_ids=ids,logits=logits.tolist(),
                max_logit_error=float(np.max(np.abs(logits-ref))),prediction_flips=int(np.sum((logits>=thr)!=(ref>=thr))),
                masked_feature_zeroing_logit_diff=mask_diff))
out=dict(scope='Exact selected source cases, frozen checkpoints, CPU replay; diagnostic only',runs=reports,
         max_logit_error=max(r['max_logit_error'] for r in reports),prediction_flips=sum(r['prediction_flips'] for r in reports),
         masked_feature_zeroing_max_diff=max(r['masked_feature_zeroing_logit_diff'] for r in reports))
(OUT/'cpu_reference.json').write_text(json.dumps(out,indent=2),encoding='utf8')
print(json.dumps({k:v for k,v in out.items() if k!='runs'}))
