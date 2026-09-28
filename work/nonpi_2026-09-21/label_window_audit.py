"""Post-hoc interval-overlap and exclusion audit; never relabel or drop training data."""
from pathlib import Path
import sys,json
import numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from src.v2.common import thresholds as TH
OUT=Path(__file__).resolve().parent/'submission_audit'
x=np.load(ROOT/'data/processed/v2/index.npz');keep=x['e_valid'].astype(bool)
overlap=np.zeros(len(keep),int)
for u in np.unique(x['e_subject']):
    ix=np.flatnonzero(x['e_subject']==u);a=x['e_start'][ix];b=x['e_end'][ix]
    occupied=np.zeros(int(b.max())+2,bool)
    for e in ix[x['e_is_blink'][ix]==1]:occupied[x['e_start'][e]:x['e_end'][e]+1]=True
    for e in ix[x['e_is_blink'][ix]==0]:overlap[e]=occupied[x['e_start'][e]:x['e_end'][e]+1].sum()
uids=np.flatnonzero(keep);fold=x['e_fold'][keep];y=x['e_is_blink'][keep]
conditions={'all':np.ones(len(uids),bool),'exclude_negative_window_overlap':~((y==0)&(overlap[keep]>0)),
            'complete_19_frames':x['e_n_missing'][keep]==0}
runs=[]
for f in range(5):
    for seed in range(3):
        a=np.load(ROOT/f'results/v2/train_encoder_final_scores/fold{f}_seed{seed}.npz')
        b=np.load(ROOT/f'results/v2/train_image_cnn_head_final_scores/fold{f}_seed{seed}.npz')
        assert np.array_equal(a['y'],y[fold==f]) and np.array_equal(a['y'],b['y'])
        for condition,sel in conditions.items():
            q=sel[fold==f]
            ap=TH.average_precision(a['ours'][q],a['y'][q]);bp=TH.average_precision(b['ours'][q],b['y'][q])
            runs.append(dict(fold=f,seed=seed,condition=condition,n=int(q.sum()),ours_ap=ap,image_ap=bp,delta=ap-bp))
summary={}
for name,mask in conditions.items():
    rows=[r for r in runs if r['condition']==name]
    summary[name]=dict(n_events=int(mask.sum()),n_positive=int(y[mask].sum()),
                       runmean_ours_ap=float(np.mean([r['ours_ap'] for r in rows])),
                       runmean_image_ap=float(np.mean([r['image_ap'] for r in rows])),
                       runmean_delta=float(np.mean([r['delta'] for r in rows])))
reasons={k:int(np.sum(x[k].astype(bool))) for k in ['e_off_seat','e_excluded_user']}
reasons['missing_policy']=int(np.sum(~x['e_valid_missing_only'].astype(bool)))
out=dict(n_before=int(len(keep)),n_retained=int(keep.sum()),n_excluded=int((~keep).sum()),exclusion_counts_nonexclusive=reasons,
 n_negative_windows_with_positive_window_overlap_before=int(np.sum(overlap>0)),
 n_negative_windows_with_positive_window_overlap_retained=int(np.sum((overlap>0)&keep)),
 overlap_frame_histogram={str(int(k)):int(v) for k,v in zip(*np.unique(overlap[(overlap>0)&keep],return_counts=True))},
 conditions=summary,runs=runs,
 limitations=['Positive labels describe whole 19-frame event windows, not per-frame eyelid closure. Window overlap is ambiguity, not proof of mislabeling.',
              'Post-hoc subsets change prevalence and sample composition; these are descriptive sensitivities with no new confidence intervals or model selection.',
              'No labels or primary events were edited. Visual concerns require independent annotation and raw temporal context.'])
(OUT/'label_windows.json').write_text(json.dumps(out,indent=2),encoding='utf8')
print(json.dumps({k:v for k,v in out.items() if k!='runs'},indent=2))
