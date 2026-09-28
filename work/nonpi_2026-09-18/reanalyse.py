"""Read-only inference audit of frozen, paired out-of-fold sidecars."""
from pathlib import Path
import sys,json,hashlib
import numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from src.v2.common import stats,thresholds as TH
from src.v2.compare_variants import load_variant,align
OUT=Path(__file__).resolve().parent/'results';OUT.mkdir(exist_ok=True)
a=load_variant(str(ROOT/'results/v2/train_encoder_final.json'))
b=load_variant(str(ROOT/'results/v2/train_image_cnn_head_final.json'))
keys,sa,sb,y,sub=align(a,b)
def diff(ix):return TH.average_precision(sa[ix],y[ix])-TH.average_precision(sb[ix],y[ix])
boot=stats.subject_bootstrap(diff,sub,2000,seed=0)
persons=[]
for subject in np.unique(sub):
    av=[];bv=[]
    for k in keys:
        ar,br=a['runs'][k],b['runs'][k]; ix=ar['subject']==subject
        if not ix.any():continue
        av.append(TH.average_precision(ar['ours'][ix],ar['y'][ix]))
        bv.append(TH.average_precision(br['ours'][ix],br['y'][ix]))
    persons.append(dict(subject=int(subject),ours=float(np.mean(av)),image_head=float(np.mean(bv)),
                        delta=float(np.mean(av)-np.mean(bv)),n_seed=len(av),
                        n_events=int(np.sum(sub==subject)//3),leave_out_pooled_delta=float(diff(np.flatnonzero(sub!=subject)))))
delta=np.array([s['delta'] for s in persons])
macro=stats.subject_bootstrap(lambda ix:float(delta[ix].mean()),np.arange(len(persons)),2000,seed=0)
run_diff=[a['runs'][k]['pr_auc_json']-b['runs'][k]['pr_auc_json'] for k in keys]
report=dict(primary_pooled_bootstrap=boot,pooled_ours=float(TH.average_precision(sa,y)),
            pooled_image_head=float(TH.average_precision(sb,y)),run_mean_delta=float(np.mean(run_diff)),
            run_delta_sd=float(np.std(run_diff,ddof=1)),subject_macro_seed_averaged_delta=macro,
            n_subjects=len(persons),n_unique_events=len(y)//3,n_pooled_predictions=len(y),
            sensitivity=[dict(delta=d,verdict=stats.non_inferiority(boot['ci_lo'],boot['ci_hi'],d),
                              status='primary' if d==.02 else 'post-hoc') for d in [.02,.01,.005]],
            subjects=persons,limitations=['Conditional on the saved training runs; no retraining uncertainty.',
                 'Seed SD is descriptive and cannot override a subject bootstrap CI excluding zero.',
                 'Subject-macro AP averages within-subject per-seed AP; it is not pooled AP.'])
(OUT/'source_robustness.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf8')
print(json.dumps({k:v for k,v in report.items() if k!='subjects'},indent=2))
