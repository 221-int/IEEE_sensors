"""Subject-cluster sensitivity for the AP mean actually printed in the table."""
from pathlib import Path
import sys,json
import numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from src.v2.common import stats,thresholds as TH
HERE=Path(__file__).resolve().parent;BASE=ROOT/'work/nonpi_2026-09-21_maskfix/results'
runs=[];people=set()
for f in range(5):
    for s in range(3):
        a=np.load(BASE/f'train_vpres_scores/fold{f}_seed{s}.npz');b=np.load(BASE/f'train_image_head_scores/fold{f}_seed{s}.npz')
        people.update(a['subject'].tolist())
        runs.append((a['ours'],b['ours'],a['y'],{int(u):np.flatnonzero(a['subject']==u) for u in np.unique(a['subject'])}))
people=np.array(sorted(people))
def metric(indices):
    chosen=people[indices];deltas=[]
    for a,b,y,lookup in runs:
        rows=[lookup[int(u)] for u in chosen if int(u) in lookup]
        if not rows:return float('nan')
        q=np.concatenate(rows);deltas.append(TH.average_precision(a[q],y[q])-TH.average_precision(b[q],y[q]))
    return float(np.mean(deltas))
ci=stats.subject_bootstrap(metric,people,2000,seed=0)
out=dict(contrast='vpres_minus_image_head',estimand='mean of 15 held-out AP differences',ci=ci,
         ni={str(d):stats.non_inferiority(ci['ci_lo'],ci['ci_hi'],d) for d in [.02,.01,.005]},
         limitations='Conditional on the 60 fixed fits; sample people together across seeds. Not uncertainty from repeated retraining.')
(HERE/'run_mean_ci.json').write_text(json.dumps(out,indent=2),encoding='utf8');print(json.dumps(out))
