"""Descriptive post-hoc error decomposition, without modifying decisions."""
import json,numpy as np
from pathlib import Path
from external_eval import intervals,match
p=Path(__file__).resolve().parent/'results'
d=np.load(p/'external_scores.npz')
rs=json.loads((p/'external_pilot.json').read_text())['runs']
gt=[(int(np.flatnonzero(d['label']==i)[0]),int(np.flatnonzero(d['label']==i)[-1])) for i in np.unique(d['label'][d['label']>=0])]
o=[]
for r in rs:
    k=f"{r['variant']}_fold{r['fold']}_seed{r['seed']}"
    positive=np.zeros(len(d['label']),bool)
    positive[d['center']]=(d[k]>=r['threshold'])&d['eligible']
    pr=intervals(positive);pairs=match(pr,gt);used={a for a,b in pairs}
    unmatched=[x for i,x in enumerate(pr) if i not in used]
    duplicate=sum(any(min(b,e)>=max(a,c) for c,e in gt) for a,b in unmatched)
    o.append(dict(variant=r['variant'],fold=r['fold'],seed=r['seed'],unmatched=len(unmatched),
                  unmatched_overlapping_gt=duplicate,unmatched_no_gt_overlap=len(unmatched)-duplicate))
summary={m:{k:float(np.mean([r[k] for r in o if r['variant']==m])) for k in ['unmatched','unmatched_overlapping_gt','unmatched_no_gt_overlap']} for m in ['ours','image_head','ear_head','ear_rule']}
(p/'external_false_event_diagnostic.json').write_text(json.dumps({'posthoc':True,'runs':o,'summary':summary},indent=2))
print(json.dumps(summary,indent=2))
