"""Run once after the isolated 15-run job exits; never promote partial data."""
from pathlib import Path
import json,sys,hashlib
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
OUT=Path(__file__).resolve().parent
from src.v2.compare_variants import load_variant,align
from src.v2.common import thresholds as TH,stats
import numpy as np
path=OUT/'results/train_gap.json'
if not path.exists():
    (OUT/'results/gap_completion_status.json').write_text(json.dumps({'status':'incomplete_or_failed','message':'Final training JSON absent. Inspect train_gap.log; do not cite partial values.'},indent=2))
    raise SystemExit(1)
g=load_variant(str(path))
if set(g['runs'])!={(f,s) for f in range(5) for s in range(3)}:
    raise SystemExit('Incomplete or duplicated run set')
out={'status':'complete','n_runs':15,'architecture':'Image-CNN-GAP + identical TCN',
     'cost':{'encoder_params':33264,'head_params':4625,'total_params':37889,
             'encoder_mmac':31.368320,'head_mmac':.045888,'total_mmac':31.414208},
     'comparisons':{},'mean_AP':float(np.mean([r['pr_auc_json'] for r in g['runs'].values()])),
     'sd_AP':float(np.std([r['pr_auc_json'] for r in g['runs'].values()],ddof=1)),
     'training_minutes':g['meta']['minutes'],
     'claim_limit':'GAP variant trained in a new environment; NumPy differs from original. No Pi timing is measured. Raw paired source comparisons only; all 15 runs retained.'}
for name,stem in [('ours','train_encoder_final'),('image_head','train_image_cnn_head_final')]:
    a=load_variant(str(ROOT/f'results/v2/{stem}.json'))
    keys,sa,sb,y,sub=align(a,g)
    ci=stats.subject_bootstrap(lambda ix:TH.average_precision(sa[ix],y[ix])-TH.average_precision(sb[ix],y[ix]),sub,2000,seed=0)
    out['comparisons'][f'{name}_minus_gap']=ci
out['all_runs_early_stopped']=all(r['converged'] for r in g['meta']['runs'])
if not out['all_runs_early_stopped']:
    out['warning']='At least one fit reached its epoch budget; inspect convergence before final claims.'
(OUT/'results/gap_completion_status.json').write_text(json.dumps(out,indent=2))
print(json.dumps(out,indent=2))
