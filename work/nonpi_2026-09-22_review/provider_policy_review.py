"""Apply the shared online eligibility rule to frozen diagnostic scores."""
from pathlib import Path
import sys,json
import numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from src.v2.deploy import inference_policy as P
HERE=Path(__file__).resolve().parent;BASE=ROOT/'work/nonpi_2026-09-21_maskfix/post_train'
AUDIT=ROOT/'work/nonpi_2026-09-21/submission_audit'
idx=np.load(ROOT/'data/processed/v2/index.npz');out={}
for group,folder in [('stratified_sample',AUDIT/'provider_sample'),('diagnostic_cases',AUDIT)]:
    rr=json.loads((BASE/f'provider_{group}.json').read_text())['runs'];byvariant={}
    for variant in ['vpres','image_head','vdrop','gap']:
        seeds=[]
        for s in range(3):
            rows=[r for r in rr if r['variant']==variant and r['seed']==s];pos=srcfn=fn=fp=eligible=total=0
            for r in rows:
                d=np.load(folder/f'landmark_U{r["subject"]}_features.npz')
                assert np.array_equal(d['event_ids'],r['event_ids'])
                ok=P.eligible_mask(d['mask']);y=np.array(r['labels']);thr=r['threshold_logit']
                pred=(np.array(r['fresh_logits'])>=thr)&ok
                sourceok=P.eligible_mask(idx['e_rows'][r['event_ids']]>=0)
                srcpred=(np.array(r['original_logits'])>=thr)&sourceok
                pos+=int(y.sum());srcfn+=int(np.sum(~srcpred&(y==1)))
                fn+=int(np.sum(~pred&(y==1)));fp+=int(np.sum(pred&(y==0)))
                eligible+=int(ok.sum());total+=len(y)
            seeds.append(dict(seed=s,n=total,eligible=eligible,source_recall=1-srcfn/pos,fresh_recall=1-fn/pos,fresh_fp=fp))
        byvariant[variant]=dict(seeds=seeds,mean_source_recall=float(np.mean([r['source_recall'] for r in seeds])),mean_fresh_recall=float(np.mean([r['fresh_recall'] for r in seeds])))
    out[group]=byvariant
out['policy']=P.metadata();out['limitations']=['Frozen crops precede the explicit float32 coordinate fix; this replays eligibility, not a new raw-video extraction.','Abstained positive events count as misses. Fixed small samples, descriptive only.']
(HERE/'provider_policy_review.json').write_text(json.dumps(out,indent=2),encoding='utf8')
print(json.dumps(out['stratified_sample'],indent=2))
