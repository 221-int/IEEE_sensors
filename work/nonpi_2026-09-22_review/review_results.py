"""Read-only audit of the corrected fits; writes only this review directory."""
from pathlib import Path
import sys, json, hashlib, datetime
import numpy as np
ROOT=Path(__file__).resolve().parents[2]; sys.path.insert(0,str(ROOT))
HERE=Path(__file__).resolve().parent
NEW=ROOT/'work/nonpi_2026-09-21_maskfix'
sys.path.insert(0,str(NEW))
import experiment_queue as Q
from src.v2.common import thresholds as TH, stats

def read(p): return json.loads(p.read_text(encoding='utf8'))
def dump(name,x): (HERE/name).write_text(json.dumps(x,indent=2,ensure_ascii=False,allow_nan=False),encoding='utf8')
def mean(x): return float(np.mean(x))

def main():
    ix=np.load(ROOT/'data/processed/v2/index.npz'); valid=np.flatnonzero(ix['e_valid'])
    fullsubjects=ix['e_subject'][valid]; fullfolds=ix['e_fold'][valid]
    models={}; audit=[]; source={}
    for v in Q.VARIANTS:
        rr=Q.records(v); assert len(rr)==15
        scores=[]; ys=[]; subs=[]; folds=[]; runmetrics=[]
        for p in rr:
            r=p['run']; d=np.load(p['scores']); f,s=r['fold'],r['seed']
            ids=np.flatnonzero(ix['e_valid']&(ix['e_fold']==f))
            assert np.array_equal(d['y'],ix['e_is_blink'][ids])
            assert np.array_equal(d['subject'],ix['e_subject'][ids])
            for key in ['ours','ear_head','ear_rule']:
                pred=d[key]>=r[key]['thr']; y=d['y']
                for k,z in [('tp',pred&(y==1)),('fp',pred&(y==0)),('fn',~pred&(y==1)),('tn',~pred&(y==0))]:
                    assert int(z.sum())==r[key][k],(v,f,s,key,k)
            log=Path(p['origin']).parent/'train.log'
            text=log.read_text(encoding='utf8',errors='replace')
            assert 'Traceback (most recent call last)' not in text
            runmetrics.append(dict(fold=f,seed=s,ap=r['ours']['pr_auc'],f1=r['ours']['f1'],best_epoch=r['best_epoch'],
                stopped_at=r['stopped_at'],early_stopped=r['converged'],ear_early_stopped=r['converged_earhead'],
                validation_optimism_ap=r['val_optimism_ap'],threshold_logit=r['ours']['thr']))
            scores.append(d['ours']); ys.append(d['y']); subs.append(d['subject']); folds.append(np.full(len(d['y']),f))
        models[v]=(np.concatenate(scores),np.concatenate(ys),np.concatenate(subs),np.concatenate(folds))
        oldfile={'vpres':'train_encoder_final','image_head':'train_image_cnn_head_final'}.get(v)
        historical=None
        if oldfile:
            old=read(ROOT/f'results/v2/{oldfile}.json')
            historical=dict(old_mean=mean([r['ours']['pr_auc'] for r in old['runs']]),
                corrected_mean=mean([r['ap'] for r in runmetrics]),
                caution='Input correction and environment changed together; not a controlled estimate of bug effect.')
        source[v]=dict(runs=runmetrics,mean_ap=mean([r['ap'] for r in runmetrics]),
            fold_means={str(f):mean([r['ap'] for r in runmetrics if r['fold']==f]) for f in range(5)},
            historical=historical)
        audit.append(dict(variant=v,verified_fits=len(rr),labels_and_subjects_matched=True,hashes_verified=True,
                          confusion_counts_verified=True,cnn_budget_exhausted=sum(not r['early_stopped'] for r in runmetrics),
                          ear_budget_exhausted=sum(not r['ear_early_stopped'] for r in runmetrics)))
        print('verified',v,flush=True)
    comparison=read(NEW/'results/comparison.json'); contrasts={}
    # Cluster reweight people within their original folds. AP pooling over seeds is
    # reserved for the historical pooled estimator; the run-mean uses 15 APs.
    for a,b in [('vpres','image_head'),('vpres','gap'),('vdrop','vpres')]:
        sa,y,u,f=models[a]; sb,yb,ub,fb=models[b]
        assert np.array_equal(y,yb) and np.array_equal(u,ub) and np.array_equal(f,fb)
        pooled=TH.average_precision(sa,y)-TH.average_precision(sb,y)
        saved=comparison['comparisons'][a+'_minus_'+b]
        assert abs(pooled-saved['pooled']['point'])<1e-10
        contrasts[a+'_minus_'+b]=dict(run_mean=source[a]['mean_ap']-source[b]['mean_ap'],
            pooled=saved['pooled'],subject_macro=saved['subject_macro'],
            margins={est:{str(d):stats.non_inferiority(saved[est]['ci_lo'],saved[est]['ci_hi'],d)
                          for d in [.02,.01,.005]} for est in ['pooled','subject_macro']})
    subjects=read(NEW/'post_train/subjects.json')
    person={v:dict(worst=sorted([r for r in subjects if r['variant']==v],key=lambda r:r['ap_mean'])[:5],
                  U1=next(r for r in subjects if r['variant']==v and r['subject']==1),
                  U54=next(r for r in subjects if r['variant']==v and r['subject']==54)) for v in Q.VARIANTS}
    provider={}; reference_checks=[]
    for label in ['stratified_sample','diagnostic_cases']:
        data=read(NEW/f'post_train/provider_{label}.json')['runs']; report={}
        for v in Q.VARIANTS:
            rows=[r for r in data if r['variant']==v]; seeds=[]
            for s in range(3):
                rr=[r for r in rows if r['seed']==s]; positive=sum(sum(r['labels']) for r in rr)
                n=sum(len(r['labels']) for r in rr); eligible=sum(sum(r['eligible']) for r in rr)
                seeds.append(dict(seed=s,n=n,eligible=eligible,positives=positive,
                    source_recall=1-sum(r['source_fn'] for r in rr)/positive,
                    fresh_recall=1-sum(r['fresh_fn_including_abstention'] for r in rr)/positive,
                    source_fp=sum(r['source_fp'] for r in rr),fresh_fp=sum(r['fresh_fp'] for r in rr)))
            for r in rows:
                f=int(ix['e_fold'][r['event_ids'][0]]); ids=np.flatnonzero(ix['e_valid']&(ix['e_fold']==f))
                pos=np.searchsorted(ids,r['event_ids']); assert np.array_equal(ids[pos],r['event_ids'])
                stored=np.load(NEW/f'results/train_{v}_scores/fold{f}_seed{r["seed"]}.npz')['ours'][pos]
                a=np.array(r['original_logits']); thr=r['threshold_logit']
                reference_checks.append(dict(sample=label,variant=v,subject=r['subject'],seed=r['seed'],
                    max_logit_error=float(np.max(np.abs(a-stored))),prediction_flips=int(np.sum((a>=thr)!=(stored>=thr)))))
            report[v]=dict(seeds=seeds,mean_source_recall=mean([r['source_recall'] for r in seeds]),
                mean_fresh_recall=mean([r['fresh_recall'] for r in seeds]),
                subject_seed_common_ap_delta=mean([r['ap_fresh_common']-r['ap_source_common'] for r in rows if r['ap_fresh_common'] is not None]))
        provider[label]=report
    external=read(NEW/'post_train/external_video9.json'); ext={}
    for v in Q.VARIANTS:
        rr=[r for r in external['runs'] if r['variant']==v and r['policy']=='min14_current_valid']; assert len(rr)==15
        ext[v]={iou:{k:mean([r['iou'][iou][k] for r in rr]) for k in ['f1','recall','precision','fp_per_min']} for iou in ['0','0.2','0.5']}
        ext[v]['negative_frame_positive_rate']=mean([r['negative_frame_positive_rate'] for r in rr])
    result=dict(created=datetime.datetime.now().astimezone().isoformat(),integrity=audit,source=source,
        contrasts=contrasts,subjects=person,providers=provider,reference_checks=reference_checks,external=ext,
        external_scope=dict(n_gt=external['n_gt'],minutes=external['minutes'],independent_videos=1),
        caveats=['CI conditions on fixed trained models; 15 fits are not 15 independent participants.',
                 'Pooled score AP, average per-run AP, and equal-person AP are different estimands.',
                 'NI margin 0.02 retained; 0.01/0.005 are sensitivity analyses.',
                 'External single video and error-selected diagnostic cases are exploratory.',
                 'Early stopping is an operational check, not a proof of optimization convergence.'])
    dump('result_review.json',result)
    print(json.dumps(dict(integrity=audit,subjects={v:{u:person[v][u]['ap_mean'] for u in ['U1','U54']} for v in Q.VARIANTS},providers=provider['stratified_sample'],external=ext),indent=2))

if __name__=='__main__': main()
