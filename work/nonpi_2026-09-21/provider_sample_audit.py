"""Freeze an error-independent source sample, then audit landmark transfer on CPU."""
from pathlib import Path
import sys,json,csv,hashlib,time,os,argparse
import numpy as np
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1];sys.path.insert(0,str(HERE));sys.path.insert(0,str(ROOT))
import landmark_case_audit as A
from src.v2.common import thresholds as TH
OUT=HERE/'submission_audit/provider_sample';OUT.mkdir(exist_ok=True);A.OUT=OUT

def prepare():
    file=OUT/'selection.json'
    if file.exists():return json.loads(file.read_text())
    x=np.load(ROOT/'data/processed/v2/index.npz');rng=np.random.default_rng(20260921)
    with (ROOT/'data/raw/mEBAL2/glasses_labels_58.csv').open(encoding='utf8') as f:
        glasses={int(r['user']):int(r['glasses']) for r in csv.DictReader(f)}
    groups={};subjects=np.unique(x['e_subject'][x['e_valid'].astype(bool)])
    for u in subjects:
        batch=int(x['e_batch2020'][np.flatnonzero(x['e_subject']==u)[0]])
        groups.setdefault(f'batch2020={batch},glasses={glasses[int(u)]}',[]).append(int(u))
    selected=[]
    for group,candidates in sorted(groups.items()):
        for u in sorted(rng.choice(candidates,min(2,len(candidates)),replace=False).tolist()):
            events=[]
            for label in [0,1]:
                pool=np.flatnonzero(x['e_valid']&(x['e_subject']==u)&(x['e_is_blink']==label))
                events.extend(rng.choice(pool,min(10,len(pool)),replace=False).tolist())
            selected.append(dict(subject=u,stratum=group,event_ids=sorted(events)))
    plan=dict(created=time.strftime('%Y-%m-%dT%H:%M:%S'),seed=20260921,selection='Two people per collection-batch x glasses stratum; up to ten events per class per person. Predictions do not enter selection.',
              scope='Post-hoc source-domain deployment diagnostic, not external validation or a new architecture-selection set.',
              primary='Report detection coverage, source-provider versus fresh-provider AP on common eligible events, and fixed-source-threshold recall with abstentions counted as misses.',
              inference='Frozen subject-held-out fold and all three seeds; CPU for both reference and new crops; no fitting.',
              segment_policy='60 native-frame warmup before each non-overlapping selected segment; not uninterrupted tracking.',
              no_tuning=True,groups=groups,selected=selected)
    file.write_text(json.dumps(plan,indent=2),encoding='utf8');return plan

def summarize(plan):
    records=[]
    for sample in plan['selected']:
        u=sample['subject'];r=json.loads((OUT/f'landmark_U{u}.json').read_text());y=np.array(r['labels']);eligible=np.array(r['eligible'])
        for p in r['predictions']:
            old=np.array(p['reference_cpu_logits']);new=np.array(p['logits']);thr=p['threshold']
            both=y[eligible];has_two=len(np.unique(both))==2
            oldpred=old>=thr;newpred=(new>=thr)&eligible
            records.append(dict(subject=u,stratum=sample['stratum'],variant=p['variant'],seed=p['seed'],n=len(y),eligible=int(eligible.sum()),
                source_ap_common=TH.average_precision(old[eligible],both) if has_two else None,
                fresh_ap_common=TH.average_precision(new[eligible],both) if has_two else None,
                delta_ap_common=TH.average_precision(new[eligible],both)-TH.average_precision(old[eligible],both) if has_two else None,
                source_tp=int(np.sum(oldpred&(y==1))),source_fp=int(np.sum(oldpred&(y==0))),
                fresh_tp=int(np.sum(newpred&(y==1))),fresh_fp=int(np.sum(newpred&(y==0))),n_positive=int(y.sum()),
                prediction_changes=int(np.sum(oldpred!=newpred)),cpu_reference_max_error=p['cpu_reference_max_error'],
                cpu_reference_prediction_flips=p['cpu_reference_prediction_flips']))
    means={}
    for variant in ['ours','image_head','ear_head']:
        rr=[r for r in records if r['variant']==variant]
        means[variant]=dict(subject_seed_mean_delta_ap=float(np.mean([r['delta_ap_common'] for r in rr if r['delta_ap_common'] is not None])),
            source_recall=sum(r['source_tp'] for r in rr)/sum(r['n_positive'] for r in rr),
            fresh_recall_with_abstention_misses=sum(r['fresh_tp'] for r in rr)/sum(r['n_positive'] for r in rr),
            source_fp_per_run_mean=float(np.mean([r['source_fp'] for r in rr])),fresh_fp_per_run_mean=float(np.mean([r['fresh_fp'] for r in rr])))
    report=dict(subjects=len(plan['selected']),events=sum(len(s['event_ids']) for s in plan['selected']),runs=records,means=means,
        limitations=['Small stratified sample with equal stratum allocation and class-balanced subsampling; no population-wide or external-generalization claim.',
                     'Per-subject AP on up to 20 events is coarse. Three seed predictions are correlated, not three independent people.',
                     'AP conditions on common eligible events; recall also reports all sampled positive events, counting abstentions as misses.',
                     'No Pi time/power or uninterrupted continuous event detection is evaluated.'])
    (OUT/'summary.json').write_text(json.dumps(report,indent=2),encoding='utf8')
    return {k:v for k,v in report.items() if k!='runs'}

def main():
    ap=argparse.ArgumentParser();ap.add_argument('command',choices=['prepare','run']);args=ap.parse_args();plan=prepare()
    if args.command=='prepare':print(json.dumps(plan,indent=2));return
    try:
        for sample in plan['selected']:
            u=sample['subject']
            A.dump('status.json',dict(state='running',subject=u,pid=os.getpid(),updated=time.strftime('%Y-%m-%dT%H:%M:%S')))
            if not (OUT/f'landmark_U{u}.json').exists():
                A.run_user(u,sample['event_ids'],'Error-independent stratified sample; descriptive source-domain comparison only.')
        report=summarize(plan);A.dump('status.json',dict(state='complete',updated=time.strftime('%Y-%m-%dT%H:%M:%S'),summary='summary.json'));print(json.dumps(report,indent=2))
    except Exception as exc:A.dump('status.json',dict(state='failed',error=repr(exc)));raise

if __name__=='__main__':main()
