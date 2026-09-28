"""Re-evaluate already observed external scores under explicitly named policies."""
from pathlib import Path
import sys,json,importlib.util
import numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
HERE=Path(__file__).resolve().parent;OLD=HERE.parent/'nonpi_2026-09-18'
sp=importlib.util.spec_from_file_location('pilot',OLD/'external_eval.py');P=importlib.util.module_from_spec(sp);sp.loader.exec_module(P)

def decisions(valid,policy):
    windows=np.arange(18,len(valid))[:,None]+np.arange(-18,1)
    if policy=='pilot': return valid[windows].sum(1)>=14
    if policy=='benchmark': return valid[18:].copy()
    if policy=='aligned_current_valid': return (valid[windows].sum(1)>=14)&valid[18:]
    raise ValueError(policy)

def main():
    d=np.load(OLD/'results/external_features.npz');z=np.load(OLD/'results/external_scores.npz')
    label,valid=d['label'],d['valid'];n=len(label);end=np.arange(18,n);center=end-9
    windows=end[:,None]+np.arange(-18,1);known=np.all(label[windows]!=-2,1)
    domain=center[known];duration=len(domain)/30/60
    gt=[]
    for bid in np.unique(label[label>=0]):
        ids=np.flatnonzero(label==bid);a,b=int(ids[0]),int(ids[-1])
        if a>=domain.min() and b<=domain.max():gt.append((a,b))
    summary=[]
    for variant,stem,metric in [('ours','train_encoder_final','ours'),('image_head','train_image_cnn_head_final','ours'),('ear_head','train_encoder_final','ear_head')]:
        source=json.loads((ROOT/f'results/v2/{stem}.json').read_text(encoding='utf8'))
        for policy in ['pilot','benchmark','aligned_current_valid']:
            eligible=known&decisions(valid,policy)
            run_reports=[]
            for run in source['runs']:
                key=f'{variant}_fold{run["fold"]}_seed{run["seed"]}'
                pred=np.zeros(n,bool);pred[center]=(z[key]>=run[metric]['thr'])&eligible
                intervals=P.intervals(pred);vals={}
                for iou in [0,.1,.2,.5]:
                    count=len(P.match(intervals,gt,iou));m=P.rate(count,len(intervals)-count,len(gt)-count)
                    m['fp_per_min']=m['fp']/duration;vals[str(iou)]=m
                neg=known&(label[center]==-1)
                run_reports.append(dict(fold=run['fold'],seed=run['seed'],iou=vals,
                                        negative_frame_positive_rate=float(np.mean(pred[center[neg]]))))
            summary.append(dict(variant=variant,policy=policy,decision_coverage=float(eligible.sum()/known.sum()),
              current_invalid_but_pilot_eligible=int(np.sum(known&decisions(valid,'pilot')&~valid[18:])),
              under_14_but_benchmark_decides=int(np.sum(known&decisions(valid,'benchmark')&~decisions(valid,'pilot'))),
              means={str(i):{k:float(np.mean([r['iou'][str(i)][k] for r in run_reports])) for k in ['recall','precision','f1','fp_per_min']} for i in [0,.1,.2,.5]},runs=run_reports))
    synthetic=[]
    for missing in [0,1,5,6,18,19]:
        v=np.ones(19,bool);v[:missing]=False
        synthetic.append(dict(missing=missing,last_valid=bool(v[-1]),decision={p:bool(decisions(v,p)[0]) for p in ['pilot','benchmark','aligned_current_valid']}))
    v=np.ones(19,bool);v[-1]=False
    synthetic.append(dict(missing=1,last_valid=False,decision={p:bool(decisions(v,p)[0]) for p in ['pilot','benchmark','aligned_current_valid']}))
    out=dict(scope='Post-hoc policy sensitivity on already observed EyeBlink8 video 9; no thresholds changed.',
             n_gt=len(gt),duration_minutes=duration,summary=summary,synthetic_decision_cases=synthetic,
             limitations=['Benchmark policy reproduces eligibility only; benchmark runner itself does not produce event counts or apply validation thresholds.',
                          'Suppressed decisions are represented as no positive; this can fragment event intervals. App holding/state-machine policies require separate evaluation.',
                          'No change to existing inference code, pilot results, manuscript or Pi measurements.'])
    path=HERE/'submission_audit/deployment_policy.json';path.write_text(json.dumps(out,indent=2),encoding='utf8')
    print(json.dumps({k:v for k,v in out.items() if k!='summary'},indent=2))
    for row in summary:print(row['variant'],row['policy'],row['means'])

if __name__=='__main__':main()
