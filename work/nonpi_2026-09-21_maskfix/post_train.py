"""After the corrected queue, refresh diagnostics from preserved input features.

No new threshold fitting, architecture selection, manuscript edits or notifications.
"""
from pathlib import Path
import sys,json,time,os,argparse,importlib.util
import numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
HERE=Path(__file__).resolve().parent;OUT=HERE/'post_train';OUT.mkdir(exist_ok=True)
OLD=ROOT/'work/nonpi_2026-09-18';AUDIT=ROOT/'work/nonpi_2026-09-21/submission_audit'
from src.v2.common import thresholds as TH

def dump(name,obj):
    p=OUT/name;t=p.with_suffix('.tmp');t.write_text(json.dumps(obj,indent=2,allow_nan=False),encoding='utf8');t.replace(p)
def state(value,**kwargs):dump('status.json',dict(state=value,updated=time.strftime('%Y-%m-%dT%H:%M:%S'),pid=os.getpid(),**kwargs))
def read(p):return json.loads(p.read_text(encoding='utf8'))

def model(variant,fold,seed,device):
    import torch
    from torch import nn
    from src.v2.model import encoder as E
    record=read(HERE/f'records/{variant}/fold{fold}_seed{seed}.json');folder=Path(record['models'])
    if variant in ['vpres','vdrop']:front=E.build(variant)
    else:
        front=E.build_image_cnn(16)
        if variant=='gap':front.fc=nn.Sequential(nn.AdaptiveAvgPool2d(1),nn.Flatten(),nn.Linear(64,64),nn.ReLU(True),nn.Dropout(.5),nn.Linear(64,16))
    head=E.build_head()
    front.load_state_dict(torch.load(folder/'encoder.pt',map_location='cpu',weights_only=True))
    head.load_state_dict(torch.load(folder/'head.pt',map_location='cpu',weights_only=True))
    return front.to(device).eval(),head.to(device).eval(),record['run']['ours']['thr']

def infer(front,head,crops,mask,device):
    import torch
    from src.v2.dataset import crop as C
    n,t=mask.shape;flat=crops.reshape(n*t,64,160);x=C.batch_input(flat);x[mask.ravel()==0]=0
    with torch.no_grad():
        zs=[front(torch.from_numpy(x[j:j+128]).to(device)).cpu() for j in range(0,len(x),128)]
        z=torch.cat(zs).reshape(n,t,16)
        return head(z.to(device),torch.from_numpy(mask).to(device)).cpu().numpy()

def source_subjects():
    from src.v2.common import splits
    x=np.load(ROOT/'data/processed/v2/index.npz');ids=np.flatnonzero(x['e_valid']);subjects=x['e_subject'][ids];fold=x['e_fold'][ids];labels=x['e_is_blink'][ids]
    output=[]
    for v in ['vpres','image_head','vdrop','gap']:
        data=read(HERE/f'results/train_{v}.json');scores=np.zeros((3,len(ids)));pred=np.zeros((3,len(ids)),bool)
        for r in data['runs']:
            f,s=r['fold'],r['seed'];sel=fold==f
            z=np.load(HERE/f'results/train_{v}_scores/fold{f}_seed{s}.npz')
            assert np.array_equal(z['y'],labels[sel]) and np.array_equal(z['subject'],subjects[sel])
            scores[s,sel]=z['ours'];pred[s,sel]=z['ours']>=r['ours']['thr']
        for u in np.unique(subjects):
            q=subjects==u;y=labels[q]
            output.append(dict(variant=v,subject=int(u),n=int(q.sum()),
                               ap_mean=float(np.mean([TH.average_precision(scores[s,q],y) for s in range(3)])),
                               fn_seed=[int(np.sum(~pred[s,q]&(y==1))) for s in range(3)],
                               fp_seed=[int(np.sum(pred[s,q]&(y==0))) for s in range(3)]))
    dump('subjects.json',output)

def providers(device):
    from src.v2.dataset import crop as C
    from src.v2.common import splits
    ix=np.load(ROOT/'data/processed/v2/index.npz');frames=np.load(ROOT/'data/processed/v2/frames_m22.npy',mmap_mode='r');assign=splits.load_folds()
    for label,base in [('diagnostic_cases',AUDIT),('stratified_sample',AUDIT/'provider_sample')]:
        reports=[]
        for file in sorted(base.glob('landmark_U*_features.npz')):
            d=np.load(file);ids=d['event_ids'].astype(int);u=int(ix['e_subject'][ids[0]]);fold=assign[u]
            y=ix['e_is_blink'][ids];freshmask=d['mask'].astype(np.float32);eligible=freshmask.sum(1)>=14
            rows=ix['e_rows'][ids];oldmask=(rows>=0).astype(np.float32);oldcrops=np.asarray(frames[np.maximum(rows,0)])
            for v in ['vpres','image_head','vdrop','gap']:
                for s in range(3):
                    front,head,thr=model(v,fold,s,device)
                    a=infer(front,head,oldcrops,oldmask,device);b=infer(front,head,d['crops'],freshmask,device)
                    yy=y[eligible];twoclass=len(np.unique(yy))==2
                    oldpred=a>=thr;newpred=(b>=thr)&eligible
                    reports.append(dict(subject=u,variant=v,seed=s,event_ids=ids.tolist(),threshold_logit=thr,
                        original_logits=a.tolist(),fresh_logits=b.tolist(),eligible=eligible.tolist(),labels=y.astype(int).tolist(),
                        ap_source_common=TH.average_precision(a[eligible],yy) if twoclass else None,
                        ap_fresh_common=TH.average_precision(b[eligible],yy) if twoclass else None,
                        source_fn=int(np.sum(~oldpred&(y==1))),fresh_fn_including_abstention=int(np.sum(~newpred&(y==1))),
                        source_fp=int(np.sum(oldpred&(y==0))),fresh_fp=int(np.sum(newpred&(y==0)))))
                    del front,head
        dump(f'provider_{label}.json',dict(scope=label,corrected_models=True,runs=reports,
             limits=['Source domain, frozen selected sample; not independent external validation.',
                     'Diagnostic cases selected on old errors; stratified sample selected without predictions.',
                     'Warm-started segments, not uninterrupted streaming. No Pi measurements.']))

def pilot(device):
    import torch
    from src.v2.dataset import crop as C
    sp=importlib.util.spec_from_file_location('legacy_pilot_helpers',OLD/'external_eval.py');P=importlib.util.module_from_spec(sp);sp.loader.exec_module(P)
    d=np.load(OLD/'results/external_features.npz');crops=d['crops'];valid=d['valid'];labels=d['label'];n=len(valid)
    end=np.arange(18,n);center=end-9;windows=end[:,None]+np.arange(-18,1);mask=valid[windows].astype(np.float32)
    known=np.all(labels[windows]!=-2,1);eligible=known&(mask.sum(1)>=14)
    domain=center[known];minutes=len(domain)/30/60;gt=[]
    for bid in np.unique(labels[labels>=0]):
        q=np.flatnonzero(labels==bid);a,b=int(q[0]),int(q[-1])
        if a>=domain.min() and b<=domain.max():gt.append((a,b))
    scores={};results=[]
    x=C.batch_input(crops);x[~valid]=0
    for v in ['vpres','image_head','vdrop','gap']:
        for f in range(5):
            for seed in range(3):
                front,head,thr=model(v,f,seed,device)
                with torch.no_grad():
                    zz=np.concatenate([front(torch.from_numpy(x[j:j+128]).to(device)).cpu().numpy() for j in range(0,n,128)])
                    zz[~valid]=0
                    ss=np.concatenate([head(torch.from_numpy(zz[windows[j:j+256]]).to(device),torch.from_numpy(mask[j:j+256]).to(device)).cpu().numpy() for j in range(0,len(windows),256)])
                key=f'{v}_fold{f}_seed{seed}';scores[key]=ss
                for policy,ok in [('pilot_min14',eligible),('min14_current_valid',eligible&valid[end])]:
                    pred=np.zeros(n,bool);pred[center]=(ss>=thr)&ok;intervals=P.intervals(pred);mm={}
                    for iou in [0,.1,.2,.5]:
                        tp=len(P.match(intervals,gt,iou));m=P.rate(tp,len(intervals)-tp,len(gt)-tp);m['fp_per_min']=m['fp']/minutes;mm[str(iou)]=m
                    results.append(dict(variant=v,fold=f,seed=seed,policy=policy,threshold_logit=thr,iou=mm,
                                        negative_frame_positive_rate=float(np.mean(pred[center[known&(labels[center]==-1)]]))))
                del front,head
    np.savez_compressed(OUT/'external_video9_scores.npz',**scores)
    dump('external_video9.json',dict(scope='Previously observed single-video exploratory pilot, not full EyeBlink8 validation.',
         corrected_models=True,n_gt=len(gt),minutes=minutes,runs=results,
         limits=['No threshold tuning; source validation thresholds only.',
                 'Binocular event IDs, not published per-eye completeness benchmark.',
                 'No independent external subject confidence intervals, no Pi latency.']))

def main():
    ap=argparse.ArgumentParser();ap.add_argument('command',choices=['watch','run']);args=ap.parse_args()
    try:
        if args.command=='watch':
            while True:
                status=read(HERE/'status.json') if (HERE/'status.json').exists() else {}
                if status.get('state')=='complete':break
                if status.get('state') in ['failed','stopped_for_input_leakage_fix']:raise RuntimeError('Training queue failed/stopped; postprocessing not run')
                state('waiting_for_corrected_training',training=status.get('completed',{}));time.sleep(60)
        import experiment_queue as Q
        for v in Q.VARIANTS:
            if len(Q.records(v))!=15:raise RuntimeError('All four corrected variants need 15 verified fits')
        import torch
        torch.set_num_threads(2);device='cuda' if torch.cuda.is_available() else 'cpu'
        state('running',stage='source_subjects');source_subjects()
        state('running',stage='landmark_providers');providers(device)
        state('running',stage='external_video9');pilot(device)
        state('complete',limits='Full external dataset, RNN raw data, deployment policy implementation and Pi remeasurement remain separate. Manuscript unchanged.')
    except Exception as exc:state('failed',error=repr(exc));raise

if __name__=='__main__':main()
