"""Frozen-threshold, source-trained external transfer pilot. No training here."""
from pathlib import Path
import sys, json, hashlib, argparse
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
import numpy as np
from src.v2.common import thresholds as TH
OUT=Path(__file__).resolve().parent
VIDEO=ROOT/'data/_legacy_public/eyeblink8/eyeblink8/9/27122013_152435_cam.avi'

def dump(name,obj):
    (OUT/'results').mkdir(exist_ok=True)
    (OUT/'results'/name).write_text(json.dumps(obj,indent=2,ensure_ascii=False,allow_nan=False),encoding='utf8')

def intervals(x):
    a=np.r_[False,np.asarray(x,bool),False].astype(int)
    return list(zip(np.flatnonzero(np.diff(a)==1),np.flatnonzero(np.diff(a)==-1)-1))

def match(pred,gt,min_iou=0):
    edges=[]
    for a,b in pred:
        cand=[]
        for j,(c,d) in enumerate(gt):
            overlap=max(0,min(b,d)-max(a,c)+1)
            iou=overlap/(max(b,d)-min(a,c)+1)
            if overlap>0 and iou>=min_iou: cand.append(j)
        edges.append(cand)
    owner={}
    def visit(i,seen):
        for j in edges[i]:
            if j in seen: continue
            seen.add(j)
            if j not in owner or visit(owner[j],seen):
                owner[j]=i
                return True
        return False
    for i in range(len(pred)): visit(i,set())
    return [(i,j) for j,i in owner.items()]

def rate(tp,fp,fn):
    return dict(tp=int(tp),fp=int(fp),fn=int(fn),precision=tp/(tp+fp) if tp+fp else 0,
                recall=tp/(tp+fn) if tp+fn else 0,f1=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0)

def preprocess():
    import cv2
    from src.v2.deploy.frontend import EyeFrontend
    from src.v2.dataset import crop as C
    rows=[]
    for line in VIDEO.with_suffix('.tag').read_text().splitlines():
        bits=line.split(':')
        if len(bits)==19 and bits[0].isdigit(): rows.append(bits)
    ann=np.array(rows)
    stamps=np.loadtxt(VIDEO.with_suffix('.txt'))
    assert np.array_equal(stamps[:,0],np.arange(len(stamps)))
    times=stamps[:,1]
    assert np.all(np.diff(times)>0)
    cap=cv2.VideoCapture(str(VIDEO))
    fps=cap.get(cv2.CAP_PROP_FPS); declared=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    crops=[];ears=[];valid=[]
    with EyeFrontend(lazy=False) as front:
        while True:
            ok,frame=cap.read()
            if not ok: break
            mesh=front.mesh(frame)
            g=None if mesh is None else C.crop_both_eyes(frame,mesh)[0]
            crops.append(g if g is not None else np.zeros((64,160),np.uint8))
            valid.append(g is not None)
            ears.append(list(C.ear_both(mesh).values()) if g is not None else [0]*4)
            if len(crops)%1000==0: print('decoded',len(crops),flush=True)
    cap.release()
    n=len(crops)
    assert n<=len(times), (n,len(times))
    extra_timestamps=len(times)-n
    # Annotation ends exactly at decoded frame n-1; recorder timestamps have an
    # unmatched tail. Preserve the discrepancy, never synthesize missing frames.
    assert int(ann[-1,0])==n-1
    times=times[:n]
    label=np.full(n,-2,int)
    label[ann[:,0].astype(int)]=ann[:,1].astype(int)
    # Native capture timestamp -> nearest sample on 30 Hz grid. Ties prefer earlier.
    target=np.arange(0,times[-1]+1e-9,1/30)
    right=np.searchsorted(times,target).clip(0,n-1);left=(right-1).clip(0,n-1)
    mapping=np.where(abs(times[left]-target)<=abs(times[right]-target),left,right)
    np.savez_compressed(OUT/'results/external_features.npz',crops=np.asarray(crops)[mapping],
                        ear=np.asarray(ears,np.float32)[mapping],valid=np.asarray(valid)[mapping],
                        label=label[mapping],time=target,source_frame=mapping)
    meta=dict(scope='one-video pilot, EyeBlink8 video 9; previously used unlabelled for timing',
              video=str(VIDEO),sha256=hashlib.sha256(VIDEO.read_bytes()).hexdigest(),
              annotation_sha256=hashlib.sha256(VIDEO.with_suffix('.tag').read_bytes()).hexdigest(),
              declared_frames=declared,decoded_frames=n,native_fps=fps,unmatched_trailing_timestamps=extra_timestamps,timestamp_duration=float(times[-1]-times[0]),
              target_frames=len(mapping),duplicate_samples=int(len(mapping)-len(np.unique(mapping))),
              unused_native_frames=int(n-len(np.unique(mapping))),crop_coverage=float(np.mean(valid)),
              annotation_range=[int(ann[0,0]),int(ann[-1,0])],annotated_frames=len(ann),
              original_blink_ids=int(len(np.unique(ann[ann[:,1].astype(int)>=0,1]))))
    dump('external_preprocessing.json',meta)

def evaluate():
    import torch
    from src.v2.model import encoder as E
    from src.v2.dataset import crop as C
    from src.v2.common import repro
    repro.seal(0);torch.set_num_threads(2)
    dev='cuda' if torch.cuda.is_available() else 'cpu'
    d=np.load(OUT/'results/external_features.npz')
    crops,ear,valid,label,times=[d[k] for k in ['crops','ear','valid','label','time']]
    n=len(label);end=np.arange(18,n);center=end-9
    windows=end[:,None]+np.arange(-18,1)
    mk=valid[windows].astype(np.float32)
    known=np.all(label[windows]!=-2,axis=1)
    eligible=(mk.sum(1)>=14)&known
    gt=[];events=[];excluded=[]
    for bid in np.unique(label[label>=0]):
        ix=np.flatnonzero(label==bid);a,b=int(ix[0]),int(ix[-1]);gt.append((a,b))
        c=(a+b)//2; j=c-9
        if j>=0 and j<len(end) and known[j]: events.append((j,1))
        else: excluded.append(int(bid))
    for c in range(9,n-9,19):
        j=c-9
        if known[j] and np.all(label[c-9:c+10]==-1):events.append((j,0))
    indices,y=np.asarray(events,int).T
    # Only complete, annotated GT intervals inside the evaluated center domain.
    centers_known=center[known]
    evaluated_gt=[(a,b) for a,b in gt if a>=centers_known.min() and b<=centers_known.max()]
    duration=(len(centers_known)/30)/60
    output=[];all_scores={}
    for variant,stem in [('ours','train_encoder_final'),('image_head','train_image_cnn_head_final'),('ear_head','train_encoder_final'),('ear_rule','train_encoder_final')]:
        source=json.loads((ROOT/f'results/v2/{stem}.json').read_text(encoding='utf8'))
        for run in source['runs']:
            fold,seed=run['fold'],run['seed'];key=f'{variant}_fold{fold}_seed{seed}'
            folder=ROOT/f'models/v2/{stem}/fold{fold}_seed{seed}'
            thr=run[variant if variant.startswith('ear') else 'ours']['thr']
            if variant=='ear_rule':
                v=np.where(mk>0,ear[windows,2],np.nan)
                fi=np.argmax(mk>0,axis=1);la=18-np.argmax((mk>0)[:,::-1],axis=1)
                edge=(v[np.arange(len(v)),fi]+v[np.arange(len(v)),la])/2
                with np.errstate(all='ignore'):
                    scores=1-np.min(np.where(mk>0,v,np.inf),axis=1)/edge
                scores=np.nan_to_num(scores,nan=0,posinf=0,neginf=0)
            else:
                front=E.build() if variant=='ours' else E.build_image_cnn(16) if variant=='image_head' else E.build_ear_frontend()
                head=E.build_head()
                front.load_state_dict(torch.load(folder/('earhead_front.pt' if variant=='ear_head' else 'encoder.pt'),map_location='cpu',weights_only=True))
                head.load_state_dict(torch.load(folder/('earhead_head.pt' if variant=='ear_head' else 'head.pt'),map_location='cpu',weights_only=True))
                front.to(dev).eval();head.to(dev).eval()
                with torch.no_grad():
                    zs=[]
                    for i in range(0,n,128):
                        x=ear[i:i+128] if variant=='ear_head' else C.batch_input(crops[i:i+128])
                        zs.append(front(torch.from_numpy(x).to(dev)).cpu().numpy())
                    z=np.concatenate(zs)
                    # Missing positions have zero features; mask applied identically in TCN.
                    z[~valid]=0
                    ss=[]
                    for i in range(0,len(windows),256):
                        ss.append(head(torch.from_numpy(z[windows[i:i+256]]).to(dev),torch.from_numpy(mk[i:i+256]).to(dev)).cpu().numpy())
                    scores=np.concatenate(ss)
            prediction=(scores>=thr)&eligible
            pred_full=np.zeros(n,bool);pred_full[center]=prediction
            pred_intervals=intervals(pred_full)
            neg=known & (label[center]==-1)
            continuous={}
            for tolerance in [0,.1]:
                pairs=match(pred_intervals,evaluated_gt,tolerance)
                tp=len(pairs);r=rate(tp,len(pred_intervals)-tp,len(evaluated_gt)-tp)
                delays=[float(times[pred_intervals[i][0]+9]-times[evaluated_gt[j][0]]) for i,j in pairs]
                r.update(false_detections_per_minute=r['fp']/duration,alert_delay_seconds_median=float(np.median(delays)) if delays else None,
                         alert_delay_seconds_p90=float(np.percentile(delays,90)) if delays else None)
                continuous[str(tolerance)]=r
            pe=prediction[indices];scorable=eligible[indices]
            e=rate(np.sum(pe&(y==1)),np.sum(pe&(y==0)),np.sum((~pe)&(y==1)))
            e['ap_scorable']=float(TH.average_precision(scores[indices][scorable],y[scorable]))
            e['coverage']=float(np.mean(scorable))
            output.append(dict(variant=variant,fold=fold,seed=seed,threshold=float(thr),event_window=e,
                               continuous=continuous,decision_coverage=float(np.mean(eligible[known])),
                               negative_frame_positive_rate=float(np.mean(prediction[neg]))))
            all_scores[key]=scores.astype(np.float32)
            print(key,e['ap_scorable'],continuous['0']['f1'],flush=True)
    dump('external_pilot.json',dict(scope='descriptive video-9 pilot, no population inference',
         n_source_runs_per_method=15,n_external_videos=1,annotation_unit='bilateral blink ID, not per-eye completeness',
         event_positive=int(y.sum()),event_negative=int((y==0).sum()),excluded_event_ids=excluded,
         continuous_gt=len(evaluated_gt),total_resampled_blinks=len(gt),annotated_center_minutes=duration,
         runs=output))
    np.savez_compressed(OUT/'results/external_scores.npz',**all_scores,center=center,end=end,eligible=eligible,
                        event_indices=indices,event_y=y,label=label,time=times,source_frame=d['source_frame'])

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('mode',choices=['preprocess','evaluate']);a=ap.parse_args()
    (OUT/'results').mkdir(exist_ok=True)
    preprocess() if a.mode=='preprocess' else evaluate()
