"""Audit frozen sidecars and build post-hoc diagnostic cases; no GPU or fitting."""
from pathlib import Path
import sys, json, hashlib, zipfile, csv
import numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from src.v2.common import splits, thresholds as TH
from src.v2.dataset import crop as C
OUT=Path(__file__).resolve().parent/'submission_audit'

def dump(name,obj):
    (OUT/name).write_text(json.dumps(obj,indent=2,ensure_ascii=False,allow_nan=False),encoding='utf8')

def metrics(y,p):
    tp=int(np.sum((y==1)&p));fp=int(np.sum((y==0)&p))
    fn=int(np.sum((y==1)&~p));tn=int(np.sum((y==0)&~p))
    return dict(tp=tp,fp=fp,fn=fn,tn=tn,recall=tp/(tp+fn) if tp+fn else None,
                f1=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else None)

def main():
    OUT.mkdir(exist_ok=True)
    idx=np.load(ROOT/'data/processed/v2/index.npz')
    original=np.flatnonzero(idx['e_valid']);rows=idx['e_rows'][original]
    sub=idx['e_subject'][original];y=idx['e_is_blink'][original]
    assign=splits.load_folds();fold=np.array([assign[int(s)] for s in sub])
    assert np.array_equal(fold,idx['e_fold'][original])
    valid=rows>=0
    assert np.all(idx['f_subject'][rows[valid]]==np.broadcast_to(sub[:,None],rows.shape)[valid])
    assert np.all(idx['f_frame_idx'][rows[valid]]==
                  (idx['e_start'][original,None]+np.arange(19))[valid])
    assert np.array_equal((~valid).sum(1),idx['e_n_missing'][original])
    assert np.all(valid.sum(1)>=14)
    split_checks=[]
    for f in range(5):
        masks=splits.subject_masks(sub,assign,splits.fold_rotation(f))
        sets={k:set(sub[v].tolist()) for k,v in masks.items()}
        frame_sets={k:set(rows[v][rows[v]>=0].tolist()) for k,v in masks.items()}
        pairs=[('train','val'),('train','test'),('val','test')]
        assert all(not sets[a]&sets[b] and not frame_sets[a]&frame_sets[b] for a,b in pairs)
        assert np.all(sum(m.astype(int) for m in masks.values())==1)
        split_checks.append(dict(fold=f,subjects={k:len(v) for k,v in sets.items()},
                                 events={k:int(v.sum()) for k,v in masks.items()},overlap=0))
    models={'ours':('train_encoder_final','ours'),'image_head':('train_image_cnn_head_final','ours'),
            'ear_head':('train_encoder_final','ear_head')}
    scores={k:np.full((3,len(y)),np.nan) for k in models}
    predictions={k:np.zeros((3,len(y)),bool) for k in models}
    checked=[]
    for model,(stem,key) in models.items():
        result=json.loads((ROOT/f'results/v2/{stem}.json').read_text())
        assert {(r['fold'],r['seed']) for r in result['runs']}=={(f,s) for f in range(5) for s in range(3)}
        for r in result['runs']:
            f,s=r['fold'],r['seed'];sel=fold==f
            with np.load(ROOT/f'results/v2/{stem}_scores/fold{f}_seed{s}.npz') as z:
                assert np.array_equal(z['subject'],sub[sel]) and np.array_equal(z['y'],y[sel])
                assert int(z['fold'])==f and int(z['seed'])==s and np.isfinite(z[key]).all()
                scores[model][s,sel]=z[key];predictions[model][s,sel]=z[key]>=r[key]['thr']
                ap=TH.average_precision(z[key],y[sel]);assert abs(ap-r[key]['pr_auc'])<1e-10
                m=metrics(y[sel],predictions[model][s,sel])
                assert all(m[k]==r[key][k] for k in ['tp','fp','fn','tn'])
                checked.append(dict(model=model,fold=f,seed=s,ap_error=abs(ap-r[key]['pr_auc']),confusion_equal=True))
        assert np.isfinite(scores[model]).all()
    frames=np.load(ROOT/'data/processed/v2/frames_m22.npy',mmap_mode='r')
    # All subjects represented; 16 deterministic frames per subject.
    norm_rows=np.concatenate([np.flatnonzero(idx['f_subject']==s)[np.linspace(0,np.sum(idx['f_subject']==s)-1,16,dtype=int)] for s in np.unique(sub)])
    norm_diff=0.
    for chunk in np.array_split(norm_rows,16):
        ims=np.asarray(frames[chunk]);a=C.batch_input(ims);b=np.concatenate([C.to_input_tensor(im) for im in ims])
        norm_diff=max(norm_diff,float(np.max(np.abs(a-b))))
    assert norm_diff==0
    vote={k:v.sum(0)>=2 for k,v in predictions.items()}
    finite_mean=lambda a:float(np.mean(a[np.isfinite(a)])) if np.isfinite(a).any() else None
    md={}
    for field in ['f_span_px','f_brightness_m22','f_contrast_m22','f_sharpness_m22','f_tilt_deg']:
        a=np.where(valid,idx[field][np.maximum(rows,0)],np.nan)
        md[field]=np.nanmedian(a,axis=1)
    people=[];events=[];selections={}
    for u in np.unique(sub):
        ix=sub==u
        person=dict(subject=int(u),n_events=int(ix.sum()),n_blink=int(y[ix].sum()),
                    n_missing_events=int((~valid[ix]).any(1).sum()),models={})
        for model in models:
            aps=[TH.average_precision(scores[model][s,ix],y[ix]) for s in range(3)]
            person['models'][model]=dict(ap_seed=aps,mean_ap=float(np.mean(aps)),
                                        diagnostic_majority=metrics(y[ix],vote[model][ix]))
        err=vote['ours']!=y
        person['diagnostic_disagreement']=dict(ours_wrong_image_correct=int(np.sum(ix&err&(vote['image_head']==y))),
                                             ours_wrong_ear_correct=int(np.sum(ix&err&(vote['ear_head']==y))))
        person['metadata_by_ours_error']={str(b):{k:finite_mean(v[ix&(err==b)]) for k,v in md.items()} for b in [False,True]}
        people.append(person)
        if u not in (1,54): continue
        picked=[]
        for label in [0,1]:
            consistent=(predictions['ours']!=y).sum(0)==3
            es=np.flatnonzero(ix&(y==label)&consistent)[:4]
            cs=np.flatnonzero(ix&(y==label)&~err)[:2]
            picked.extend(es.tolist()+cs.tolist())
        selections[str(u)]=[int(original[i]) for i in picked]
        for i in np.flatnonzero(ix):
            events.append(dict(subject=int(u),event_index=int(original[i]),event_id=int(idx['e_event_id'][original[i]]),
                               start_frame=int(idx['e_start'][original[i]]),end_frame=int(idx['e_end'][original[i]]),
                               label=int(y[i]),n_missing=int((~valid[i]).sum()),
                               metadata={k:float(v[i]) for k,v in md.items()},
                               seed_predictions={k:v[:,i].astype(int).tolist() for k,v in predictions.items()},
                               seed_logits={k:v[:,i].tolist() for k,v in scores.items()},
                               selected_for_review=i in picked))
        import cv2
        for i in picked:
            canvas=np.full((4*86+42,5*160,3),245,np.uint8)
            title=f'U{u} e{original[i]} y{y[i]} O{vote["ours"][i]:.0f} I{vote["image_head"][i]:.0f} E{vote["ear_head"][i]:.0f} seedErrors{int(np.sum(predictions["ours"][:,i]!=y[i]))}/3'
            cv2.putText(canvas,title,(6,25),cv2.FONT_HERSHEY_SIMPLEX,.52,(0,0,0),1)
            for t,row in enumerate(rows[i]):
                yy=42+(t//5)*86;xx=(t%5)*160
                tile=cv2.cvtColor(np.asarray(frames[row]),cv2.COLOR_GRAY2BGR) if row>=0 else np.zeros((64,160,3),np.uint8)
                canvas[yy+20:yy+84,xx:xx+160]=tile
                cv2.putText(canvas,f't{t} f{int(idx["e_start"][original[i]])+t}',(xx+2,yy+14),cv2.FONT_HERSHEY_SIMPLEX,.35,(0,0,0),1)
            cv2.imwrite(str(OUT/f'U{u}_event{original[i]}.png'),canvas)
    archives=[]
    for p in (Path.home()/'Downloads').glob('*.zip'):
        if any(k in p.name for k in ['표1','전력']):
            with zipfile.ZipFile(p) as z:
                members=[dict(name=i.filename,bytes=i.file_size) for i in z.infolist() if not i.is_dir()]
                archives.append(dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest(),members=members,
                                     raw_files=[m for m in members if Path(m['name']).suffix.lower() in ['.json','.npz','.pt','.pth']]))
    dump('integrity.json',dict(n_events=len(y),n_subjects=len(people),folds=split_checks,sidecar_checks=checked,
         normalization=dict(n=len(norm_rows),max_abs_diff=norm_diff),row_subject_frame_alignment=True,
         limitations=['Sidecars have labels/subject order but no unique event IDs; their order is checked against trainer iteration order, not cryptographically recoverable event IDs.',
                     'No evidence of cross-split frame-index overlap; this is not a pixel-duplicate or identity-verification audit.',
                     'Threshold metric reproduction does not independently prove validation-only selection; training code is reviewed separately.']))
    dump('subjects.json',people);dump('failure_events.json',events);dump('review_selection.json',selections)
    dump('rnn_handoff_inventory.json',archives)
    print(json.dumps(dict(n_events=len(y),folds=len(split_checks),sidecars=len(checked),normalization_max_diff=norm_diff,
                         cases=[p for p in people if p['subject'] in [1,54]],selected=selections),indent=2))

if __name__=='__main__': main()
