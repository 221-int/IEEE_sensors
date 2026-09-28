"""Regression evidence for masked-frame contamination and its isolated correction."""
from pathlib import Path
import sys,copy,json
import numpy as np,torch
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from src.v2.train_encoder import Bundle
from src.v2.common import splits
from src.v2.model import encoder as E
from maskfix import MaskSafeBundle
torch.set_num_threads(2)
OUT=Path(__file__).resolve().parent

def mechanism(x,mask,y,builder):
    n,t=x.shape[:2];zero=x.copy();zero[mask==0]=0
    torch.manual_seed(0);f=builder();h=E.build_head();f2=copy.deepcopy(f);h2=copy.deepcopy(h)
    zlist=[];slist=[];glist=[]
    for front,head,xx in [(f,h,x),(f2,h2,zero)]:
        front.train();head.train();torch.manual_seed(17)
        z=front(torch.from_numpy(xx.reshape(n*t,1,64,160))).reshape(n,t,16)
        s=head(z,torch.from_numpy(mask))
        torch.nn.functional.binary_cross_entropy_with_logits(s,torch.from_numpy(y).float()).backward()
        zlist.append(z.detach());slist.append(s.detach());glist.append(torch.cat([p.grad.flatten() for p in list(front.parameters())+list(head.parameters()) if p.grad is not None]))
    return dict(valid_embedding_max_difference=float((zlist[0][mask>0]-zlist[1][mask>0]).abs().max()),
                logit_max_difference=float((slist[0]-slist[1]).abs().max()),gradient_max_difference=float((glist[0]-glist[1]).abs().max()))

def main():
    old=Bundle(str(ROOT/'data/processed/v2'));safe=MaskSafeBundle(str(ROOT/'data/processed/v2'));assign=splits.load_folds()
    details=[]
    for fold in range(5):
        m=splits.subject_masks(old.subject,assign,splits.fold_rotation(fold));q=m['train']&(old.mask.sum(1)<19)
        details.append(dict(fold=fold,placeholder_subject=int(old.idx['f_subject'][0]),
          placeholder_subject_role=next(k for k,v in splits.fold_rotation(fold).items() if assign[int(old.idx['f_subject'][0])] in v),
          training_events_with_missing=int(q.sum()),placeholder_frames=int(np.sum(19-old.mask[q].sum(1)))))
    ids=np.flatnonzero(splits.subject_masks(old.subject,assign,splits.fold_rotation(1))['train']&(old.mask.sum(1)<19))[:8]
    original,mask,y=old.batch(ids)
    class Guard:
        def __getitem__(self,rows):
            assert np.all(np.asarray(rows)!=0),'Held-out frame0 was read for a missing position'
            return old.frames[rows]
    safe.frames=Guard();fixed,m2,y2=safe.batch(ids)
    assert np.array_equal(mask,m2) and np.array_equal(y,y2)
    assert np.array_equal(original[mask>0],fixed[mask>0])
    assert np.all(fixed[mask==0]==0)
    models={}
    for name,builder in [('vpres',lambda:E.build('vpres')),('vdrop',lambda:E.build('vdrop')),('image_head',lambda:E.build_image_cnn(16))]:
        legacy=mechanism(original,mask,y,builder);corrected=mechanism(fixed,mask,y,builder)
        if name!='image_head':assert legacy['gradient_max_difference']>0
        else:assert legacy['gradient_max_difference']==0
        assert all(v==0 for v in corrected.values())
        models[name]=dict(legacy=legacy,corrected=corrected)
    # Check valid tensors and masks across all source events in bounded batches.
    safe.frames=old.frames;maxdiff=0.;checked=0
    for start in range(0,len(old.y),128):
        q=np.arange(start,min(start+128,len(old.y)));a,m,_=old.batch(q);b,mm,_=safe.batch(q)
        assert np.array_equal(m,mm) and np.all(b[m==0]==0)
        # Frame normalization is independent across images.
        assert np.array_equal(a[m>0],b[m>0]);checked+=len(q)
    report=dict(pass_all=True,scope='Input-content leakage regression; no statement about final AP impact.',folds=details,
                mechanism_batch_events=ids.tolist(),models=models,all_event_valid_inputs_unchanged=checked,
                guard_no_placeholder_access=True,all_missing_tensors_zero=True)
    (OUT/'preflight.json').write_text(json.dumps(report,indent=2),encoding='utf8');print(json.dumps(report,indent=2))

if __name__=='__main__':main()
