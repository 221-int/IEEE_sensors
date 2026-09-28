"""CPU-only paired landmark-provider diagnostic on prespecified failure cases.

Not an unbiased test set: cases were selected using held-out source errors.
FaceMesh is warmed up for 60 preceding native frames per disjoint segment.
"""
from pathlib import Path
import sys,json,zipfile,shutil,time,os
os.environ.setdefault('OMP_NUM_THREADS','2')
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
import numpy as np,cv2
from src.v2.dataset import crop as C,mebal2 as M
from src.v2.common import splits
from src.v2.deploy.frontend import EyeFrontend
OUT=Path(__file__).resolve().parent/'submission_audit'

def dump(name,obj):
    path=OUT/name;tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(obj,indent=2,allow_nan=False),encoding='utf8');tmp.replace(path)

def run_user(user,event_ids,selection_description='Error-selected cases, descriptive only; no model selection or generalization inference.'):
    idx=np.load(ROOT/'data/processed/v2/index.npz');stored=np.load(ROOT/'data/processed/v2/frames_m22.npy',mmap_mode='r')
    frame_ids=sorted({int(f) for e in event_ids for f in range(int(idx['e_start'][e]),int(idx['e_end'][e])+1)})
    want=set(frame_ids);warm=set()
    for e in event_ids:warm.update(range(max(0,int(idx['e_start'][e])-60),int(idx['e_end'][e])+1))
    direct=ROOT/f'data/raw/mEBAL2/User {user}/RealSense/Color_Webcam/color.mp4'
    reused=Path(__file__).resolve().parent/f'submission_audit/raw/U{user}_color.mp4'
    if direct.exists():video=direct
    elif reused.exists():video=reused
    else:
        video=OUT/f'raw/U{user}_color.mp4';video.parent.mkdir(exist_ok=True)
        member=f'User {user}/RealSense/Color_Webcam/color.mp4'
        for zp in sorted((ROOT/'data/mEBAL2').glob('Webcams-EEG *.zip')):
            with zipfile.ZipFile(zp) as z:
                if member not in z.namelist():continue
                expected=z.getinfo(member).file_size
                if not video.exists() or video.stat().st_size!=expected:
                    print('extract',user,expected,flush=True)
                    partial=video.with_suffix('.partial')
                    with z.open(member) as src,partial.open('wb') as dst:shutil.copyfileobj(src,dst,8<<20)
                    partial.replace(video)
                break
        if not video.exists():raise FileNotFoundError(member)
    print('landmark CSV',user,flush=True)
    with zipfile.ZipFile(ROOT/'data/mEBAL2/Processed_Data.zip') as z:
        with z.open(f'Processed_Data/User {user}/box.csv') as f:_,box=M.scan_stream(f,want,4,1)
        with z.open(f'Processed_Data/User {user}/landmarks.csv') as f:_,lm=M.scan_stream(f,want,3,M.N_MESH)
    feats={};source_comparison=[];prev=-2;front=None
    row_of={int(idx['f_frame_idx'][r]):int(r) for r in np.flatnonzero(idx['f_subject']==user)}
    try:
        for f,frame in M.iter_frames(str(video),warm):
            if f!=prev+1:
                if front:front.close()
                front=EyeFrontend(lazy=False)
            prev=f;mesh=front.mesh(frame)
            if f not in want:continue
            g,meta=(None,None) if mesh is None else front.crop_from_mesh(frame,mesh)
            feats[f]=(np.zeros((64,160),np.uint8) if g is None else g,g is not None,
                      np.zeros(4,np.float32) if g is None else np.array(list(C.ear_both(mesh).values()),np.float32))
            sel=M.select_face(f,lm.get(f),box.get(f))
            if f in row_of and sel.status==M.OK:
                ref,_=front.crop_from_mesh(frame,sel.mesh[:,:2]);cached=np.asarray(stored[row_of[f]])
                source_comparison.append(dict(frame=f,stored_equal=bool(ref is not None and np.array_equal(ref,cached)),
                  fresh_valid=g is not None,crop_mae=None if g is None else float(np.abs(g.astype(float)-cached).mean())))
            # Original context at event centers, diagnostic visual only.
            if any(f==int(idx['e_start'][e])+9 for e in event_ids):
                cv2.imwrite(str(OUT/f'U{user}_raw_frame{f}.jpg'),cv2.resize(frame,(960,540)))
            if len(feats)%100==0:print('decoded target',user,len(feats),flush=True)
    finally:
        if front:front.close()
    assert set(feats)==want,(len(feats),len(want))
    windows=np.array([[f for f in range(int(idx['e_start'][e]),int(idx['e_end'][e])+1)] for e in event_ids])
    assert windows.shape[1]==19
    new=np.array([[feats[f][0] for f in w] for w in windows]);mask=np.array([[feats[f][1] for f in w] for w in windows],np.float32)
    ear=np.array([[feats[f][2] for f in w] for w in windows],np.float32)
    old_rows=idx['e_rows'][event_ids];old_mask=(old_rows>=0).astype(np.float32)
    old_flat=np.maximum(old_rows,0).ravel()
    # Preserve actual crops/masks to permit later independent replay.
    np.savez_compressed(OUT/f'landmark_U{user}_features.npz',event_ids=event_ids,crops=new,mask=mask,ear=ear,frame_ids=windows)
    import torch
    torch.set_num_threads(2)
    from src.v2.model import encoder as E
    fold=splits.load_folds()[user];preds=[]
    for variant,stem,key in [('ours','train_encoder_final','ours'),('image_head','train_image_cnn_head_final','ours'),('ear_head','train_encoder_final','ear_head')]:
        result=json.loads((ROOT/f'results/v2/{stem}.json').read_text(encoding='utf8'))
        for run in result['runs']:
            if run['fold']!=fold:continue
            modeldir=ROOT/f'models/v2/{stem}/fold{fold}_seed{run["seed"]}'
            front=E.build() if variant=='ours' else E.build_image_cnn(16) if variant=='image_head' else E.build_ear_frontend()
            head=E.build_head()
            front.load_state_dict(torch.load(modeldir/('earhead_front.pt' if variant=='ear_head' else 'encoder.pt'),map_location='cpu',weights_only=True));front.eval()
            head.load_state_dict(torch.load(modeldir/('earhead_head.pt' if variant=='ear_head' else 'head.pt'),map_location='cpu',weights_only=True));head.eval()
            with torch.no_grad():
                n,t=mask.shape;x=ear.reshape(n*t,4) if variant=='ear_head' else C.batch_input(new.reshape(n*t,64,160))
                zs=[]
                for j in range(0,len(x),32):zs.append(front(torch.from_numpy(x[j:j+32])))
                logits=head(torch.cat(zs).reshape(n,t,-1),torch.from_numpy(mask)).numpy()
                old_x=idx['f_ear'][old_flat].copy() if variant=='ear_head' else C.batch_input(np.asarray(stored[old_flat]))
                if variant=='ear_head':old_x[old_mask.ravel()==0]=0
                old_z=[front(torch.from_numpy(old_x[j:j+32])) for j in range(0,len(old_x),32)]
                old_logits=head(torch.cat(old_z).reshape(n,t,-1),torch.from_numpy(old_mask)).numpy()
            test_ids=np.flatnonzero(idx['e_valid']&(idx['e_fold']==fold));positions=np.searchsorted(test_ids,event_ids)
            assert np.array_equal(test_ids[positions],event_ids)
            with np.load(ROOT/f'results/v2/{stem}_scores/fold{fold}_seed{run["seed"]}.npz') as sidecar:
                saved=sidecar[key][positions]
            preds.append(dict(variant=variant,seed=run['seed'],fold=fold,threshold=run[key]['thr'],
                              logits=logits.tolist(),predictions=(logits>=run[key]['thr']).astype(int).tolist(),
                              reference_cpu_logits=old_logits.tolist(),reference_saved_logits=saved.tolist(),
                              cpu_reference_max_error=float(np.max(np.abs(old_logits-saved))),
                              cpu_reference_prediction_flips=int(np.sum((old_logits>=run[key]['thr'])!=(saved>=run[key]['thr'])))))
    report=dict(user=user,event_ids=event_ids,labels=idx['e_is_blink'][event_ids].astype(int).tolist(),
                valid_frames=mask.sum(1).astype(int).tolist(),eligible=(mask.sum(1)>=14).tolist(),
                source_crop_comparison=source_comparison,predictions=preds,
                limitations=[selection_description,
                             'Fresh FaceMesh with 60-frame warmup per segment, not uninterrupted full-session tracking.',
                             'CPU numerical differences from source GPU inference are possible.',
                             'Scores from windows with fewer than 14 valid frames are diagnostic only and must be abstained in protocol-aligned evaluation.'])
    dump(f'landmark_U{user}.json',report);print('finished U',user,flush=True)

def main():
    selected=json.loads((OUT/'review_selection.json').read_text());start=time.time()
    try:
        for user,ids in selected.items():
            if (OUT/f'landmark_U{user}.json').exists():continue
            dump('landmark_status.json',dict(state='running',user=int(user),started=time.strftime('%Y-%m-%dT%H:%M:%S'),pid=os.getpid()))
            run_user(int(user),ids)
        dump('landmark_status.json',dict(state='complete',seconds=time.time()-start))
    except Exception as exc:
        dump('landmark_status.json',dict(state='failed',error=repr(exc)));raise

if __name__=='__main__':main()
