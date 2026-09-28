"""Re-extract the fixed 8-person sample under explicit float32 crop geometry."""
from pathlib import Path
import sys,json,zipfile,time
import numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
HERE=Path(__file__).resolve().parent;OUT=HERE/'recrop/provider_sample';OUT.mkdir(parents=True,exist_ok=True)
OLD=ROOT/'work/nonpi_2026-09-21/submission_audit/provider_sample'
from src.v2.dataset import crop as C,mebal2 as M
from src.v2.deploy.frontend import EyeFrontend
idx=np.load(ROOT/'data/processed/v2/index.npz');stored=np.load(ROOT/'data/processed/v2/frames_m22.npy',mmap_mode='r')
report=[];start=time.time()
for file in sorted(OLD.glob('landmark_U*_features.npz')):
    u=int(file.stem.split('_U')[1].split('_')[0]);d=np.load(file);ids=d['event_ids'];wanted=set(d['frame_ids'].ravel().tolist())
    cache=OUT/file.name; rep=OUT/f'geometry_U{u}.json'
    if cache.exists() and rep.exists():report.append(json.loads(rep.read_text()));continue
    print('recrop user',u,flush=True)
    with zipfile.ZipFile(ROOT/'data/mEBAL2/Processed_Data.zip') as z:
        with z.open(f'Processed_Data/User {u}/box.csv') as f:_,box=M.scan_stream(f,wanted,4,1)
        with z.open(f'Processed_Data/User {u}/landmarks.csv') as f:_,lm=M.scan_stream(f,wanted,3,M.N_MESH)
    warm=set()
    for e in ids:warm.update(range(max(0,int(idx['e_start'][e])-60),int(idx['e_end'][e])+1))
    video=OLD/f'raw/U{u}_color.mp4'
    if not video.exists():video=ROOT/f'work/nonpi_2026-09-21/submission_audit/raw/U{u}_color.mp4'
    rows={int(idx['f_frame_idx'][r]):int(r) for r in np.flatnonzero(idx['f_subject']==u)}
    prev=-2;fe=None;features={};matched=0;bad=[]
    try:
        for f,frame in M.iter_frames(str(video),warm):
            if f!=prev+1:
                if fe:fe.close()
                fe=EyeFrontend(lazy=False)
            prev=f;mesh=fe.mesh(frame)
            if f not in wanted:continue
            g,_=(None,None) if mesh is None else fe.crop_from_mesh(frame,mesh)
            features[f]=(np.zeros((64,160),np.uint8) if g is None else g,g is not None)
            sel=M.select_face(f,lm.get(f),box.get(f))
            if f in rows and sel.status==M.OK:
                ref,_=C.crop_both_eyes(frame,sel.mesh[:,:2]);matched+=1
                if ref is None or not np.array_equal(ref,stored[rows[f]]):bad.append(f)
    finally:
        if fe:fe.close()
    assert set(features)==wanted
    crops=np.array([[features[int(f)][0] for f in win] for win in d['frame_ids']])
    mask=np.array([[features[int(f)][1] for f in win] for win in d['frame_ids']],np.float32)
    np.savez_compressed(cache,event_ids=ids,crops=crops,mask=mask,frame_ids=d['frame_ids'])
    r=dict(subject=u,source_frames_checked=matched,source_mismatch_frames=bad,
        fresh_changed_crop_positions=int(np.sum(np.any(crops!=d['crops'],axis=(-1,-2)))),
        fresh_changed_valid_positions=int(np.sum(mask!=d['mask'])))
    rep.write_text(json.dumps(r,indent=2));report.append(r);print(json.dumps(r),flush=True)
summary=dict(users=report,source_frames_checked=sum(r['source_frames_checked'] for r in report),
             source_mismatches=sum(len(r['source_mismatch_frames']) for r in report),seconds=time.time()-start,
             limits='Frozen 8-person/160-event sample only; 60-frame warmup per segment; no new participants.')
(HERE/'recrop_summary.json').write_text(json.dumps(summary,indent=2),encoding='utf8')
print('REINFER corrected models',flush=True)
sys.path.insert(0,str(ROOT/'work/nonpi_2026-09-21_maskfix'));import post_train as P
import torch;torch.set_num_threads(2)
P.AUDIT=HERE/'recrop';P.OUT=HERE/'recrop_results';P.OUT.mkdir(exist_ok=True)
P.providers('cuda' if torch.cuda.is_available() else 'cpu')
print('complete',flush=True)
