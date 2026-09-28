"""Diagnose the two cached-crop mismatches without altering crop.py or data."""
from pathlib import Path
import sys,json,zipfile,inspect,importlib.util
import numpy as np,cv2
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
HERE=Path(__file__).resolve().parent
from src.v2.dataset import crop as C,mebal2 as M

def main():
    idx=np.load(ROOT/'data/processed/v2/index.npz'); stored=np.load(ROOT/'data/processed/v2/frames_m22.npy',mmap_mode='r')
    spec=importlib.util.spec_from_file_location('preserved_crop',HERE/'before/src/v2/dataset/crop.py')
    original=importlib.util.module_from_spec(spec);sys.modules[spec.name]=original;spec.loader.exec_module(original)
    source=inspect.getsource(original.crop_both_eyes)
    # NumPy 2 weak scalar promotion: a float32 scalar +/- a Python float stays
    # float32; NumPy 1.26 promotes this operation to float64. Keep every other
    # crop operation identical to the original function.
    weak=source
    for axis,dim in [(0,'w'),(1,'h')]:
        for op in ['-','+']:
            term=f'center[{axis}] {op} crop_{dim} / 2'
            weak=weak.replace(term,f'np.float32(center[{axis}] {op} np.float32(crop_{dim} / 2))')
    space=dict(vars(C));exec(weak,space);weakcrop=space['crop_both_eyes']
    reports=[]
    for u,f in [(19,33969),(55,49892)]:
        cache=HERE/f'crop_case_U{u}_{f}.npz'
        if not cache.exists():
            with zipfile.ZipFile(ROOT/'data/mEBAL2/Processed_Data.zip') as z:
                with z.open(f'Processed_Data/User {u}/box.csv') as stream: _,box=M.scan_stream(stream,{f},4,1)
                with z.open(f'Processed_Data/User {u}/landmarks.csv') as stream: _,lm=M.scan_stream(stream,{f},3,M.N_MESH)
            sel=M.select_face(f,lm.get(f),box.get(f));assert sel.status==M.OK
            video=ROOT/f'work/nonpi_2026-09-21/submission_audit/provider_sample/raw/U{u}_color.mp4'
            frame=next(M.iter_frames(str(video),[f]))[1]
            np.savez_compressed(cache,frame=frame,mesh=sel.mesh[:,:2])
        d=np.load(cache);row=int(np.flatnonzero((idx['f_subject']==u)&(idx['f_frame_idx']==f))[0]);target=stored[row]
        variants={}
        for name,func in [('current_numpy126',original.crop_both_eyes),('numpy2_scalar_promotion',weakcrop)]:
            g,meta=func(d['frame'],d['mesh']);diff=np.abs(g.astype(float)-target)
            variants[name]=dict(equal=bool(np.array_equal(g,target)),max_diff=float(diff.max()),mean_diff=float(diff.mean()),changed_pixels=int(np.sum(diff!=0)),meta=meta.as_dict())
        reports.append(dict(subject=u,frame=f,row=row,variants=variants))
        print(json.dumps(reports[-1]),flush=True)
    (HERE/'crop_recheck.json').write_text(json.dumps(dict(numpy=np.__version__,opencv=cv2.__version__,cases=reports),indent=2),encoding='utf8')

if __name__=='__main__':main()
