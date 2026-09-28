"""Resumable single-GPU experiment queue. Never edits manuscript files."""
from pathlib import Path
import argparse,copy,datetime,hashlib,json,os,shutil,subprocess,sys,time,traceback
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
OUT=Path(__file__).resolve().parent
import numpy as np
from src.v2.common import thresholds as TH,stats,splits
from src.v2.compare_variants import load_variant,align
PAIRS=[(f,s) for f in range(5) for s in range(3)]
VARIANTS=['vpres','image_head','vdrop','gap']
OLD=ROOT/'work/nonpi_2026-09-18'
def now():return datetime.datetime.now().astimezone().isoformat()
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf8'))
def write(p,obj):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf8');os.replace(tmp,p)
def copy_once(src,dst):
    src,dst=Path(src),Path(dst);dst.parent.mkdir(parents=True,exist_ok=True)
    if dst.exists():
        if sha(src)!=sha(dst):raise RuntimeError(f'Existing preserved artifact differs: {dst}')
    else:shutil.copy2(src,dst)
def record_path(v,f,s):return OUT/f'records/{v}/fold{f}_seed{s}.json'
def validate_record(payload,expected):
    r=payload['run'];key=(int(r['fold']),int(r['seed']))
    if key!=expected:raise ValueError('Wrong fold/seed')
    p=Path(payload['scores']);d=np.load(p,allow_pickle=False)
    if (int(d['fold']),int(d['seed']))!=key:raise ValueError('Sidecar fold/seed mismatch')
    ref=np.load(ROOT/f'results/v2/train_encoder_final_scores/fold{key[0]}_seed{key[1]}.npz')
    for name in ['y','subject']:
        if not np.array_equal(d[name],ref[name]):raise ValueError('Held-out order/labels differ')
    if len(d['y'])!=r['n_test']:raise ValueError('Wrong number of events')
    for name in ['ours','ear_head','ear_rule']:
        if not np.all(np.isfinite(d[name])):raise ValueError('Nonfinite scores')
        ap=float(TH.average_precision(d[name],d['y']))
        if abs(ap-r[name]['pr_auc'])>1e-10:raise ValueError('Score/AP mismatch')
    for name in ['encoder.pt','head.pt','earhead_front.pt','earhead_head.pt']:
        if not (Path(payload['models'])/name).is_file():raise ValueError('Missing checkpoint')
    for name,value in payload['hashes'].items():
        if sha(name)!=value:raise ValueError('Completed artifact changed: '+name)
    return payload
def make_record(variant,r,score,models,env,origin):
    target=OUT/f'results/train_{variant}_scores/fold{r["fold"]}_seed{r["seed"]}.npz'
    copy_once(score,target)
    paths=[target]+[models/n for n in ['encoder.pt','head.pt','earhead_front.pt','earhead_head.pt']]
    p=dict(run=copy.deepcopy(r),scores=str(target),models=str(models),env=env,origin=str(origin),
           hashes={str(x):sha(x) for x in paths},verified_at=now())
    p['run']['scores_file']=str(target)
    validate_record(p,(r['fold'],r['seed']))
    write(record_path(variant,r['fold'],r['seed']),p)
    return p
def records(variant):
    out=[]
    for f,s in PAIRS:
        p=record_path(variant,f,s)
        if p.exists():out.append(validate_record(read(p),(f,s)))
    return out
def prepare():
    import torch
    frozen=OUT/'provenance.json'
    src=[ROOT/x for x in ['src/v2/train_encoder.py','src/v2/model/encoder.py','src/v2/dataset/crop.py',
         'src/v2/common/repro.py','src/v2/common/thresholds.py','src/v2/common/stats.py','src/v2/common/splits.py',
         'src/v2/common/folds_5fold.json','data/processed/v2/index.npz','work/nonpi_2026-09-18/train_gap.py',
         'work/nonpi_2026-09-21_maskfix/run_one.py','work/nonpi_2026-09-21_maskfix/maskfix.py',
         'work/nonpi_2026-09-21_maskfix/experiment_queue.py']]
    manuscripts=[Path.home()/'Downloads/MCE_blink_detection.tex',OLD/'manuscript/MCE_blink_detection.tex',
                 OLD/'manuscript/MCE_blink_detection.pdf',ROOT/'output/pdf/MCE_blink_detection_nonpi_review.pdf']
    provenance=dict(created=now(),python=sys.version,numpy=np.__version__,torch=torch.__version__,
                    source_hashes={str(p):sha(p) for p in src},manuscript_hashes={str(p):sha(p) for p in manuscripts if p.exists()},
                    frames_file=dict(path='data/processed/v2/frames_m22.npy',bytes=(ROOT/'data/processed/v2/frames_m22.npy').stat().st_size))
    if frozen.exists():
        before=read(frozen)
        if before['source_hashes']!=provenance['source_hashes']:raise RuntimeError('Source/data changed since queue preparation')
    else:write(frozen,provenance)
    return {v:len(records(v)) for v in VARIANTS}

def merge(variant):
    rr=records(variant)
    obj=dict(status='complete' if len(rr)==15 else 'partial',variant=variant,n_done=len(rr),
             runs=[p['run'] for p in rr],run_provenance=[{k:p[k] for k in ['env','origin','models','hashes']} for p in rr],
             minutes=sum(p['run']['seconds'] for p in rr)/60,
             config=dict(front='ours' if variant in ['vpres','vdrop'] else 'image_cnn_head',
                         arch=variant if variant in ['vpres','vdrop'] else 'vpres',control_architecture=variant,
                         latent=16,folds=list(range(5)),seeds=[0,1,2],missing_input='zero_before_encoder_valid_gather_only'),
             note='Fresh corrected fits only; missing frame tensors zero before encoder. Exploratory model comparison; no Pi measurement.')
    name=f'train_{variant}.json' if len(rr)==15 else f'train_{variant}.partial.json'
    write(OUT/'results'/name,obj)
    return len(rr)
def compare_all():
    from src.v2.model import encoder as E
    import torch
    paths={v:OUT/f'results/train_{v}.json' for v in VARIANTS}
    variants={k:load_variant(str(p)) for k,p in paths.items()}
    if any(set(v['runs'])!=set(PAIRS) for v in variants.values()):raise ValueError('Full 15-run comparison required')
    comparisons={}
    for ka,kb in [('vpres','image_head'),('vpres','gap'),('vdrop','vpres'),('vdrop','gap'),('vdrop','image_head')]:
        a,b=variants[ka],variants[kb];keys,sa,sb,y,sub=align(a,b)
        ci=stats.subject_bootstrap(lambda ix:TH.average_precision(sa[ix],y[ix])-TH.average_precision(sb[ix],y[ix]),sub,2000,seed=0)
        macro=[]
        for person in np.unique(sub):
            delta=[]
            for k in keys:
                x,z=a['runs'][k],b['runs'][k];ix=x['subject']==person
                if ix.any():delta.append(TH.average_precision(x['ours'][ix],x['y'][ix])-TH.average_precision(z['ours'][ix],z['y'][ix]))
            macro.append(np.mean(delta))
        macro=np.array(macro)
        comparisons[ka+'_minus_'+kb]=dict(pooled=ci,subject_macro=stats.subject_bootstrap(lambda ix:float(macro[ix].mean()),np.arange(len(macro)),2000,seed=0),
             noninferiority=[dict(delta=d,verdict=stats.non_inferiority(ci['ci_lo'],ci['ci_hi'],d)) for d in [.02,.01,.005]])
    metrics={k:{metric:dict(mean=float(np.mean([r['ours'][metric] for r in v['meta']['runs']])),sd=float(np.std([r['ours'][metric] for r in v['meta']['runs']],ddof=1))) for metric in ['pr_auc','recall','precision','f1','accuracy']} for k,v in variants.items()}
    costs={}
    for k in ['vpres','vdrop']:
        costs[k]=dict(params=sum(p.numel() for p in E.build(k).parameters())+sum(p.numel() for p in E.build_head().parameters()),mmac=E.analyse(k)['total_mmac']+E.temporal_head_mmac())
    costs.update(gap=dict(params=37889,mmac=31.414208),image_head=dict(params=476161,mmac=31.85248))
    warnings=[dict(variant=k,fold=r['fold'],seed=r['seed'],stopped_at=r['stopped_at']) for k,v in variants.items() for r in v['meta']['runs'] if not r['converged']]
    write(OUT/'results/comparison.json',dict(comparisons=comparisons,metrics=metrics,costs=costs,convergence_warnings=warnings,
        limitations=['Exploratory architecture selection, not untouched confirmatory evaluation.',
                     'Bootstrap conditions on fixed trained models; correlated fits are not independent people.',
                     'All four candidates are fresh corrected runs in the same environment; old results are historical.',
                     'No new Pi latency or power measurements; external validation remains necessary.']))
    md=['# Complete source-domain experiment results','', 'All 15 fits per candidate verified. Manuscript editing remains on hold.','',
        '| Model | AP mean | AP SD | Params | MMAC |','|---|---:|---:|---:|---:|']
    for k in variants:md.append(f"| {k} | {metrics[k]['pr_auc']['mean']:.6f} | {metrics[k]['pr_auc']['sd']:.6f} | {costs[k]['params']} | {costs[k]['mmac']:.6f} |")
    md+=['','Paired subject-bootstrap AP differences (95% percentile intervals):','']
    for k,v in comparisons.items():
        c=v['pooled'];md.append(f"- {k}: {c['point']:+.6f} [{c['ci_lo']:+.6f}, {c['ci_hi']:+.6f}]")
    md+=['','These are exploratory results. Do not change the paper or designate a winner without checking uncertainty, convergence and external validation.']
    (OUT/'RESULTS.md').write_text('\n'.join(md),encoding='utf8')
def status(state,**kwargs):
    write(OUT/'status.json',dict(state=state,updated=now(),controller_pid=os.getpid(),**kwargs))
def process_alive(pid):
    import ctypes
    from ctypes import wintypes
    kernel=ctypes.WinDLL('kernel32',use_last_error=True)
    kernel.OpenProcess.argtypes=[wintypes.DWORD,wintypes.BOOL,wintypes.DWORD]
    kernel.OpenProcess.restype=wintypes.HANDLE
    kernel.GetExitCodeProcess.argtypes=[wintypes.HANDLE,ctypes.POINTER(wintypes.DWORD)]
    kernel.CloseHandle.argtypes=[wintypes.HANDLE]
    handle=kernel.OpenProcess(0x1000,False,int(pid))
    if not handle:return ctypes.get_last_error()==5  # Access denied: conservatively do not duplicate.
    code=wintypes.DWORD()
    try:return bool(kernel.GetExitCodeProcess(handle,ctypes.byref(code))) and code.value==259
    finally:kernel.CloseHandle(handle)
def run():
    import msvcrt
    lock=(OUT/'controller.lock').open('a+b');lock.seek(0)
    if lock.read(1)==b'':lock.write(b'0');lock.flush()
    lock.seek(0);msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
    previous=OUT/'status.json'
    if previous.exists():
        prev=read(previous)
        if prev.get('state')=='running' and process_alive(prev.get('child_pid',0)):
            raise RuntimeError('Previous child is still alive; refusing duplicate training. Inspect status.json.')
    counts=prepare()
    try:
        # Alternate variants: both experiments make progress during a long queue.
        jobs=[(v,f,s) for f,s in PAIRS for v in VARIANTS]
        for variant,f,s in jobs:
            if record_path(variant,f,s).exists():continue
            prepare()  # Refuse changed code/data before every new process.
            folder=OUT/f'attempts/{variant}/fold{f}_seed{s}';folder.mkdir(parents=True,exist_ok=True)
            finished=sorted(folder.glob('attempt*/result.json'))
            if finished:
                # Recover a fit that finished before a controller interruption.
                result=finished[-1];d=read(result)
                if len(d['runs'])!=1:raise ValueError('Invalid recovered fit')
                make_record(variant,d['runs'][0],result.parent/'result_scores'/f'fold{f}_seed{s}.npz',
                            result.parent/'models'/f'fold{f}_seed{s}',d['env'],result)
                counts[variant]=merge(variant)
                continue
            prior=list(folder.glob('attempt*'));attempt=folder/f'attempt{len(prior)+1:03d}';attempt.mkdir()
            result=attempt/'result.json';log=attempt/'train.log';modelbase=attempt/'models'
            cmd=[sys.executable,'-u',str(OUT/'run_one.py'),variant,'--folds',str(f),'--seeds',str(s),'--save-models','--models-dir',str(modelbase),'--out',str(result)]
            cmd+=['--front','image_cnn_head'] if variant in ['gap','image_head'] else ['--arch',variant]
            env=dict(os.environ,PYTHONHASHSEED='0',PYTHONUTF8='1',MPLCONFIGDIR=str(OUT/'mplcache'))
            with log.open('w',encoding='utf8') as stream:
                proc=subprocess.Popen(cmd,cwd=ROOT,env=env,stdout=stream,stderr=subprocess.STDOUT)
                start=time.time()
                while proc.poll() is None:
                    status('running',variant=variant,fold=f,seed=s,child_pid=proc.pid,log=str(log),completed=counts,elapsed_fit_seconds=round(time.time()-start))
                    time.sleep(15)
            if proc.returncode!=0:raise RuntimeError(f'{variant} fold{f} seed{s} failed ({proc.returncode}); {log}')
            d=read(result)
            if len(d['runs'])!=1:raise ValueError('Single-fit artifact expected')
            score=result.with_name('result_scores')/f'fold{f}_seed{s}.npz'
            make_record(variant,d['runs'][0],score,modelbase/f'fold{f}_seed{s}',d['env'],result)
            counts[variant]=merge(variant)
        for v in VARIANTS:merge(v)
        status('analysing',completed=counts)
        compare_all()
        before=read(OUT/'provenance.json')['manuscript_hashes']
        unchanged={p:sha(p)==h for p,h in before.items()}
        status('complete',completed=counts,manuscripts_unchanged=unchanged,results=str(OUT/'RESULTS.md'))
    except BaseException as e:
        status('failed',completed=counts,error=str(e),traceback=traceback.format_exc())
        raise
    finally:lock.close()
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('command',choices=['prepare','run','analyse']);a=ap.parse_args()
    if a.command=='prepare':print(json.dumps(prepare()))
    elif a.command=='analyse':compare_all()
    else:run()
