"""Diagnose stored-GPU/CPU boundary cases without retraining or retuning."""
from pathlib import Path
import sys,os,json
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
ROOT=Path(__file__).resolve().parents[2];HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'work/nonpi_2026-09-21_maskfix'))
import numpy as np,torch
import post_train as P
from src.v2.common import repro
from src.v2.dataset import crop as C
torch.set_num_threads(2)
idx=np.load(ROOT/'data/processed/v2/index.npz');frames=np.load(ROOT/'data/processed/v2/frames_m22.npy',mmap_mode='r')
cases=json.loads((HERE/'gpu_cpu_boundary_cases.json').read_text());results=[]
for case in cases:
    f,s=case['fold'],case['seed'];repro.seal(s)
    ids=np.flatnonzero(idx['e_valid']&(idx['e_fold']==f));position=int(np.searchsorted(ids,case['event']))
    start=(position//32)*32;batch=ids[start:start+32];rows=idx['e_rows'][batch];mask=(rows>=0).astype(np.float32)
    x=np.zeros((*rows.shape,1,64,160),np.float32);x[rows>=0]=C.batch_input(np.asarray(frames[rows[rows>=0]]))
    front,head,thr=P.model(case['variant'],f,s,'cuda');outputs={}
    for use_tf32 in [True,False]:
        torch.backends.cudnn.allow_tf32=use_tf32
        with torch.inference_mode():
            z=front(torch.from_numpy(x.reshape(-1,1,64,160)).cuda()).reshape(len(batch),19,16)
            logits=head(z,torch.from_numpy(mask).cuda()).cpu().numpy()
        outputs[str(use_tf32)]=float(logits[position-start])
    results.append(dict(**case,replayed_original_batch_gpu_logits=outputs,
        replay_tf32_on_minus_saved=outputs['True']-case['gpu'],
        replay_tf32_off_minus_cpu=outputs['False']-case['cpu']))
    print(json.dumps(results[-1]),flush=True)
(HERE/'gpu_cpu_boundary_diagnosis.json').write_text(json.dumps(dict(cases=results,
    scope='Four selected boundary cases only. TF32 on/off inference diagnostic, no training or threshold change.',
    caution='Historical training metadata did not explicitly store TF32 flags; matched replay is evidence, not a complete historical runtime proof.'),indent=2),encoding='utf8')
