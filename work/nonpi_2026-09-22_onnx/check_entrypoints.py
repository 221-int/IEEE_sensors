"""Exercise the existing Ours/Image-CNN entrypoint with real new ONNX sessions.

Only camera/frontend are simulated from a held-out event; no timing claims.
"""
from pathlib import Path
import sys,json,tempfile,types,contextlib,io
from unittest.mock import patch,MagicMock
import numpy as np
ROOT=Path(__file__).resolve().parents[2];HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(ROOT))
from src.v2.deploy import run_video as R
idx=np.load(ROOT/'data/processed/v2/index.npz');frames=np.load(ROOT/'data/processed/v2/frames_m22.npy',mmap_mode='r')
root=HERE/'packages/pi_default/onnx/fold0_seed0';result=[]
for mode,sub in [('ours',''),('image_cnn_head','image_cnn_head')]:
    folder=root/sub;contract=json.loads((folder/'contract.json').read_text())
    original=HERE/'onnx/fold0_seed0'/sub;fixture=np.load(original/'verification_scores.npz')
    candidate=np.flatnonzero(np.all(fixture['mask']==1,axis=1))[0]
    event=int(fixture['event_ids'][candidate]);rows=idx['e_rows'][event];crops=np.asarray(frames[rows])
    cap=MagicMock();cap.isOpened.return_value=True
    cap.read.side_effect=[(True,np.full((32,48,3),j,np.uint8)) for j in range(19)]+[(False,None)]
    frontend=MagicMock();frontend.mesh.return_value=np.zeros((468,2),np.float32)
    frontend.crop_from_mesh.side_effect=lambda frame,mesh:(crops[int(frame[0,0,0])],None)
    sampler=MagicMock(temp=[],cpu=[],rss=[],flags=set())
    with tempfile.TemporaryDirectory() as tmp,contextlib.ExitStack() as ctx:
        out=Path(tmp)/'out.json'
        args=['run_video','--mode',mode,'--source','fixture','--onnx-dir',str(root),'--intra-threads','2','--no-spin','--duration','100','--out',str(out)]
        ctx.enter_context(patch.object(sys,'argv',args))
        ctx.enter_context(patch('cv2.VideoCapture',return_value=cap))
        ctx.enter_context(patch.dict(sys.modules,{'src.v2.deploy.frontend':types.SimpleNamespace(EyeFrontend=lambda **kw:frontend)}))
        ctx.enter_context(patch.object(R,'Sampler',return_value=sampler));ctx.enter_context(patch.object(R,'_sh',return_value=None))
        ctx.enter_context(patch.object(R.repro,'ensure_hashseed'));ctx.enter_context(patch.object(R.repro,'seal'))
        ctx.enter_context(patch.object(R.repro,'env_fingerprint',return_value={'fixture':True}))
        with contextlib.redirect_stdout(io.StringIO()):assert R.main()==0
        report=json.loads(out.read_text(encoding='utf8'))
    expected=int(fixture['torch_probability'][candidate]>=contract['threshold_probability'])
    assert report['n_decisions']==1 and report['n_positive_decisions']==expected
    assert report['weights']=='trained (graph hash verified)'
    assert report['inference_policy']['threshold_probability']==contract['threshold_probability']
    result.append(dict(mode=mode,variant=contract['variant'],event=event,trained_graph_contract_loaded=True,
        n_decisions=1,expected_positive=expected,actual_positive=report['n_positive_decisions'],passed=True))
(HERE/'entrypoint_verification.json').write_text(json.dumps(dict(scope='Real ONNX sessions and frozen held-out crop; mocked camera/frontend, no timing measurements.',runs=result),indent=2),encoding='utf8')
print(json.dumps(result,indent=2))
