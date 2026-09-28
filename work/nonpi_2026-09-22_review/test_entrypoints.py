"""Camera-free integration: run the actual benchmark/demo loops with failed crops."""
from pathlib import Path
import sys,json,tempfile,unittest,types,contextlib,io,copy
from unittest.mock import patch,MagicMock
import numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from src.v2.deploy import run_video,run_video_power,pi_demo,inference_policy as IP

class EntryTests(unittest.TestCase):
    def test_both_benchmark_loops_match_offline_eligibility(self):
        n=50;valid=np.ones(n,bool);valid[10:16]=False;valid[30]=False
        expected=sum(bool(valid[i-18:i+1].sum()>=14 and valid[i]) for i in range(18,n))
        for mod in [run_video,run_video_power]:
            with self.subTest(module=mod.__name__),tempfile.TemporaryDirectory() as tmp,contextlib.ExitStack() as ctx:
                cap=MagicMock();cap.isOpened.return_value=True
                cap.read.side_effect=[(True,np.full((64,80,3),i,np.uint8)) for i in range(n)]+[(False,None)]
                fe=MagicMock();fe.mesh.side_effect=lambda frame:None if 10<=int(frame[0,0,0])<16 else np.zeros((468,2))
                fe.crop_from_mesh.side_effect=lambda frame,mesh:(None,None) if int(frame[0,0,0])==30 else (np.zeros((64,160),np.uint8),None)
                enc=MagicMock();enc.run.return_value=[np.ones((1,16),np.float32)]
                head=MagicMock();head.run.return_value=[np.array([.9],np.float32)]
                frontend=types.SimpleNamespace(EyeFrontend=lambda **kwargs:fe)
                sampler=MagicMock(temp=[],cpu=[],rss=[],flags=set())
                power=MagicMock(samples=[],parse_errors=0,missing_rails=set(),first_raw_sample=None)
                out=Path(tmp)/'out.json'
                argv=['bench','--mode','ours','--source','fixture','--duration','100','--out',str(out),
                      '--onnx-dir',str(ROOT/'work/nonpi_2026-09-21/trained_deployment/ours'),
                      '--intra-threads','2','--no-spin']
                if mod is run_video_power:argv+=['--idle-seconds','0']
                ctx.enter_context(patch.object(sys,'argv',argv));ctx.enter_context(patch.dict(sys.modules,{'src.v2.deploy.frontend':frontend}))
                ctx.enter_context(patch('cv2.VideoCapture',return_value=cap))
                ctx.enter_context(patch.object(mod,'make_session',side_effect=[enc,head]))
                ctx.enter_context(patch.object(mod,'Sampler',return_value=sampler))
                ctx.enter_context(patch.object(mod,'_sh',return_value=None))
                ctx.enter_context(patch.object(mod.repro,'ensure_hashseed'));ctx.enter_context(patch.object(mod.repro,'seal'))
                ctx.enter_context(patch.object(mod.repro,'env_fingerprint',return_value={'test_fixture':True}))
                if mod is run_video_power:ctx.enter_context(patch.object(mod,'PowerSampler',return_value=power))
                with contextlib.redirect_stdout(io.StringIO()):self.assertEqual(mod.main(),0)
                r=json.loads(out.read_text(encoding='utf8'))
                self.assertEqual(r['n_decisions'],expected);self.assertEqual(head.run.call_count,expected)
                self.assertEqual(r['n_positive_decisions'],expected)
                for call in head.run.call_args_list:
                    mask=call.args[1]['mask'];self.assertTrue(IP.eligible_mask(mask)[0])
                    self.assertTrue(np.all(call.args[1]['vectors'][mask==0]==0))

    def test_demo_crop_failure_never_calls_ear_and_sequence_gaps_abstain(self):
        sequence=list(range(23))+list(range(29,46));states=[]
        reader=MagicMock(connected=True,err='');reader.latest.side_effect=[(np.full((64,80,3),i,np.uint8),i) for i in sequence]
        fe=MagicMock();fe.mesh.return_value=np.zeros((468,2))
        meta=types.SimpleNamespace(center_x=40.,center_y=32.,crop_w_px=60.,crop_h_px=24.,tilt_deg=0.)
        fe.crop_from_mesh.side_effect=lambda frame,mesh:(None,None) if int(frame[0,0,0])==22 else (np.zeros((64,160),np.uint8),meta)
        enc=MagicMock();enc.run.return_value=[np.ones((1,16),np.float32)]
        head=MagicMock();head.run.return_value=[np.array([.9],np.float32)]
        ort=types.SimpleNamespace(SessionOptions=MagicMock,InferenceSession=MagicMock(side_effect=[enc,head]))
        frontend=types.SimpleNamespace(EyeFrontend=lambda **kwargs:fe)
        def render(canvas,ui,state):
            states.append(copy.deepcopy(state))
            if len(states)==len(sequence):raise KeyboardInterrupt
            return canvas
        with tempfile.TemporaryDirectory() as tmp,contextlib.ExitStack() as ctx:
            argv=['demo','--headless','--no-mirror','--blink-th','.5','--profiles',str(Path(tmp)/'profiles.json')]
            ctx.enter_context(patch.object(sys,'argv',argv));ctx.enter_context(patch.dict(sys.modules,{'src.v2.deploy.frontend':frontend,'onnxruntime':ort}))
            ctx.enter_context(patch.object(pi_demo,'StreamReader',return_value=reader))
            ctx.enter_context(patch.object(pi_demo,'render',side_effect=render))
            ctx.enter_context(patch.object(pi_demo,'_sh',return_value=None))
            ear=ctx.enter_context(patch.object(pi_demo.C,'ear_both',side_effect=AssertionError('CNN must not fall back to EAR')))
            with contextlib.redirect_stdout(io.StringIO()):self.assertEqual(pi_demo.main(),0)
        self.assertFalse(states[17]['decision_valid']);self.assertTrue(states[18]['decision_valid'])
        self.assertEqual(states[18]['session_blinks'],1)
        self.assertFalse(states[22]['decision_valid']);self.assertEqual(states[22]['prob'],0)
        self.assertFalse(states[23]['decision_valid']);self.assertTrue(states[-1]['decision_valid'])
        self.assertEqual(states[-1]['session_blinks'],2);self.assertEqual(ear.call_count,0)

if __name__=='__main__':unittest.main(verbosity=2)
