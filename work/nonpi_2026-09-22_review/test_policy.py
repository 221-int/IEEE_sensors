from pathlib import Path
import sys,unittest,tempfile,json,hashlib,math
from collections import deque
import numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from src.v2.deploy import inference_policy as P

class PolicyTests(unittest.TestCase):
    def test_missing_boundaries_and_current_frame(self):
        for count in range(20):
            mask=np.arange(19)>=19-count
            self.assertEqual(bool(P.eligible_mask(mask)),count>=14)
            mask[-1]=False
            self.assertFalse(P.eligible_mask(mask))
        self.assertFalse(P.ring_ready(deque([np.ones(16)]*18,maxlen=19)))

    def test_skipped_frames_and_abstention_event_breaks(self):
        ring=deque([np.ones(16)]*19,maxlen=19)
        P.advance_missing(ring,5);ring.append(np.ones(16));self.assertTrue(P.ring_ready(ring))
        P.advance_missing(ring,6);ring.append(np.ones(16));self.assertFalse(P.ring_ready(ring))
        P.advance_missing(ring,1000000);self.assertEqual(list(ring),[None]*19)
        state=False;starts=[]
        for i,p in enumerate([.6,.8,None,.7,.3,.5]):
            state,onset=P.candidate_onset(p,.5,state)
            if onset:starts.append(i)
        self.assertEqual(starts,[0,3,5])

    def test_threshold_contract_must_match_graphs(self):
        with tempfile.TemporaryDirectory() as d:
            base=Path(d); graph=base/'head.onnx';graph.write_bytes(b'test fixture')
            self.assertIsNone(P.load_contract(base,[graph]))
            with self.assertRaises(FileNotFoundError):P.load_contract(base,[graph],base/'missing.json')
            obj=dict(event_len=19,min_valid_frames=14,weights='trained',input=dict(norm='frame_standardize'),
                head_output='probability (sigmoid in graph)',threshold_logit=-1.,threshold_probability=1/(1+math.e),
                graph_sha256={graph.name:hashlib.sha256(graph.read_bytes()).hexdigest()})
            c=base/'contract.json';c.write_text(json.dumps(obj))
            self.assertAlmostEqual(P.load_contract(base,[graph])['threshold_probability'],1/(1+math.e))
            graph.write_bytes(b'wrong model')
            with self.assertRaises(ValueError):P.load_contract(base,[graph])

    def test_training_never_gathers_missing_row_and_matches_corrected_runner(self):
        from src.v2.train_encoder import Bundle
        sys.path.insert(0,str(ROOT/'work/nonpi_2026-09-21_maskfix'))
        from maskfix import MaskSafeBundle
        rng=np.random.default_rng(8);frames=rng.integers(0,256,(21,64,160),dtype=np.uint8)
        class Guard:
            def __getitem__(self,rows):
                if np.any(rows==0) or np.any(rows<0):raise AssertionError('Read missing placeholder')
                return frames[rows]
        batch=Bundle.__new__(Bundle);batch.frames=Guard();batch.rows=np.array([[-1]*5+list(range(1,15)),[-1]*19])
        batch.mask=(batch.rows>=0).astype(np.float32);batch.y=np.array([1,0],np.float32)
        x,m,y=batch.batch(np.array([0,1]));safe=MaskSafeBundle.__new__(MaskSafeBundle);safe.__dict__.update(batch.__dict__)
        expected=safe.batch(np.array([0,1]))
        for a,b in zip((x,m,y),expected):np.testing.assert_array_equal(a,b)
        np.testing.assert_array_equal(x[m==0],0)

    def test_frozen_external_scores_match_online_policy(self):
        folder=ROOT/'work/nonpi_2026-09-21_maskfix'
        d=np.load(ROOT/'work/nonpi_2026-09-18/results/external_features.npz');v=d['valid'];labels=d['label']
        end=np.arange(18,len(v));windows=end[:,None]+np.arange(-18,1);known=np.all(labels[windows]!=-2,1)
        reference=known&(v[windows].sum(1)>=14)&v[end];actual=known&P.eligible_mask(v[windows])
        np.testing.assert_array_equal(reference,actual)
        ring=deque(maxlen=19);ready=[]
        for valid in v:
            ring.append(np.zeros(16) if valid else None)
            if len(ring)==19:ready.append(P.ring_ready(ring))
        np.testing.assert_array_equal(np.array(ready)&known,reference)
        scores=np.load(folder/'post_train/external_video9_scores.npz')
        runs=json.loads((folder/'post_train/external_video9.json').read_text())['runs']
        for r in runs:
            if r['policy']!='min14_current_valid':continue
            ss=scores[f'{r["variant"]}_fold{r["fold"]}_seed{r["seed"]}'];thr=r['threshold_logit']
            offline=(ss>=thr)&actual;state=False;onsets=[]
            # Use logit scores and the same threshold to avoid float32 sigmoid rounding.
            for j,(score,ok) in enumerate(zip(ss,actual)):
                state,on=P.candidate_onset(float(score) if ok else None,thr,state)
                if on:onsets.append(j)
            expected=np.flatnonzero(offline&~np.r_[False,offline[:-1]])
            np.testing.assert_array_equal(onsets,expected)

if __name__=='__main__':unittest.main(verbosity=2)
