"""Prepare trained paired ONNX graphs for future Pi measurement; no Pi claims."""
from pathlib import Path
import sys,json,hashlib
import numpy as np,torch,onnxruntime as ort
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from src.v2.model import encoder as E
from src.v2.dataset import crop as C
OUT=Path(__file__).resolve().parent/'trained_deployment';OUT.mkdir(exist_ok=True)
torch.set_num_threads(2)

class ProbabilityHead(torch.nn.Module):
    def __init__(self,head):super().__init__();self.head=head
    def forward(self,z,mask):return torch.sigmoid(self.head(z,mask))

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    idx=np.load(ROOT/'data/processed/v2/index.npz');frames=np.load(ROOT/'data/processed/v2/frames_m22.npy',mmap_mode='r')
    # Deterministic diagnostic sample from the selected checkpoint's held-out fold.
    candidate=idx['e_valid'].astype(bool)&(idx['e_fold']==0)
    ids=np.concatenate([np.flatnonzero(candidate&(idx['e_is_blink']==label))[:8] for label in [0,1]])
    rows=idx['e_rows'][ids];mask=(rows>=0).astype(np.float32)
    x=C.batch_input(np.asarray(frames[np.maximum(rows,0).ravel()]));report=[]
    for variant,stem in [('ours','train_encoder_final'),('image_head','train_image_cnn_head_final')]:
        d=OUT/variant;d.mkdir(exist_ok=True)
        ckpt=ROOT/f'models/v2/{stem}/fold0_seed0'
        source=json.loads((ROOT/f'results/v2/{stem}.json').read_text(encoding='utf8'))
        run=next(r for r in source['runs'] if r['fold']==0 and r['seed']==0)
        front=E.build() if variant=='ours' else E.build_image_cnn(16);head=E.build_head()
        front.load_state_dict(torch.load(ckpt/'encoder.pt',weights_only=True,map_location='cpu'));front.eval()
        head.load_state_dict(torch.load(ckpt/'head.pt',weights_only=True,map_location='cpu'));head.eval()
        wrapped=ProbabilityHead(head).eval()
        with torch.no_grad():
            torch.onnx.export(front,torch.from_numpy(x[:2]),str(d/'encoder.onnx'),input_names=['crop'],output_names=['vector'],
                dynamic_axes={'crop':{0:'batch'},'vector':{0:'batch'}},opset_version=17,dynamo=False)
            torch.onnx.export(wrapped,(torch.zeros(2,19,16),torch.ones(2,19)),str(d/'head.onnx'),
                input_names=['vectors','mask'],output_names=['prob'],dynamic_axes={'vectors':{0:'batch'},'mask':{0:'batch'},'prob':{0:'batch'}},opset_version=17,dynamo=False)
            zz=front(torch.from_numpy(x)).numpy().reshape(16,19,16)
            pp=wrapped(torch.from_numpy(zz),torch.from_numpy(mask)).numpy()
        opts=ort.SessionOptions();opts.intra_op_num_threads=2
        sess=ort.InferenceSession(str(d/'encoder.onnx'),sess_options=opts,providers=['CPUExecutionProvider'])
        hs=ort.InferenceSession(str(d/'head.onnx'),sess_options=opts,providers=['CPUExecutionProvider'])
        zo=sess.run(None,{'crop':x})[0].reshape(16,19,16)
        po=hs.run(None,{'vectors':zo,'mask':mask})[0]
        enc_error=float(np.max(np.abs(zo-zz)));prob_error=float(np.max(np.abs(po-pp)))
        assert enc_error<1e-4 and prob_error<1e-4,(variant,enc_error,prob_error)
        thr=float(run['ours']['thr']);pth=float(1/(1+np.exp(-thr)))
        assert np.array_equal(po>=pth,pp>=pth)
        stress_mask=np.ones((5,19),np.float32)
        for j,missing in enumerate([0,1,5,6,19]):stress_mask[j,:missing]=0
        with torch.no_grad():stress_ref=wrapped(torch.from_numpy(zz[:5]),torch.from_numpy(stress_mask)).numpy()
        stress_out=hs.run(None,{'vectors':zz[:5],'mask':stress_mask})[0]
        stress_error=float(np.max(np.abs(stress_ref-stress_out)))
        assert np.isfinite(stress_out).all() and stress_error<1e-4
        record=dict(variant=variant,weights='trained',fold=0,seed=0,checkpoint=str(ckpt),
          selection='fixed fold0 seed0 for engineering verification; not selected for best accuracy',
          checkpoint_sha256={n:sha(ckpt/n) for n in ['encoder.pt','head.pt']},
          graph_sha256={n:sha(d/n) for n in ['encoder.onnx','head.onnx']},
          input=dict(shape=['batch',1,64,160],norm=C.INPUT_NORM,margin=C.MARGIN),
          threshold_logit=thr,threshold_probability=pth,min_valid_frames=14,event_len=19,
          head_output='probability (sigmoid in graph)',
          verification=dict(n_source_events=16,n_frames=len(x),event_indices=ids.tolist(),encoder_max_abs_error=enc_error,
                            probability_max_abs_error=prob_error,threshold_predictions_equal=True,
                            mask_stress_missing_counts=[0,1,5,6,19],mask_stress_max_probability_error=stress_error),
          deployment_policy='Require complete 19-position ring and at least 14 valid frames; missing frames retain their time positions. Current-frame skip/abstention and event postprocessing must be explicitly declared by the caller.',
          limits=['No Raspberry Pi timing or power was measured.',
                  'Existing Pi results for random Image-CNN weights do not become trained-weight measurements by this export.',
                  'This export is a preparation artifact, not a replacement for final architecture selection.'])
        (d/'contract.json').write_text(json.dumps(record,indent=2),encoding='utf8');report.append(record)
    (OUT/'verification.json').write_text(json.dumps(report,indent=2),encoding='utf8')
    print(json.dumps([{k:r[k] for k in ['variant','threshold_probability','verification']} for r in report],indent=2))

if __name__=='__main__':main()
