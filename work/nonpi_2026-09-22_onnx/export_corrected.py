"""Export all 60 corrected FP32 fits and verify contracts without changing sources.

CPU PyTorch versus CPU ONNX Runtime; engineering equivalence, not a Pi benchmark.
"""
from pathlib import Path
import sys,json,hashlib,datetime,platform,traceback
import numpy as np
import torch,onnx,onnxruntime as ort
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
HERE=Path(__file__).resolve().parent;STUDY=ROOT/'work/nonpi_2026-09-21_maskfix'
sys.path.insert(0,str(STUDY))
import experiment_queue as Q
import post_train as P
from src.v2.dataset import crop as C
from src.v2.deploy import inference_policy as IP
torch.set_num_threads(2)
VARIANTS=['vpres','image_head','vdrop','gap']

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf8'))
def dump(p,obj):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix(p.suffix+'.tmp')
    t.write_text(json.dumps(obj,indent=2,ensure_ascii=False,allow_nan=False),encoding='utf8');t.replace(p)
def now():return datetime.datetime.now().astimezone().isoformat()
def status(state,**kwargs):dump(HERE/'status.json',dict(state=state,updated=now(),**kwargs))

class ProbabilityHead(torch.nn.Module):
    def __init__(self,head):super().__init__();self.head=head
    def forward(self,z,mask):return torch.sigmoid(self.head(z,mask))

def output_dir(v,f,s):
    root=HERE/f'onnx/fold{f}_seed{s}'
    return root if v=='vpres' else root/('image_cnn_head' if v=='image_head' else v)

def select_events(record,idx):
    """Balanced fixed sample + threshold-near and missing-input engineering cases."""
    r=record['run'];f,s=r['fold'],r['seed'];ids=np.flatnonzero(idx['e_valid']&(idx['e_fold']==f))
    scores=np.load(record['scores']);assert np.array_equal(scores['y'],idx['e_is_blink'][ids])
    rng=np.random.default_rng(20260922+10*f+s)
    balanced=np.concatenate([rng.choice(ids[idx['e_is_blink'][ids]==label],16,replace=False) for label in [0,1]])
    near=ids[np.argsort(np.abs(scores['ours']-r['ours']['thr']),kind='stable')[:16]]
    missing=ids[np.any(idx['e_rows'][ids]<0,axis=1)][:8]
    chosen=np.unique(np.r_[balanced,near,missing])
    return chosen,dict(balanced=balanced.tolist(),threshold_near=near.tolist(),missing=missing.tolist())

def sessions(folder,front_name):
    opts=ort.SessionOptions();opts.intra_op_num_threads=2;opts.inter_op_num_threads=1
    opts.add_session_config_entry('session.intra_op.allow_spinning','0')
    opts.add_session_config_entry('session.inter_op.allow_spinning','0')
    return [ort.InferenceSession(str(folder/n),sess_options=opts,providers=['CPUExecutionProvider']) for n in [front_name,'head.onnx']]

def one(v,f,s,idx,frames):
    rec=Q.validate_record(read(Q.record_path(v,f,s)),(f,s));ckpt=Path(rec['models'])
    folder=output_dir(v,f,s);folder.mkdir(parents=True,exist_ok=True)
    front_name='backbone.onnx' if v=='image_head' else 'encoder.onnx'
    complete=folder/'contract.json'
    if complete.exists():
        c=IP.load_contract(folder,[folder/front_name,folder/'head.onnx'])
        assert c['checkpoint_sha256']=={n:sha(ckpt/n) for n in ['encoder.pt','head.pt']}
        assert c['source_record_sha256']==sha(Q.record_path(v,f,s))
        return c
    front,head,thr=P.model(v,f,s,'cpu');wrapped=ProbabilityHead(head).eval()
    ids,selection=select_events(rec,idx);rows=idx['e_rows'][ids];mask=(rows>=0).astype(np.float32)
    x=np.zeros((*rows.shape,1,64,160),np.float32);x[rows>=0]=C.batch_input(np.asarray(frames[rows[rows>=0]]))
    flat=x.reshape(-1,1,64,160)
    with torch.inference_mode():
        torch.onnx.export(front,torch.from_numpy(flat[:2]),str(folder/front_name),opset_version=17,dynamo=False,
            input_names=['crop'],output_names=['vector'],dynamic_axes={'crop':{0:'batch'},'vector':{0:'batch'}})
        torch.onnx.export(wrapped,(torch.zeros(2,19,16),torch.ones(2,19)),str(folder/'head.onnx'),opset_version=17,dynamo=False,
            input_names=['vectors','mask'],output_names=['blink_prob'],
            dynamic_axes={'vectors':{0:'batch'},'mask':{0:'batch'},'blink_prob':{0:'batch'}})
    graph_details={}
    for n in [front_name,'head.onnx']:
        graph=onnx.load(str(folder/n));onnx.checker.check_model(graph,full_check=True)
        assert graph.ir_version<=10
        assert all(op.domain in ('','ai.onnx') for op in graph.opset_import)
        assert not any(t.data_type in [onnx.TensorProto.INT8,onnx.TensorProto.UINT8,onnx.TensorProto.FLOAT16] for t in graph.graph.initializer)
        graph_details[n]=dict(ir_version=graph.ir_version,opsets={op.domain:op.version for op in graph.opset_import},
                            inputs=[x.name for x in graph.graph.input],outputs=[x.name for x in graph.graph.output])
    enc,hs=sessions(folder,front_name)
    assert [a.name for a in enc.get_inputs()]==['crop']
    assert [a.name for a in hs.get_inputs()]==['vectors','mask']
    with torch.inference_mode():
        zt=np.concatenate([front(torch.from_numpy(flat[j:j+64])).numpy() for j in range(0,len(flat),64)]).reshape(len(ids),19,16)
        logits=head(torch.from_numpy(zt),torch.from_numpy(mask)).numpy()
        pt=torch.sigmoid(torch.from_numpy(logits)).numpy()
    zo=np.concatenate([enc.run(None,{'crop':flat[j:j+64]})[0] for j in range(0,len(flat),64)]).reshape(len(ids),19,16)
    po=hs.run(None,{'vectors':zo,'mask':mask})[0]
    pthr=float(1/(1+np.exp(-thr)))
    enc_error=float(np.max(np.abs(zo-zt)));prob_error=float(np.max(np.abs(po-pt)))
    flips=int(np.sum((po>=pthr)!=(logits>=thr)))
    assert enc_error<1e-4 and prob_error<1e-4 and flips==0,(v,f,s,enc_error,prob_error,flips)
    # Deployment caches only valid encoded vectors, putting zeros at missing positions.
    cached=zo.copy();cached[mask==0]=0
    cached_prob=hs.run(None,{'vectors':cached,'mask':mask})[0]
    cached_error=float(np.max(np.abs(cached_prob-po)));assert cached_error<1e-6
    # Dynamic batch=1 and mixed boundary masks, including invalid-all-missing input.
    rng=np.random.default_rng(119);mz=np.ones((7,19),np.float32)
    for j,n in enumerate([0,1,5,6,18,19]):mz[j,:n]=0
    mz[-1,-1]=0
    z=rng.normal(size=(7,19,16)).astype(np.float32)
    with torch.inference_mode():pr=wrapped(torch.from_numpy(z),torch.from_numpy(mz)).numpy()
    oo=hs.run(None,{'vectors':z,'mask':mz})[0]
    stress_error=float(np.max(np.abs(pr-oo)));assert np.isfinite(oo).all() and stress_error<1e-4
    changed=z.copy();changed[mz==0]=123.45
    assert np.max(np.abs(hs.run(None,{'vectors':changed,'mask':mz})[0]-oo))<1e-6
    onez=enc.run(None,{'crop':flat[:1]})[0];assert onez.shape==(1,16)
    single=hs.run(None,{'vectors':cached[:1],'mask':mask[:1]})[0]
    assert single.shape==(1,) and np.max(np.abs(single-cached_prob[:1]))<1e-5
    test_ids=np.flatnonzero(idx['e_valid']&(idx['e_fold']==f));saved=np.load(rec['scores'])['ours'][np.searchsorted(test_ids,ids)]
    saved_flips=int(np.sum((saved>=thr)!=(logits>=thr)))
    report=dict(n_source_events=len(ids),n_crop_positions=len(flat),selection=selection,event_indices=ids.tolist(),
        encoder_max_abs_error=enc_error,probability_max_abs_error=prob_error,threshold_prediction_flips=flips,
        zero_missing_vector_max_probability_error=cached_error,mask_stress_max_probability_error=stress_error,
        mask_stress_valid_counts=mz.sum(1).astype(int).tolist(),mask_stress_online_eligible=IP.eligible_mask(mz).tolist(),
        dynamic_batch_one_pass=True,graph_checker_pass=True,
        saved_gpu_vs_cpu_max_logit_error=float(np.max(np.abs(saved-logits))),saved_gpu_vs_cpu_prediction_flips=saved_flips)
    np.savez_compressed(folder/'verification_scores.npz',event_ids=ids,labels=idx['e_is_blink'][ids],mask=mask,
        torch_logits=logits,torch_probability=pt,onnx_probability=po,saved_gpu_logits=saved)
    contract=dict(schema_version=1,variant=v,weights='trained',fold=f,seed=s,
        study='nonpi_2026-09-21_maskfix',missing_training_input='zero_before_encoder_valid_gather_only',
        selection='All 5 folds x 3 seeds exported; default engineering bundle fixed to fold0/seed0, not best score.',
        checkpoint=str(ckpt),source_record=str(Q.record_path(v,f,s)),source_record_sha256=sha(Q.record_path(v,f,s)),
        checkpoint_sha256={n:sha(ckpt/n) for n in ['encoder.pt','head.pt']},
        graph_sha256={n:sha(folder/n) for n in [front_name,'head.onnx']},
        graph_files=dict(encoder=front_name,head='head.onnx'),graph_details=graph_details,
        input=dict(shape=['batch',1,64,160],dtype='float32',norm=C.INPUT_NORM,margin=C.MARGIN,
                   geometry='explicit_float32_crop_bounds',color='grayscale from BGR crop'),
        encoder_output=dict(shape=['batch',16],dtype='float32'),
        head_inputs=dict(vectors=['batch',19,16],mask=['batch',19]),
        head_output='probability (sigmoid in graph)',threshold_logit=thr,threshold_probability=pthr,
        threshold_selection='Original source validation accuracy; unchanged, no refitting on test or external data.',
        min_valid_frames=14,event_len=19,policy=IP.metadata(),requires_current_valid=True,
        nominal_source_fps=30,window_center_offset_frames=9,
        precision='FP32; not quantized',export_runtime=dict(python=platform.python_version(),torch=torch.__version__,
            onnx=onnx.__version__,onnxruntime=ort.__version__,numpy=np.__version__,provider='CPUExecutionProvider'),
        source_sha256={n:sha(ROOT/n) for n in ['src/v2/model/encoder.py','src/v2/dataset/crop.py','src/v2/deploy/inference_policy.py']},
        verification=report,limits=['CPU export equivalence only; no Raspberry Pi latency/power measurement.',
            'The Pi pinned ONNX Runtime version must be checked on device; current desktop runtime differs.',
            'No all-data retraining or final clinical/consumer operating-point validation.',
            'Graph outputs on ineligible windows are diagnostic only; deployment must abstain.'])
    contract['policy']=IP.metadata(contract)
    dump(complete,contract)
    assert IP.load_contract(folder,[folder/front_name,folder/'head.onnx']) is not None
    return contract

def main():
    idx=np.load(ROOT/'data/processed/v2/index.npz');frames=np.load(ROOT/'data/processed/v2/frames_m22.npy',mmap_mode='r')
    manifest_path=HERE/'preserved_before.json'
    if not manifest_path.exists():
        before=read(ROOT/'work/nonpi_2026-09-22_review/verification.json')['manuscript_unchanged']
        paths=[Path(p) for p in before]
        for base in [ROOT/'models/v2/onnx',ROOT/'work/nonpi_2026-09-21/trained_deployment']:
            paths.extend(p for p in base.rglob('*') if p.is_file())
        dump(manifest_path,{str(p):sha(p) for p in paths})
    reports=[]
    try:
        for f in range(5):
            for s in range(3):
                for v in VARIANTS:
                    status('running',done=len(reports),total=60,variant=v,fold=f,seed=s)
                    c=one(v,f,s,idx,frames);reports.append(c)
                    print(f'{len(reports)}/60 {v} fold{f} seed{s} error={c["verification"]["probability_max_abs_error"]:.3g}',flush=True)
        preserved={p:sha(p)==h for p,h in read(manifest_path).items()};assert all(preserved.values())
        summary=dict(created=now(),n_models=len(reports),n_graphs=2*len(reports),
            max_encoder_error=max(c['verification']['encoder_max_abs_error'] for c in reports),
            max_probability_error=max(c['verification']['probability_max_abs_error'] for c in reports),
            total_threshold_prediction_flips=sum(c['verification']['threshold_prediction_flips'] for c in reports),
            total_saved_gpu_vs_cpu_prediction_flips=sum(c['verification']['saved_gpu_vs_cpu_prediction_flips'] for c in reports),
            n_event_model_checks=sum(c['verification']['n_source_events'] for c in reports),
            contracts=[str((output_dir(c['variant'],c['fold'],c['seed'])/'contract.json').relative_to(HERE)) for c in reports],
            original_files_unchanged=preserved,default_bundle='onnx/fold0_seed0',
            limits='Desktop CPU numerical verification only; Pi device validation and benchmark are separate.')
        dump(HERE/'verification.json',summary);status('complete',done=60,total=60,report=str(HERE/'verification.json'))
        print(json.dumps({k:v for k,v in summary.items() if k not in ['contracts','original_files_unchanged']},indent=2))
    except BaseException as e:
        status('failed',done=len(reports),error=repr(e),traceback=traceback.format_exc());raise

if __name__=='__main__':main()
