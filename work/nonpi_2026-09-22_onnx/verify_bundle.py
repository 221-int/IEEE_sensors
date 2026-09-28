"""Portable CPU check for a transferred ONNX bundle; requires numpy/onnxruntime.

No torch, camera, source dataset, or original Windows paths are required.
Run: python verify_bundle.py --root . --out device_verification.json
"""
from pathlib import Path
import sys,json,hashlib,argparse,platform,datetime
import numpy as np
import onnxruntime as ort
HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(HERE/'runtime'))
import inference_policy as IP

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def verify(root):
    manifest=json.loads((root/'manifest.json').read_text(encoding='utf8'))
    contracts=manifest.get('contracts',[])
    if not contracts or len(contracts)!=manifest.get('models') or len(set(contracts))!=len(contracts):
        raise ValueError('Bundle must contain the declared nonempty set of model contracts')
    if any('\\' in name for name in [*contracts,*manifest['sha256']]):
        raise ValueError('Portable manifest paths must use forward slashes')
    for name,expected in manifest['sha256'].items():
        p=(root/name).resolve()
        if not p.is_relative_to(root.resolve()):raise ValueError('Manifest path leaves bundle')
        if sha(p)!=expected:raise ValueError('Bundle hash mismatch: '+name)
    reports=[]
    for name in contracts:
        if name not in manifest['sha256']:
            raise ValueError('Unhashed model contract: '+name)
        path=root/name;raw=json.loads(path.read_text(encoding='utf8'));folder=path.parent
        frontpath=folder/raw['graph_files']['encoder'];headpath=folder/raw['graph_files']['head']
        for graph in [frontpath,headpath]:
            if not graph.resolve().is_relative_to(folder.resolve()):
                raise ValueError('Graph path leaves model folder')
        c=IP.load_contract(folder,[frontpath,headpath])
        fixture=np.load(folder/'runtime_fixture.npz',allow_pickle=False)
        opts=ort.SessionOptions();opts.intra_op_num_threads=2;opts.inter_op_num_threads=1
        opts.add_session_config_entry('session.intra_op.allow_spinning','0')
        opts.add_session_config_entry('session.inter_op.allow_spinning','0')
        enc=ort.InferenceSession(str(frontpath),sess_options=opts,providers=['CPUExecutionProvider'])
        head=ort.InferenceSession(str(headpath),sess_options=opts,providers=['CPUExecutionProvider'])
        z=enc.run(None,{'crop':fixture['crop']})[0]
        p=head.run(None,{'vectors':fixture['vectors'],'mask':fixture['mask']})[0]
        enc_error=float(np.max(np.abs(z-fixture['expected_vectors'])))
        head_error=float(np.max(np.abs(p-fixture['expected_probability'])))
        maskmatch=np.array_equal(IP.eligible_mask(fixture['mask']),fixture['eligible'])
        assert enc_error<1e-4 and head_error<1e-4 and maskmatch,(name,enc_error,head_error)
        assert np.isfinite(z).all() and np.isfinite(p).all()
        reports.append(dict(contract=name,variant=c['variant'],fold=c['fold'],seed=c['seed'],
            encoder_max_abs_error=enc_error,head_max_abs_error=head_error,eligible_mask_matches=bool(maskmatch)))
    return dict(passed=True,created=datetime.datetime.now().astimezone().isoformat(),
        platform=platform.platform(),machine=platform.machine(),python=platform.python_version(),
        numpy=np.__version__,onnxruntime=ort.__version__,provider='CPUExecutionProvider',
        models=len(reports),graphs=2*len(reports),reports=reports,
        scope='Graph/contract hashes and numerical fixtures only; not a video accuracy or latency/power benchmark.')

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,default=HERE)
    ap.add_argument('--out',type=Path,default=None);args=ap.parse_args()
    r=verify(args.root.resolve());text=json.dumps(r,indent=2)
    if args.out:args.out.write_text(text,encoding='utf8')
    print(text)

if __name__=='__main__':main()
