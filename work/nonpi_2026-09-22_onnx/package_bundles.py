"""Build reproducible transfer bundles and test the portable verifier locally."""
from pathlib import Path
import sys,json,hashlib,shutil,zipfile,subprocess,os
import numpy as np,torch
ROOT=Path(__file__).resolve().parents[2];HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'work/nonpi_2026-09-21_maskfix'))
import post_train as P
from src.v2.dataset import crop as C
from src.v2.deploy import inference_policy as IP
torch.set_num_threads(2)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def copy(src,dst):
    assert dst.resolve().is_relative_to((HERE/'packages').resolve())
    dst.parent.mkdir(parents=True,exist_ok=True)
    shutil.copy2(src,dst)

def main():
    verification=json.loads((HERE/'verification.json').read_text());assert verification['n_models']==60
    verification['contracts']=[Path(p).as_posix() for p in verification['contracts']]
    rng=np.random.default_rng(20260922)
    crops=C.batch_input(rng.integers(0,256,(2,64,160),dtype=np.uint8))
    vectors=rng.normal(size=(7,19,16)).astype(np.float32);mask=np.ones((7,19),np.float32)
    for j,missing in enumerate([0,1,5,6,18,19]):mask[j,:missing]=0
    mask[-1,-1]=0
    for contract in verification['contracts']:
        path=HERE/contract;c=json.loads(path.read_text());front,head,_=P.model(c['variant'],c['fold'],c['seed'],'cpu')
        with torch.inference_mode():
            z=front(torch.from_numpy(crops)).numpy()
            p=torch.sigmoid(head(torch.from_numpy(vectors),torch.from_numpy(mask))).numpy()
        np.savez_compressed(path.parent/'runtime_fixture.npz',crop=crops,vectors=vectors,mask=mask,
                            expected_vectors=z,expected_probability=p,eligible=IP.eligible_mask(mask))
    output=[]
    for name,contracts in [('pi_default', [p for p in verification['contracts'] if p.startswith('onnx/fold0_seed0/')]),
                           ('all_60',verification['contracts'])]:
        assert len(contracts)==(4 if name=='pi_default' else 60)
        folder=HERE/'packages'/name;folder.mkdir(parents=True,exist_ok=True)
        for item in contracts:
            path=HERE/item;c=json.loads(path.read_text())
            for n in [*c['graph_sha256'],'contract.json','runtime_fixture.npz']:
                copy(path.parent/n,folder/path.parent.relative_to(HERE)/n)
        copy(HERE/'verify_bundle.py',folder/'verify_bundle.py')
        copy(ROOT/'src/v2/deploy/inference_policy.py',folder/'runtime/inference_policy.py')
        copy(HERE/'README.md',folder/'README.md')
        manifest=dict(models=len(contracts),contracts=contracts,sha256={str(p.relative_to(folder)).replace('\\','/'):sha(p)
            for p in sorted(folder.rglob('*')) if p.is_file() and p.name not in ['manifest.json','local_verification.json'] and '__pycache__' not in str(p)})
        (folder/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf8')
        env=dict(os.environ,PYTHONIOENCODING='utf-8')
        r=subprocess.run([sys.executable,str(folder/'verify_bundle.py'),'--root',str(folder),'--out',str(folder/'local_verification.json')],
            cwd=folder,env=env,capture_output=True,text=True,encoding='utf8')
        (HERE/f'verify_{name}.log').write_text(r.stdout+r.stderr,encoding='utf8')
        if r.returncode:raise RuntimeError(r.stderr)
        archive=HERE/f'corrected_onnx_{name}.zip'
        with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED) as z:
            for p in sorted(folder.rglob('*')):
                if p.is_file() and '__pycache__' not in str(p):z.write(p,p.relative_to(folder).as_posix())
        with zipfile.ZipFile(archive) as z:assert z.testzip() is None
        digest=sha(archive);archive.with_suffix('.zip.sha256').write_text(digest+'  '+archive.name+'\n',encoding='ascii')
        output.append(dict(package=name,path=str(archive),bytes=archive.stat().st_size,sha256=digest,models=len(contracts),
            portable_verification=json.loads((folder/'local_verification.json').read_text())))
        print(name,'passed',len(contracts),'models',archive.stat().st_size,'bytes',flush=True)
    (HERE/'packages.json').write_text(json.dumps(output,indent=2),encoding='utf8')

if __name__=='__main__':main()
