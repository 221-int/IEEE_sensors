from pathlib import Path
import sys,json,hashlib,subprocess,os,difflib,datetime
ROOT=Path(__file__).resolve().parents[2];HERE=Path(__file__).resolve().parent
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
env=dict(os.environ,PYTHONIOENCODING='utf-8',PYTHONHASHSEED='0')
tests=[];log=[]
for args in [[str(HERE/'test_policy.py')],[str(HERE/'test_entrypoints.py')],['-m','unittest','src.v2.deploy.test_pi_demo_ui','-v']]:
    r=subprocess.run([sys.executable,*args],cwd=ROOT,env=env,capture_output=True,text=True,encoding='utf8')
    tests.append(dict(command=args,exit_code=r.returncode));log.append('COMMAND '+str(args)+'\n'+r.stdout+r.stderr)
    if r.returncode:raise RuntimeError(log[-1])
for mod in ['run_video','run_video_power','pi_demo']:
    r=subprocess.run([sys.executable,'-m','src.v2.deploy.'+mod,'--help'],cwd=ROOT,env=env,capture_output=True,text=True,encoding='utf8')
    tests.append(dict(command=[mod,'--help'],exit_code=r.returncode))
    if r.returncode:raise RuntimeError(r.stderr)
before=json.loads((HERE/'before_manifest.json').read_text());manuscripts={p:sha(p)==h for p,h in before['manuscript_hashes'].items()}
assert all(manuscripts.values())
preserved={p:sha(HERE/'before'/p)==h for p,h in before.items() if p!='manuscript_hashes'}
assert all(preserved.values())
sys.path.insert(0,str(ROOT/'work/nonpi_2026-09-21_maskfix'));import experiment_queue as Q
counts={v:len(Q.records(v)) for v in Q.VARIANTS};assert all(n==15 for n in counts.values())
prov=json.loads((ROOT/'work/nonpi_2026-09-21_maskfix/provenance.json').read_text())
historical_source_changes=[p for p,h in prov['source_hashes'].items() if sha(p)!=h]
assert set(Path(p).name for p in historical_source_changes)=={'crop.py','train_encoder.py'}
changes=[];files={}
for p in sorted((HERE/'before').rglob('*.py')):
    rel=p.relative_to(HERE/'before');current=ROOT/rel
    files[str(rel)]=dict(before=sha(p),after=sha(current))
    changes.extend(difflib.unified_diff(p.read_text(encoding='utf8').splitlines(True),current.read_text(encoding='utf8').splitlines(True),fromfile='before/'+rel.as_posix(),tofile='after/'+rel.as_posix()))
policy=ROOT/'src/v2/deploy/inference_policy.py'
changes.extend(difflib.unified_diff([],policy.read_text(encoding='utf8').splitlines(True),fromfile='/dev/null',tofile='after/src/v2/deploy/inference_policy.py'))
files[str(policy.relative_to(ROOT))]=dict(before=None,after=sha(policy))
(HERE/'changes.patch').write_text(''.join(changes),encoding='utf8')
(HERE/'verification.log').write_text('\n'.join(log),encoding='utf8')
result=dict(created=datetime.datetime.now().astimezone().isoformat(),tests=tests,total_unittest_cases=14,
    manuscript_unchanged=manuscripts,original_source_backups_verified=preserved,corrected_fit_records_verified=counts,
    historical_frozen_source_changes=historical_source_changes,files=files,
    limitations=['Mock camera/ONNX integration checks are not new Pi measurements.',
                'Old frozen queue intentionally cannot prepare against modified source; original files preserved.',
                'Corrected-checkpoint ONNX exports are a separate next step.'])
(HERE/'verification.json').write_text(json.dumps(result,indent=2),encoding='utf8');print(json.dumps(result,indent=2))
