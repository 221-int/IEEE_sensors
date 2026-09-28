from pathlib import Path
import hashlib,json,shutil
ROOT=Path(__file__).resolve().parents[2]; HERE=Path(__file__).resolve().parent
paths=['src/v2/dataset/crop.py','src/v2/train_encoder.py','src/v2/deploy/run_video.py',
       'src/v2/deploy/run_video_power.py','src/v2/deploy/pi_demo.py']
manifest={}
for rel in paths:
    p=ROOT/rel; q=HERE/'before'/rel; q.parent.mkdir(parents=True,exist_ok=True)
    if q.exists(): raise RuntimeError('Refuse to replace preserved source '+str(q))
    shutil.copy2(p,q); manifest[rel]=hashlib.sha256(p.read_bytes()).hexdigest()
prov=json.loads((ROOT/'work/nonpi_2026-09-21_maskfix/provenance.json').read_text())
manifest['manuscript_hashes']=prov['manuscript_hashes']
(HERE/'before_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf8')
