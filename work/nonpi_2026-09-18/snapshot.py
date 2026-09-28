"""Preserve pre-revision evidence without moving or altering original files."""
from pathlib import Path
import hashlib, json, shutil, subprocess, zipfile
ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'snapshots/nonpi_2026-09-18'
OUT.mkdir(parents=True, exist_ok=True)
archive = OUT / 'prior_evidence_complete.zip'
if archive.exists():
    raise SystemExit('Snapshot exists; refusing to overwrite')
paths = []
for folder in ['docs', 'src', 'results/v2', 'models/v2', 'paper']:
    paths.extend(p for p in (ROOT/folder).rglob('*') if p.is_file() and '__pycache__' not in p.parts)
extra = [Path.home()/'Downloads'/n for n in ['MCE_blink_detection.tex', 'IEEEmce.cls', 'upmath.sty', '표1_추가작업_결과.zip', '전력측정_전달본.zip']]
manifest = []
with zipfile.ZipFile(archive, 'w', zipfile.ZIP_DEFLATED, compresslevel=3) as z:
    for p in paths + extra:
        if not p.exists(): continue
        name = str(p.relative_to(ROOT)) if p.is_relative_to(ROOT) else 'Downloads/'+p.name
        try:
            raw = p.read_bytes()
        except PermissionError:
            manifest.append({'path':str(p), 'error':'PermissionError; not archived'})
            continue
        manifest.append({'path':str(p), 'archive_path':name, 'bytes':len(raw), 'sha256':hashlib.sha256(raw).hexdigest()})
        z.writestr(name,raw)
for args,name in [(['git','status','--short'],'git_status.txt'),(['git','diff','--no-ext-diff'],'git_diff.patch')]:
    p = subprocess.run(args,cwd=ROOT,capture_output=True)
    (OUT/name).write_bytes(p.stdout)
(OUT/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
dest=ROOT/'work/nonpi_2026-09-18/manuscript'
dest.mkdir(exist_ok=True)
for p in extra[:3]: shutil.copy2(p,dest/p.name)
print(f'Archived {len(manifest)} files, {archive.stat().st_size:,} bytes')
