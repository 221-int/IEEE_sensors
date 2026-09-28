"""Re-check the 2026-09-28 handoff inventory before new Pi-prep work (read-only)."""
import hashlib, json, os, sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
os.chdir(ROOT)
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
inv = json.loads(Path('work/handoff_2026-09-28/current_inventory.json').read_text(encoding='utf8'))
rev = json.loads(Path('work/nonpi_2026-09-22_review/verification.json').read_text(encoding='utf8'))
out = {'checked_at': datetime.now().astimezone().isoformat(), 'inventory_files': {}, 'review_sources': {}, 'packages': {}}
for p, rec in inv['files'].items():
    out['inventory_files'][p] = Path(p).exists() and sha(p) == rec['sha256']
for p, rec in rev['files'].items():
    out['review_sources'][p] = sha(p) == rec['after']
for name, rec in inv['corrected_model_packages'].items():
    out['packages'][name] = sha(rec['path']) == rec['sha256']
out['all_match'] = all(v for k in ('inventory_files', 'review_sources', 'packages') for v in out[k].values())
Path('work/pi_prep_2026-09-28/preflight_inventory.json').write_text(json.dumps(out, indent=1), encoding='utf8')
print(json.dumps({k: (v if not isinstance(v, dict) else f"{sum(v.values())}/{len(v)}") for k, v in out.items()}))
sys.exit(0 if out['all_match'] else 1)
