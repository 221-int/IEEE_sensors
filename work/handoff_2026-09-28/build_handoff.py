"""Read-only project inventory; writes only this handoff folder."""
from pathlib import Path
import hashlib
import json
import subprocess
import zipfile
from datetime import datetime

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent

def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()

paths = [
    'requirements-pi.txt',
    'docs/STATUS_2026-08-08.md', 'docs/PAPER_OUTLINE.md',
    'docs/v2/PROTOCOL.md', 'docs/EXPERIMENT_PLAN.md',
    'docs/PATENT_AND_FUTURE_WORK.md', 'docs/CHANGELOG.md',
    'src/v2/deploy/power_bench_runner.py', 'src/v2/deploy/pmic_power.py',
    'src/v2/deploy/frontend.py',
    'work/nonpi_2026-09-18/CITATION_AUDIT.md',
    'work/nonpi_2026-09-18/FUTURE_RESEARCH.md',
    'work/nonpi_2026-09-21/submission_audit/rnn_handoff_inventory.json',
    'work/nonpi_2026-09-21_maskfix/PROTOCOL.md',
    'work/nonpi_2026-09-21_maskfix/RESULTS.md',
    'work/nonpi_2026-09-21_maskfix/results/comparison.json',
    'work/nonpi_2026-09-21_maskfix/post_train/status.json',
    'work/nonpi_2026-09-22_review/REVIEW.md',
    'work/nonpi_2026-09-22_review/verification.json',
    'work/nonpi_2026-09-22_review/result_review.json',
    'work/nonpi_2026-09-22_review/run_mean_ci.json',
    'work/nonpi_2026-09-22_review/recrop_summary.json',
    'work/nonpi_2026-09-22_review/recrop_prediction_comparison.json',
    'work/nonpi_2026-09-22_review/changes.patch',
    'work/nonpi_2026-09-22_onnx/DELIVERY.md',
    'work/nonpi_2026-09-22_onnx/README.md',
    'work/nonpi_2026-09-22_onnx/verification.json',
    'work/nonpi_2026-09-22_onnx/packages.json',
    'work/nonpi_2026-09-22_onnx/entrypoint_verification.json',
    'work/nonpi_2026-09-22_onnx/delivery_status.json',
    'work/nonpi_2026-09-22_onnx/gpu_cpu_boundary_diagnosis.json',
    'work/nonpi_2026-09-22_onnx/preserved_before.json',
]
review = json.loads((ROOT / 'work/nonpi_2026-09-22_review/verification.json').read_text(encoding='utf-8'))
paths += [p.replace('\\', '/') for p in review['files']]
files = {}
for rel in sorted(set(paths)):
    path = ROOT / rel
    files[rel] = {'exists': path.is_file()}
    if path.is_file():
        files[rel].update(bytes=path.stat().st_size, sha256=digest(path))

source_checks = {p: digest(ROOT / p) == data['after'] for p, data in review['files'].items()}
old_hashes = json.loads((ROOT / 'work/nonpi_2026-09-22_onnx/preserved_before.json').read_text(encoding='utf-8'))
preserved = {p: Path(p).is_file() and digest(Path(p)) == sha for p, sha in old_hashes.items()}
packages = {}
for name in ['corrected_onnx_pi_default.zip', 'corrected_onnx_all_60.zip']:
    path = ROOT / 'work/nonpi_2026-09-22_onnx' / name
    sha = digest(path)
    expected = path.with_suffix('.zip.sha256').read_text().split()[0]
    packages[name] = {'path': str(path), 'bytes': path.stat().st_size, 'sha256': sha, 'matches_sidecar': sha == expected}
counts = {v: len(list((ROOT / 'work/nonpi_2026-09-21_maskfix/records' / v).glob('fold*_seed*.json'))) for v in ['vpres', 'image_head', 'vdrop', 'gap']}
git = subprocess.run(['git', 'status', '--short'], cwd=ROOT, capture_output=True, text=True, encoding='utf-8', errors='replace')
inventory = {
    'checked_at': datetime.now().astimezone().isoformat(), 'project': str(ROOT),
    'scope': 'File existence and SHA256 checks only; no new training or Pi measurement.',
    'files': files, 'source_matches_2026_09_22_review': source_checks,
    'legacy_models_and_manuscripts_match_onnx_snapshot': preserved,
    'corrected_model_packages': packages, 'fit_record_file_counts': counts,
    'git_status': git.stdout, 'git_status_stderr': git.stderr,
}
(HERE / 'current_inventory.json').write_text(json.dumps(inventory, ensure_ascii=False, indent=2), encoding='utf-8')
archive = HERE / 'handoff_context.zip'
with zipfile.ZipFile(archive, 'w', compression=zipfile.ZIP_DEFLATED) as z:
    for name in ['HANDOFF.md', 'NEW_SESSION_PROMPT.txt', 'current_inventory.json', 'build_handoff.py']:
        z.write(HERE / name, name)
    for rel, info in files.items():
        if info['exists']:
            z.write(ROOT / rel, 'project_context/' + rel)
archive.with_suffix('.zip.sha256').write_text(digest(archive) + '  ' + archive.name + '\n', encoding='ascii')
print(json.dumps({
    'missing_files': [p for p, i in files.items() if not i['exists']],
    'changed_review_sources': [p for p, ok in source_checks.items() if not ok],
    'changed_preserved_files': [p for p, ok in preserved.items() if not ok],
    'packages_match': all(p['matches_sidecar'] for p in packages.values()),
    'record_counts': counts, 'context_zip_bytes': archive.stat().st_size,
}, ensure_ascii=True, indent=2))
