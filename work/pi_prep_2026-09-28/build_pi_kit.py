"""Build the self-contained Pi transfer kit for paced 30 fps measurements.

Output: work/pi_prep_2026-09-28/pi_kit_2026-09-28/ and pi_kit_2026-09-28.zip (+ .sha256).
The kit is unpacked into a NEW folder on the Pi and run from there; it never
touches the Pi's existing repository or models/v2/onnx.
"""
import hashlib, json, shutil, sys, zipfile
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
NAME = "pi_kit_2026-09-28"
DEST = HERE / NAME
MARK = ".built_by_build_pi_kit"
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()

SRC = ["src/__init__.py", "src/v2/__init__.py", "src/v2/common/__init__.py", "src/v2/common/repro.py",
       "src/v2/dataset/__init__.py", "src/v2/dataset/crop.py", "src/v2/deploy/__init__.py"] + [
    f"src/v2/deploy/{m}.py" for m in (
        "frontend", "inference_policy", "pmic_power", "pacing", "bench_telemetry", "run_paced",
        "paced_session", "check_paced_results", "pi_env_check", "test_paced_bench", "test_pmic_power",
        "run_video", "run_video_power", "pi_demo", "test_pi_demo_ui")]
COPIES = {
    "requirements-pi.txt": "requirements-pi.txt",
    "results/v2/check_equivalence.json": "results/v2/check_equivalence.json",
    "results/v2/power_ours_480p_run1.json": "results/v2/power_ours_480p_run1.json",
    "data/_legacy_public/eyeblink8/eyeblink8/9/27122013_152435_cam.avi": "clips/eyeblink8_v9_cam.avi",
    "work/pi_prep_2026-09-28/protocol/pi5_paced30_v1.json": "protocol.json",
    "work/pi_prep_2026-09-28/kit_files/README_PI.md": "README_PI.md",
    "work/pi_prep_2026-09-28/kit_files/verify_kit.py": "verify_kit.py",
}
ZIP = ROOT / "work/nonpi_2026-09-22_onnx/corrected_onnx_pi_default.zip"


def main():
    if DEST.exists():
        if not (DEST / MARK).exists():
            sys.exit(f"{DEST} exists and was not made by this script; refusing to replace it")
        for child in DEST.iterdir():          # keep the folder itself (may be a shell cwd)
            shutil.rmtree(child) if child.is_dir() else child.unlink()
    DEST.mkdir(exist_ok=True)
    (DEST / MARK).write_text("generated; safe to rebuild\n")
    for rel in SRC:
        (DEST / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / rel, DEST / rel)
    for src, dst in COPIES.items():
        (DEST / dst).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / src, DEST / dst)
    for f in sorted((HERE / "kit_files/scripts").glob("*.sh")):
        (DEST / "scripts").mkdir(exist_ok=True)
        (DEST / "scripts" / f.name).write_bytes(f.read_bytes().replace(b"\r\n", b"\n"))
    expected = (ZIP.with_suffix(".zip.sha256")).read_text().split()[0]
    if sha(ZIP) != expected:
        sys.exit("model ZIP hash differs from its sidecar")
    with zipfile.ZipFile(ZIP) as z:
        z.extractall(DEST / "bundle")
    files = sorted(p for p in DEST.rglob("*") if p.is_file() and p.name not in (MARK, "kit_manifest.json"))
    manifest = {
        "kit": NAME, "built": datetime.now().astimezone().isoformat(),
        "model_zip": {"path": str(ZIP.relative_to(ROOT)).replace("\\", "/"), "sha256": expected},
        "protocol_sha256": sha(DEST / "protocol.json"),
        "sources": {rel: sha(ROOT / rel) for rel in SRC},
        "sha256": {p.relative_to(DEST).as_posix(): sha(p) for p in files},
    }
    (DEST / "kit_manifest.json").write_text(json.dumps(manifest, indent=1), encoding="utf8")
    out = HERE / f"{NAME}.zip"
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        for p in sorted(DEST.rglob("*")):
            if p.is_file() and p.name != MARK:
                info = zipfile.ZipInfo.from_file(p, f"{NAME}/{p.relative_to(DEST).as_posix()}")
                if p.suffix == ".sh":
                    info.external_attr = 0o100755 << 16
                with open(p, "rb") as fh:
                    z.writestr(info, fh.read(), zipfile.ZIP_DEFLATED)
    (HERE / f"{NAME}.zip.sha256").write_text(f"{sha(out)}  {out.name}\n")
    print(json.dumps({"files": len(manifest["sha256"]), "zip": str(out), "bytes": out.stat().st_size,
                      "sha256": sha(out)}, indent=1))


if __name__ == "__main__":
    main()
