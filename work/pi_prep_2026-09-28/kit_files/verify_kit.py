"""Check every kit file against kit_manifest.json (stdlib only).

    python verify_kit.py            # from the kit root
Files created later (results/, bundle/pi_model_verification.json, __pycache__) are ignored.
"""
import hashlib, json, sys
from pathlib import Path

root = Path(__file__).resolve().parent
m = json.loads((root / "kit_manifest.json").read_text(encoding="utf8"))
bad = []
for rel, h in m["sha256"].items():
    p = root / rel
    if not p.is_file():
        bad.append(f"missing {rel}")
    elif hashlib.sha256(p.read_bytes()).hexdigest() != h:
        bad.append(f"hash mismatch {rel}")
known = set(m["sha256"]) | {"kit_manifest.json", "bundle/pi_model_verification.json"}
extra = [p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file()]
extra = [e for e in extra if e not in known and "__pycache__" not in e
         and not e.startswith("results/pi_paced/") and not e.endswith(".tgz")
         and e != ".built_by_build_pi_kit"]
print(json.dumps({"kit": m["kit"], "files_checked": len(m["sha256"]), "problems": bad,
                  "unexpected_new_files": extra}, indent=1))
sys.exit(1 if bad else 0)
