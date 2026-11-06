#!/usr/bin/env python3
"""Rewrite the EXPECTED sha256 table in verify_hpc_ready.py from the local files.

Run from the repo root after editing any listed file; pass extra relative paths to add them.
"""
import hashlib, json, re, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
target = ROOT / "scripts/hpc/verify_hpc_ready.py"
text = target.read_text()
match = re.search(r"EXPECTED = \{.*?\n\}\n", text, re.S)
current = json.loads(re.sub(r",\s*\}", "}", match.group(0)[len("EXPECTED = "):]))
for rel in sys.argv[1:]:
    current.setdefault(rel, "")
for rel in current:
    current[rel] = hashlib.sha256((ROOT / rel).read_bytes()).hexdigest()
body = ",\n".join(f' "{k}": "{v}"' for k, v in current.items())
target.write_text(text.replace(match.group(0), "EXPECTED = {\n" + body + "\n}\n"))
print(f"updated {len(current)} hashes in {target.relative_to(ROOT)}")
