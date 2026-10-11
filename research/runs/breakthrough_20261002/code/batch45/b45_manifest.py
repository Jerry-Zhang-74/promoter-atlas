# -*- coding: utf-8 -*-
"""batch45: immutable-output SHA256 manifest + two read-only verifications."""
import json, hashlib, datetime
from pathlib import Path
OUT=Path("E:/promoter_atlas/reports/breakthrough_20261002/batch45")
def sha(p):
    h=hashlib.sha256()
    with open(p,"rb") as f:
        for b in iter(lambda:f.read(1<<20),b""): h.update(b)
    return h.hexdigest()
EXCLUDE={"sha_manifest.json","status.json","worker_selfcheck.json","independent_review.json","_per.pkl"}
files=[]
for p in sorted(OUT.rglob("*")):
    if p.is_file() and p.name not in EXCLUDE:
        files.append(p)
man={"batch":"batch45","version":"v6.32.0-synthetic-design-utility.rc1",
     "generated_at":datetime.datetime.now().astimezone().isoformat(),
     "excludes":sorted(EXCLUDE),"n_files":len(files),
     "files":{str(p.relative_to(OUT)).replace("\\","/"):{"sha256":sha(p),"bytes":p.stat().st_size} for p in files}}
(OUT/"sha_manifest.json").write_text(json.dumps(man,ensure_ascii=False,indent=2),encoding="utf-8")
# verify twice, read-only
for rnd in (1,2):
    ok=True; bad=[]
    for rel,rec in man["files"].items():
        p=OUT/rel
        if not p.exists() or sha(p)!=rec["sha256"]: ok=False; bad.append(rel)
    print("verify round",rnd,"OK" if ok else ("MISMATCH "+str(bad)))
print("manifest files:", man["n_files"])
