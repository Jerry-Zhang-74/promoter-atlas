# -*- coding: utf-8 -*-
import json, hashlib
from pathlib import Path
ROOT=Path("E:/promoter_atlas"); B=ROOT/"reports/breakthrough_20261002/batch50"
man=json.loads((B/"sha_manifest.json").read_text(encoding="utf-8"))
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def verify(pass_tag):
    ok=0; bad=[]
    for e in man["entries"]:
        p=(B/e["path"]).resolve()
        if (not p.exists()) or sha(p)!=e["sha256"] or p.stat().st_size!=e["bytes"]:
            bad.append(e["path"])
        else: ok+=1
    return {"pass":pass_tag,"read_only":True,"n_checked":len(man["entries"]),"ok":ok,"mismatches":bad,"all_match":len(bad)==0}
v1=verify("verification_1")
v2=verify("verification_2")
out={"batch":"batch50","version":"v6.36.0-morning-route-evidence.rc1",
     "generated_at":"2026-10-04T06:11:00+08:00",
     "manifest_sha256":hashlib.sha256((B/"sha_manifest.json").read_bytes()).hexdigest(),
     "verification_1":v1,"verification_2":v2,
     "two_independent_readonly_checks_consistent":(v1==dict(v2) or (v1["all_match"] and v2["all_match"])),
     "note":"both read-only; excludes manifest/status/review/verification/mail"}
(B/"manifest_verification.json").write_text(json.dumps(out,ensure_ascii=False,indent=1),encoding="utf-8")
print(json.dumps({"v1_all_match":v1["all_match"],"v2_all_match":v2["all_match"],"n":v1["n_checked"],"bad":v1["mismatches"]},ensure_ascii=False))
