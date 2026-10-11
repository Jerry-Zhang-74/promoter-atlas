# -*- coding: utf-8 -*-
import json, hashlib
from pathlib import Path
ROOT=Path("E:/promoter_atlas")
B=ROOT/"reports/breakthrough_20261002/batch50"
IMMUT=[
 "figures/fig1_local293ft_applicability.png","figures/fig1_local293ft_applicability.svg",
 "figures/fig2_public_transfer_sameassay.png","figures/fig2_public_transfer_sameassay.svg",
 "figures/fig3_route_capability.png","figures/fig3_route_capability.svg",
 "captions_zh.md","ROUTE_zh.txt","figure_inputs.json","source_evidence_map.json",
 "artifacts_report.md","worker_selfcheck.json","remaining_work.json","next_batch_proposal.json",
 "../batch49/review_notes_addendum_20261004.json",
]
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
entries=[]
for rel in IMMUT:
    p=(B/rel).resolve()
    assert p.exists(), rel
    entries.append({"path":rel.replace("\\","/"),"bytes":p.stat().st_size,"sha256":sha(p)})
man={"batch":"batch50","version":"v6.36.0-morning-route-evidence.rc1",
     "generated_at":"2026-10-04T06:10:30+08:00",
     "excluded_from_manifest":["sha_manifest.json(self)","status.json(mutable)","independent_review.json(review)","manifest_verification.json(verification)","mail"],
     "n":len(entries),"entries":entries}
(B/"sha_manifest.json").write_text(json.dumps(man,ensure_ascii=False,indent=1),encoding="utf-8")
print("manifest entries:",len(entries))
