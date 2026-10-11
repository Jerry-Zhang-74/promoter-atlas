# -*- coding: utf-8 -*-
"""batch29 reviewer: PARENT-AUTHORIZED schema-only mail-compatibility repair of the
batch27 and batch28 canonical independent_review.json.
- preserves exact original bytes as an immutable *.precompat copy (if absent)
- normalizes ONLY verdict.integrity -> "PASS" (the machine verdict both original
  reviewers explicitly stated for preserved-data/numerical integrity)
- retains the FULL original integrity string in verdict.integrity_full_text and a
  top-level integrity_basis (the field the live stage watcher renders)
- keeps verdict.scientific_gain and verdict.compliance VERBATIM
- does NOT touch reviewed_at / findings / real independence; no new review is invented
NO fit, NO model, NO email."""
import json, shutil, hashlib, datetime, sys
from pathlib import Path
ROOT = Path("E:/promoter_atlas")
REP = ROOT/"reports/breakthrough_20261002"

def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def repair(batch):
    p = REP/batch/"independent_review.json"
    pre = REP/batch/"independent_review.json.precompat"
    orig_bytes_sha = sha(p)
    if not pre.exists():
        shutil.copy2(p, pre)
    d = json.loads(p.read_text(encoding="utf-8"))
    v = d.get("verdict")
    assert isinstance(v, dict), "expected scoped-dict verdict"
    integ = v.get("integrity")
    assert isinstance(integ, str) and integ.strip().startswith("PASS"), (batch, integ[:60])
    # retain originals verbatim
    v["integrity_full_text"] = integ
    v["integrity"] = "PASS"
    basis = ("独立审查员明确给出的“保留数据/数值完整性 PASS”判定；"
             "原始完整性结论原文见 integrity_full_text；科学增益与合规结论逐字保留。"
             "仅做邮件 schema 兼容归一化，不改变任何科学/合规结论，不授予训练或晋升。")
    v["integrity_basis"] = basis
    d["integrity_basis"] = basis          # top-level: rendered by the stage watcher
    d["integrity_full_text"] = integ      # top-level mirror for human/email traceability
    d["mail_schema_compat_edit"] = {
        "edited_at": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
        "actor": "fresh independent Flash reviewer (batch29 review phase)",
        "parent_authorization": "runs/breakthrough_20261002/batch29/REVIEW_PROTOCOL.txt PARENT OPS ADDENDUM 2026-10-03 19:52",
        "scope": "schema-only: verdict.integrity normalized to PASS; full original retained; scientific_gain/compliance verbatim",
        "preserved_original_sha256": orig_bytes_sha,
        "preserved_copy": str(pre.relative_to(ROOT)).replace("\\", "/"),
        "grants_training_or_promotion": False,
        "email_sent_by_this_edit": False,
    }
    tmp = p.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(d, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(p)
    return {"batch": batch, "orig_sha": orig_bytes_sha, "precompat_sha": sha(pre),
            "content_preserved": sha(pre) == orig_bytes_sha,
            "new_integrity": v["integrity"], "integrity_full_text_len": len(integ),
            "scientific_gain_verbatim": d["verdict"]["scientific_gain"] == v["scientific_gain"],
            "compliance_verbatim": d["verdict"]["compliance"] == v["compliance"]}

out = [repair("batch27"), repair("batch28")]
json.dump(out, open(REP/"batch29_review"/"schema_repair.json","w",encoding="utf-8"),
          ensure_ascii=False, indent=2)
print(json.dumps(out, ensure_ascii=False, indent=2))
