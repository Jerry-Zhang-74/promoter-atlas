# -*- coding: utf-8 -*-
"""Pure-function test: does the LIVE stage watcher now classify batch27/28 reviews?"""
import json, sys
from pathlib import Path
ROOT = Path("E:/promoter_atlas")
sys.path.insert(0, str(ROOT/"runs/breakthrough_20261002/code/batch19/stage_email_rc3"))
import stage_watcher as sw   # import only; no send, no credentials at import time

out = {}
for batch, ctl in (("batch27","control_34"), ("batch28","control_35")):
    rj = json.load(open(ROOT/"reports/breakthrough_20261002"/batch/"independent_review.json", encoding="utf-8"))
    cls = sw.classify_review_verdict(rj)
    cs = json.load(open(ROOT/"runs/breakthrough_20261002"/ctl/"status.json", encoding="utf-8"))
    rev = sw.detect_review(batch, cs, rj, ROOT/"reports/breakthrough_20261002"/batch/"independent_review.json")
    out[batch] = {"classify": {k: cls[k] for k in ("kind","verdict","blocked","block_reason")},
                  "detect_review": ({k: rev[k] for k in ("done","verdict","original_verdict","verdict_kind")} if rev else None),
                  "scientific_original_prefix": (cls.get("scientific_original") or "")[:80]}
json.dump(out, open(ROOT/"reports/breakthrough_20261002/batch29_review/classify_test.json","w",encoding="utf-8"),
          ensure_ascii=False, indent=2)
print(json.dumps(out, ensure_ascii=False, indent=2))
