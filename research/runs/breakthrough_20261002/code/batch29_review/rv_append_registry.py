# -*- coding: utf-8 -*-
import json, hashlib, datetime
from pathlib import Path
ROOT = Path("E:/promoter_atlas")
REP = ROOT/"reports/breakthrough_20261002"
now = datetime.datetime.now().astimezone().isoformat(timespec="seconds")
# 1) CHANGELOG append-only
cl = REP/"CHANGELOG.txt"
before = cl.read_text(encoding="utf-8")
line = ("\n[%s] batch29 v6.20.1-sureproxy.rc1 -> FRESH FLASH INDEPENDENT REVIEW: integrity FAIL / scientific NONE-NOT-ACCEPTED / compliance FAIL-WITH-DISCLOSURE. "
 "HELD. 2/4 estimator executions consumed (E1 ridge, E2 true-CNN seed20261101); E3/E4 not launched. Numeric/data integrity PASS: source SHA 8868fa64...458b85 stable, "
 "independent full-table recount reproduces every headline count exactly (n_rows 153562178, n_eligible 37282554, N_D 203742844, N_R1 37770653, N_R2 74081034, N_iPCR 424419578), "
 "90/90 selected-row target values match max|dy|=0.0, E1/E2 metrics replayed exactly from saved predictions with ZERO fits. Protocol FAIL: E2 used padded pooling to maxlen2000 "
 "(code 4 -> all-zero channel, unmasked mean/max) instead of the approved native-length contract; mandatory pre-fit 100-row target verification not evidenced by worker; "
 "checkpoint not a true resume path (best epoch paired with end-of-fit global_step/RNG); split near-homology is a k-mer approximation not confirmed 90/90 alignments (honestly disclosed). "
 "TEST never opened (no selection_manifest/test metrics), no test-score selection. Production pointer unchanged (v6.3 main / v6.4 cellline pilot). No promotion. "
 "Reviewer zero fits/zero optimizer steps; canonical reports/breakthrough_20261002/batch29/independent_review.json sha256 %s.\n") % (
   now, hashlib.sha256((REP/"batch29"/"independent_review.json").read_bytes()).hexdigest())
cl.write_text(before + line, encoding="utf-8")
# 2) version_registry append-only batch_appends
rp = REP/"version_registry.json"
reg = json.loads(rp.read_text(encoding="utf-8"))
reg.setdefault("batch_appends", []).append({
  "record_type": "held_candidate_review", "batch": "batch29", "candidate_version": "v6.20.1-sureproxy.rc1",
  "track": "K562 SuRE per-fragment RNA/plasmid proxy (D=1+P) native-length CNN/ridge", "status": "HELD; NOT promoted; production pointer unchanged (v6.3 main / v6.4 cellline pilot)",
  "appended_by": "fresh independent Flash reviewer (batch29 review phase)", "appended_at": now,
  "review_verdict": {"integrity": "FAIL(protocol)", "numeric_data_integrity": "PASS", "protocol_compliance": "FAIL", "scientific_acceptance": "NO_GAIN_NOT_ACCEPTED", "breakthrough_accepted": False},
  "estimator_executions_consumed": 2, "estimator_executions_declared": 4,
  "canonical_review": "reports/breakthrough_20261002/batch29/independent_review.json",
  "prior_entries_preserved": True})
reg["last_appended_at"] = now
reg["last_appended_by"] = "fresh independent Flash reviewer (batch29)"
tmp = rp.with_suffix(".json.tmp"); tmp.write_text(json.dumps(reg, ensure_ascii=False, indent=2), encoding="utf-8"); tmp.replace(rp)
chk = json.loads(rp.read_text(encoding="utf-8"))
print("changelog bytes:", cl.stat().st_size, "| registry appends:", len(chk["batch_appends"]), "| parsed ok")
