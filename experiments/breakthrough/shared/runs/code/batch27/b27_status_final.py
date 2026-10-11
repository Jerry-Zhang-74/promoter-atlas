# -*- coding: utf-8 -*-
import json, hashlib, time
R = "reports/breakthrough_20261002/batch27/"
CL = "reports/breakthrough_20261002/CHANGELOG.txt"
REG = "reports/breakthrough_20261002/version_registry.json"
CARD = "models/cards/promoter_atlas_v6_19_1_promoter_readiness_rc1_card.txt"
def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()
st = json.load(open(R + "status.json", encoding="utf-8"))
st["updated_at_local"] = time.strftime("%Y-%m-%dT%H:%M:%S+08:00")
st["artifacts"].update({
  "source_exposure_audit.json": sha(R + "source_exposure_audit.json"),
  "next_batch_proposal.json": sha(R + "next_batch_proposal.json"),
  "selfcheck.json": sha(R + "selfcheck.json"),
  "tier2_cohort": sha("data/breakthrough_20261002/batch27/sure_tier2_cohort.tsv.gz"),
  "source_primary_table": sha("data/breakthrough_20261002/batch27/source/GSE78709_SuRE-counts-K562_B45_B55_LP170105.txt.gz"),
  "CHANGELOG.txt": sha(CL), "version_registry.json": sha(REG), "version_card": sha(CARD),
  "PARENT_FINDINGS_ADDENDUM_20261003.txt": sha("runs/breakthrough_20261002/batch27/PARENT_FINDINGS_ADDENDUM_20261003.txt")
})
st["code_dir"] = "runs/breakthrough_20261002/code/batch27"
st["handoff"] = {
  "worker_handoff_deadline": "2026-10-03T20:00:00+08:00",
  "review_deadline": "2026-10-03T20:40:00+08:00",
  "finished_early": True,
  "worker_writes_only_selfcheck": True,
  "independent_review_written_by": "fresh reviewer -> " + R + "independent_review.json",
  "emails_sent_by_worker": 0,
  "stage_watcher_pid": 4248,
  "note": "batch25 and batch26 emails already accepted - never resend; watcher owns stage milestones"
}
json.dump(st, open(R + "status.json", "w", encoding="utf-8"), indent=2)
print("status finalized", st["updated_at_local"])
