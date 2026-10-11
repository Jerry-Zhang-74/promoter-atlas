# -*- coding: utf-8 -*-
"""batch59 worker: ATOMIC append of the report/release revision only (production pointer
UNCHANGED). Refuses a double-append. Copies nothing into the sealed packs."""
from __future__ import annotations
import datetime as dt, json, os

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
REG = os.path.join(ROOT, "reports", "breakthrough_20261002", "version_registry.json")
CHG = os.path.join(ROOT, "reports", "breakthrough_20261002", "CHANGELOG.txt")
REP = os.path.join(ROOT, "reports", "breakthrough_20261002", "batch59")
VER = "v6.38.0-first-second-release-verification"


def main():
    now = dt.datetime.now().astimezone().isoformat()
    reg = json.load(open(REG, encoding="utf-8"))
    if any(e.get("batch") == "batch59" for e in reg.get("batch_appends", [])):
        raise SystemExit("batch59 already appended; refusing to double-append")
    st = json.load(open(os.path.join(REP, "status.json"), encoding="utf-8"))
    dm = json.load(open(os.path.join(REP, "delivery_manifest.json"), encoding="utf-8"))
    rec = {"record_type": "release_verification_revision", "batch": "batch59", "version": VER,
           "kind": "FINAL concrete delivery of the two genuine trained models: independent SHA verification of "
                   "both weights + all manifest-listed files (two read-only passes), real small-GPU smoke "
                   "(total %.3fs), external Chinese/space cwd + strict-input boundary tests, reviewed release "
                   "revisions and flat competition ZIPs. NOT a new training/science batch; NOT project completion." % st["gpu_used_s"],
           "appended_by": "batch59 worker", "appended_at": now,
           "production_pointer_unchanged": "v6.3 main / v6.4 cellline pilot",
           "delivery_state": st["delivery_state"],
           "first_model": dm["first_model"], "second_model": dm["second_model"],
           "weight_sha_verified": st["weight_sha_verified"],
           "packaging_fix": st["packaging_fix"],
           "zero_fit_solve_backward_new_metrics_rebootstrap": True,
           "independent_review_pending": "reports/breakthrough_20261002/batch59/independent_review.json"}
    reg["batch_appends"].append(rec)
    reg["last_appended_at"] = now
    reg["last_appended_by"] = "batch59 worker"
    tmp = REG + ".tmp"
    json.dump(reg, open(tmp, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    os.replace(tmp, REG)
    old = open(CHG, encoding="utf-8").read()
    if not old.endswith("\n"):
        old += "\n"
    line = ("[" + now + "] batch59 " + VER + " -> FINAL concrete delivery of the two genuine trained models "
            "(Nabs first / Njoint second). Independent SHA verification: weights match expected "
            "17a6f622...130f / c2698add...0cee, and EVERY file listed in each release SHA256SUMS.txt "
            "re-hashes correctly in two independent read-only passes. Real small-GPU smoke (total %.3fs, "
            "budget 30s): Nabs reproduces the cached known TEST record (abs diff 2e-08 <= tol 1e-4); Njoint "
            "runs 3 new artificial FASTA on CUDA. External Chinese+space cwd + flat-zip extract read test OK; "
            "strict-input boundaries 601/2000/invalid/empty/duplicate all fail explicitly (rc=2, per-record "
            "messages). Concrete packaging fix in the NEW reviewed revision only: MODEL_MANIFEST checkpoint.path "
            "made bundle-relative (weight bytes/SHA unchanged; staged originals PRESERVED). Scientific scope "
            "unchanged: public HEK293 retrospective, reference proxy, local 293FT UNVALIDATED, 6 epochs with "
            "selected 0-based epoch 4 (after 5th epoch), global_step=531860 is historical sample count not "
            "optimizer steps (b53 defect, not hidden), >600 UNSUPPORTED, pair comparison FAIL preserved "
            "(~0.12%% < 10%%, CI includes 0). ZERO fit/solve/backward/training/new-metrics/rebootstrap; no "
            "secrets/raw data/reference genome packaged; no upload/publication; independent review pending. "
            "Production pointer unchanged (v6.3 main / v6.4 cellline pilot)." % st["gpu_used_s"])
    open(CHG + ".tmp", "w", encoding="utf-8").write(old + "\n" + line + "\n")
    os.replace(CHG + ".tmp", CHG)
    print(json.dumps({"registry": REG, "batch_appends": len(reg["batch_appends"]),
                      "changelog": CHG, "appended_at": now}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
