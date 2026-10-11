# -*- coding: utf-8 -*-
"""batch59 worker: finalize status.json + delivery_manifest.json (TWO-PASS read-only verify)."""
from __future__ import annotations
import datetime as dt, hashlib, json, os

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
REP = os.path.join(ROOT, "reports", "breakthrough_20261002", "batch59")
REV = {"Nabs": os.path.join(ROOT, "releases", "batch52_first_model", "v6.38.0-nabs-reviewed"),
       "Njoint": os.path.join(ROOT, "releases", "batch52_second_model", "v6.38.0-njoint-reviewed")}
ZIP = {"Nabs": os.path.join(ROOT, "releases", "batch52_first_model", "v6.38.0-nabs-reviewed-flat.zip"),
       "Njoint": os.path.join(ROOT, "releases", "batch52_second_model", "v6.38.0-njoint-reviewed-flat.zip")}
WSHA = {"Nabs": "17a6f622c3d684743d75263af958cb71dad6f13f99e57f6fc69d5336e74c130f",
        "Njoint": "c2698adde33155e28eace574e05dd42d4eef737002e0c68084053ea9dccb0cee"}


def sh(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def verify_dir(d):
    entries = [l.rstrip("\n") for l in open(os.path.join(d, "SHA256SUMS.txt"), encoding="utf-8") if l.strip()]
    ok = []
    for _ in (1, 2):
        n = 0
        for ln in entries:
            w, rel = ln.split("  ", 1)
            fp = os.path.join(d, rel.replace("/", os.sep))
            if os.path.exists(fp) and sh(fp) == w:
                n += 1
        ok.append(n)
    return {"entries": len(entries), "pass1": ok[0], "pass2": ok[1], "consistent": ok[0] == ok[1] == len(entries)}


def main():
    now = dt.datetime.now().astimezone().isoformat()
    sm = json.load(open(os.path.join(REP, "smoke_boundary.json"), encoding="utf-8"))
    vf = json.load(open(os.path.join(REP, "verification.json"), encoding="utf-8"))
    zb = json.load(open(os.path.join(REP, "zip_build.json"), encoding="utf-8"))
    arms = {}
    for a in ("Nabs", "Njoint"):
        ck = os.path.join(REV[a], "model", a + "_selected.ckpt")
        arms[a] = {"reviewed_dir": os.path.relpath(REV[a], ROOT).replace(os.sep, "/"),
                   "reviewed_dir_verification": verify_dir(REV[a]),
                   "weight_sha256": sh(ck), "weight_sha_match": sh(ck) == WSHA[a],
                   "zip": zb["arms"][a]["zip"], "zip_sha256": zb["arms"][a]["zip_sha256"],
                   "zip_testzip_ok": zb["arms"][a]["zip_testzip_ok"],
                   "has_root_predict": zb["arms"][a]["has_root_predict"],
                   "no_pycache": zb["arms"][a]["no_pycache"]}
    state = "READY_PENDING_INDEPENDENT_REVIEW"
    scope = {"public_HEK293": "retrospective frozen-group split on author GSE301246 fold0 (target Log2RPM_HEK293)",
             "reference_proxy": "GRCh37/hg19 proxy, NOT the donor clone sequence",
             "local_293FT": "UNVALIDATED", "supported_length_bp": [100, 600], "gt_600": "UNSUPPORTED",
             "epochs": {"complete": 6, "selected_epoch_zero_based": 4, "note": "selected checkpoint is after the 5th epoch"},
             "global_step_note": "531860 is historical sample count, NOT actual optimizer steps (known b53 defect; checkpoint not edited to hide it)",
             "nabs_resume": "NOT claimed as actual-resume verification (b58 separately does real resume; b57 unsupported-only)",
             "native_window_policy": "center-zero 600; no RC/window/crop/ESB/GFP translation",
             "pair_comparison": "FAIL preserved: Njoint improvement ~0.12% < predeclared 10%; bootstrap CI includes 0"}
    remaining = ["independent fresh review (11:00): reports/breakthrough_20261002/batch59/independent_review.json (verdict integrity/compliance PASS|FAIL; scientific_gain separate)",
                 "atomic registry append of the report/release revision only (production pointer unchanged)"]
    status = {"batch": "batch59", "version": "v6.38.0-first-second-release-verification",
              "kind": "final concrete delivery of the two genuine trained models", "clock": now,
              "worker_role": "batch59 worker (control_71)", "delivery_state": state,
              "blocks_first_model_release": False, "does_not_wait_for_batch58_final_review": True,
              "zero_fit_solve_backward_training": True, "zero_new_metrics": True, "zero_rebootstrap": True,
              "gpu_used_s": sm["gpu_total_s"], "gpu_budget_s": 30.0, "gpu_within_budget": sm["gpu_within_budget"],
              "weights_readonly": True,
              "weight_sha_verified": {"Nabs": vf["releases"]["Nabs"]["source_checkpoint_match"],
                                      "Njoint": vf["releases"]["Njoint"]["source_checkpoint_match"]},
              "staged_originals_preserved": True, "arms": arms,
              "smoke_evidence": {"nabs_known_test_record": {k: sm["gpu_nabs_known_test"][k] for k in
                                  ("cached_activity_pred", "release_activity_pred", "abs_diff", "tol_documented", "match_within_tol")},
                                 "njoint_new_fasta_ok": sm["gpu_njoint_new_fasta"]["ok"],
                                 "external_cwd_chinese_space_path": True,
                                 "boundary_all_explicit_fail": sm["all_boundary_fail"],
                                 "boundary_note": "per-record length/base/id errors are reported per-record; an empty whole-file is reported as whole-file empty (not claimed as a per-row null)."},
              "scientific_scope": scope,
              "packaging_fix": "MODEL_MANIFEST.json checkpoint.path changed from out-of-bundle absolute source path to bundle-relative model/<arm>_selected.ckpt; weight bytes/SHA unchanged (new reviewed revision only; staged originals preserved).",
              "standalone": "self-contained: source code + weights + requirements + actual metrics; no absolute asset dependencies, no keys/credentials, no raw data/reference genome.",
              "remaining_work": remaining}
    json.dump(status, open(os.path.join(REP, "status.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=2, sort_keys=True)
    manifest = {"batch": "batch59", "clock": now, "delivery_state": state,
                "first_model": {"dir": arms["Nabs"]["reviewed_dir"], "zip": arms["Nabs"]["zip"],
                                "weight_sha256": arms["Nabs"]["weight_sha256"], "zip_sha256": arms["Nabs"]["zip_sha256"],
                                "ready": arms["Nabs"]["reviewed_dir_verification"]["consistent"] and arms["Nabs"]["weight_sha_match"]},
                "second_model": {"dir": arms["Njoint"]["reviewed_dir"], "zip": arms["Njoint"]["zip"],
                                 "weight_sha256": arms["Njoint"]["weight_sha256"], "zip_sha256": arms["Njoint"]["zip_sha256"],
                                 "ready": arms["Njoint"]["reviewed_dir_verification"]["consistent"] and arms["Njoint"]["weight_sha_match"]},
                "verification_2pass_readonly": {a: arms[a]["reviewed_dir_verification"] for a in arms},
                "smoke_evidence_file": "reports/breakthrough_20261002/batch59/smoke_boundary.json",
                "scientific_scope": scope, "remaining_work": remaining, "no_upload_or_publication": True}
    json.dump(manifest, open(os.path.join(REP, "delivery_manifest.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=2, sort_keys=True)
    print(json.dumps({"delivery_state": state, "Nabs_ready": manifest["first_model"]["ready"],
                      "Njoint_ready": manifest["second_model"]["ready"]}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
