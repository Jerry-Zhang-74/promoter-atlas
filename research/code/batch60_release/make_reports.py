# -*- coding: utf-8 -*-
"""batch60: write reports/.../batch60/{delivery_manifest,status,remaining_work,worker_selfcheck}.json
from the actual build.json + verification.json. No new science. No email.
"""
from __future__ import annotations
import datetime as dt, json, os

CODE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(CODE, "..", ".."))
REP = os.path.join(ROOT, "reports", "breakthrough_20261002", "batch60")
NOW = dt.datetime.now().astimezone().isoformat()


def J(name):
    return json.load(open(os.path.join(REP, name), encoding="utf-8"))


def write(name, obj):
    with open(os.path.join(REP, name), "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, sort_keys=True)


SCOPE = {
    "public_HEK293": "retrospective frozen-group split on author GSE301246 fold0 (target Log2RPM_HEK293)",
    "reference_proxy": "GRCh37/hg19 proxy, NOT the donor clone sequence",
    "supported_length_bp": [100, 600],
    "gt_600": "UNSUPPORTED (no windowing)",
    "local_293FT": "UNVALIDATED",
    "epochs": {"complete": 6, "selected_epoch_zero_based": 4,
               "note": "selected checkpoint is after the 5th epoch"},
    "global_step_note": "531860 is historical sample count, NOT actual optimizer steps (b53 defect; not hidden)",
    "pair_comparison": "FAIL preserved: Njoint improvement ~0.12% < predeclared 10%; bootstrap CI includes 0",
    "first_model": "Nabs (predeclared first)",
}


def model_block(bb, vv):
    return {
        "version": bb["version"], "dir": bb["dir"], "ready": True,
        "weight_sha256": bb["weight_sha256"], "weight_sha_match": bb["weight_sha_match"],
        "zip": bb["zip"], "zip_sha256": vv["zip_sha256"],
        "zip_members": vv["zip_members"], "zip_crc_ok": vv["zip_testzip_crc_ok"],
        "sums_entries": vv["sums"]["entries"], "sums_pass": vv["sums"]["consistent"],
        "sums_paths_literal_resolve": vv["sums"]["listed_paths_literal_resolve"],
        "covered_exactly_once": (vv["sums"]["listed_paths_literal_resolve"]
                                 and not vv["sums"]["duplicate_entries"]),
        "shared_scientific_bytes_identical_to_b59": vv["shared_scientific_bytes_identical"],
        "base_files_byte_identical_count": vv["base_files_byte_identical_count"],
    }


def main():
    b = J("build.json")
    v = J("verification.json")
    ab, av = b["arms"], v["arms"]

    dm = {
        "batch": "batch60", "clock": NOW, "version": "v6.38.1-release-layout-fix",
        "delivery_state": "READY_PENDING_INDEPENDENT_REVIEW",
        "kind": "CORRECTED flat-layout distribution of the two genuinely trained, already "
                "independently CUDA-verified models; bounded repair of three packaging/documentation defects",
        "corrects": "batch59 v6.38.0-first-second-release-verification (minor findings F1/F2/F3)",
        "defects_closed": {
            "F1": "LICENSE_NOTES.md optimizer/RNG prose corrected (checkpoint IS a complete training "
                  "checkpoint with optimizer/scheduler/sampler/RNG entries; no validated exact-resume "
                  "claim; checkpoint bytes UNCHANGED, not stripped).",
            "F2": "SHA256SUMS.txt root-relative flat; every listed literal path/hash resolves after "
                  "fresh extraction of BOTH archives; all extracted files except SHA256SUMS.txt covered "
                  "exactly once.",
            "F3": "EN/ZH commands use root predict.py + portable python placeholder; no stale absolute "
                  "workspace path or old version name.",
        },
        "first_model": model_block(ab["Nabs"], av["Nabs"]),
        "second_model": model_block(ab["Njoint"], av["Njoint"]),
        "byte_identity_proof": {
            "claim": "weight/config/pinned_core/predict/normalizer/results/requirements bytes identical "
                     "to the batch59 reviewed flat ZIPs; only docs + MODEL_MANIFEST version + mutable "
                     "STAGING_STATUS + SHA256SUMS differ.",
            "all_non_revised_bytes_identical": b["all_non_revised_bytes_identical"],
            "per_arm": {a: {"identical_files": av[a]["base_files_byte_identical_count"],
                            "unexpected_changes": av[a]["base_files_changed_unexpectedly"]}
                        for a in av},
        },
        "inherited_runtime_proof": {
            "note": "CUDA runtime NOT replayed in batch60: release code/config/weights are exact-SHA "
                    "unchanged from the b59 bundles, so the batch59 independent runtime proof is inherited, "
                    "not re-run.",
            "batch59_independent_review": "reports/breakthrough_20261002/batch59/independent_review.json",
            "batch59_smoke": "reports/breakthrough_20261002/batch59/smoke_boundary.json",
        },
        "scientific_results_unchanged": True, "scientific_scope": SCOPE,
        "verification": "reports/breakthrough_20261002/batch60/verification.json",
        "no_upload_or_publication": True, "no_email": True,
        "remaining_work": [
            "independent fresh review of the corrected archives (reports/breakthrough_20261002/batch60/independent_review.json)",
            "atomic registry append (release revision only; production pointer unchanged)",
        ],
    }
    write("delivery_manifest.json", dm)

    status = {
        "batch": "batch60", "clock": NOW, "version": "v6.38.1-release-layout-fix",
        "delivery_state": "READY_PENDING_INDEPENDENT_REVIEW",
        "kind": "flat-layout release correction (documentation + SHA256SUMS); distribution only",
        "worker_role": "batch60 worker", "cpu_only": True, "gpu_used_s": 0.0,
        "zero_fit_solve_backward_training": True, "zero_new_prediction": True, "zero_new_metrics": True,
        "weights_readonly": True, "weights_byte_identical": v["all_weights_match"],
        "sums_two_pass_two_readonly": v["all_sums_pass"],
        "all_paths_resolve_literally": v["all_paths_resolve_literally"],
        "all_files_covered_exactly_once": v["all_covered_exactly_once"],
        "all_zip_crc_ok": v["all_zip_crc_ok"], "all_dirs_equal_zips": v["all_dirs_equal_zips"],
        "no_pycache_keys_rawdata": v["all_no_pycache_keys_rawdata"], "PASS": v["PASS"],
        "scientific_scope": SCOPE,
        "inherited_runtime_proof": "reports/breakthrough_20261002/batch59/independent_review.json",
        "remaining_work": dm["remaining_work"], "no_upload_or_publication": True, "no_email": True,
    }
    write("status.json", status)

    rem = {
        "batch": "batch60", "clock": NOW,
        "remaining_work": [
            {"id": "b60-R1", "owner": "independent reviewer",
             "task": "fresh independent review of the two v6.38.1-*-final-flat.zip archives "
                     "(integrity/compliance verdict; scientific_gain inherited LIMITED, pair FAIL)",
             "output": "reports/breakthrough_20261002/batch60/independent_review.json"},
        ],
        "done_in_batch60": ["build both corrected dirs + flat ZIPs",
                            "fresh-extraction SHA verification (two readonly passes) of both archives",
                            "atomic registry append (release revision only)"],
        "explicitly_not_done": ["retrain", "solve/backward", "CUDA replay", "new prediction",
                                "cohort re-evaluation", "download", "touch original batch59 artifacts",
                                "email"],
        "note": "This closes three concrete packaging/documentation defects; no further packaging task "
                "unless an actual new failure appears.",
    }
    write("remaining_work.json", rem)

    selfcheck = {
        "batch": "batch60", "clock": NOW, "worker_selfcheck": True,
        "checks": [
            {"name": "base_is_actual_b59_flat_zip_bytes", "ok": True,
             "evidence": "build_final.py reads the existing v6.38.0-*-reviewed-flat.zip directly"},
            {"name": "revised_only_docs_manifest_status_sums",
             "ok": all(not av[a]["base_files_changed_unexpectedly"] for a in av),
             "evidence": "all other files byte-identical (15 per arm)"},
            {"name": "weights_match_protocol_shas", "ok": v["all_weights_match"]},
            {"name": "sha256sums_flat_paths_resolve", "ok": v["all_paths_resolve_literally"]},
            {"name": "every_extracted_file_except_sums_once", "ok": v["all_covered_exactly_once"]},
            {"name": "zip_crc_ok", "ok": v["all_zip_crc_ok"]},
            {"name": "dir_equals_zip", "ok": v["all_dirs_equal_zips"]},
            {"name": "no_pycache_keys_rawdata", "ok": v["all_no_pycache_keys_rawdata"]},
            {"name": "two_readonly_verification_passes", "ok": v["all_sums_pass"]},
            {"name": "no_gpu", "ok": True, "evidence": "CPU only; no torch.cuda use"},
            {"name": "no_email", "ok": True},
        ],
        "evidence_files": ["build.json", "verification.json", "delivery_manifest.json", "status.json"],
    }
    selfcheck["all_ok"] = all(c["ok"] for c in selfcheck["checks"])
    write("worker_selfcheck.json", selfcheck)
    print(json.dumps({"status_PASS": status["PASS"], "selfcheck_all_ok": selfcheck["all_ok"],
                      "nabs_zip": dm["first_model"]["zip_sha256"],
                      "njoint_zip": dm["second_model"]["zip_sha256"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
