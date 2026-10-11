# -*- coding: utf-8 -*-
"""batch60: ATOMIC append of the release revision only; production pointer UNCHANGED.
Refuses a double-append. Copies nothing into the sealed packs. No email.
"""
from __future__ import annotations
import datetime as dt, json, os

CODE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(CODE, "..", ".."))
REG = os.path.join(ROOT, "reports", "breakthrough_20261002", "version_registry.json")
CHG = os.path.join(ROOT, "reports", "breakthrough_20261002", "CHANGELOG.txt")
REP = os.path.join(ROOT, "reports", "breakthrough_20261002", "batch60")
VER = "v6.38.1-release-layout-fix"


def main():
    now = dt.datetime.now().astimezone().isoformat()
    reg = json.load(open(REG, encoding="utf-8"))
    if any(e.get("batch") == "batch60" for e in reg.get("batch_appends", [])):
        raise SystemExit("batch60 already appended; refusing to double-append")
    dm = json.load(open(os.path.join(REP, "delivery_manifest.json"), encoding="utf-8"))
    rec = {"record_type": "release_revision_only",
           "batch": "batch60", "version": VER,
           "kind": "CORRECTED flat-layout distribution of the two genuinely trained, "
                   "already independently CUDA-verified models (batch59). Repairs three concrete "
                   "packaging/documentation defects (F1 LICENSE_NOTES optimizer/RNG prose, "
                   "F2 root-relative flat SHA256SUMS, F3 stale absolute-path/version in EN/ZH "
                   "commands). NOT a new model/performance claim; NO retraining/prediction.",
           "appended_by": "batch60 worker",
           "appended_at": now,
           "production_pointer_unchanged": reg.get("production_pointer"),
           "first_model": dm["first_model"], "second_model": dm["second_model"],
           "weight_sha_verified": {"Nabs": dm["first_model"]["weight_sha256"] ==
                                   "17a6f622c3d684743d75263af958cb71dad6f13f99e57f6fc69d5336e74c130f",
                                   "Njoint": dm["second_model"]["weight_sha256"] ==
                                   "c2698adde33155e28eace574e05dd42d4eef737002e0c68084053ea9dccb0cee"},
           "scientific_results_unchanged": True,
           "new_revision_files": ["releases/batch52_first_model/v6.38.1-nabs-final",
                                  "releases/batch52_second_model/v6.38.1-njoint-final"],
           "inherited_runtime_proof": "reports/breakthrough_20261002/batch59/independent_review.json"}
    reg["batch_appends"].append(rec)
    reg["last_appended_at"] = now
    reg["last_appended_by"] = "batch60 worker"
    tmp = REG + ".tmp"
    json.dump(reg, open(tmp, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    os.replace(tmp, REG)

    old = open(CHG, encoding="utf-8").read()
    if not old.endswith("\n"):
        old += "\n"
    line = ("[" + now + "] batch60 " + VER + " -> CORRECTED flat-layout distribution of the two "
            "genuine trained models (batch59). Bounded repair of three concrete defects: (F1) "
            "LICENSE_NOTES now truthfully states the checkpoint is a COMPLETE training checkpoint "
            "containing optimizer/scheduler/sampler/RNG entries (packaging fact only; no validated "
            "exact-resume claim; checkpoint bytes unchanged); (F2) SHA256SUMS.txt is root-relative "
            "flat and every listed literal path/hash re-hashes after fresh extraction (no code/ "
            "stripping), all extracted files except SHA256SUMS covered exactly once; (F3) English "
            "and Chinese commands use the root predict.py and a portable python placeholder with no "
            "stale absolute workspace path or old version name. Weights/config/pinned_core/predict/"
            "normalizer/results/requirements BYTE-IDENTICAL to batch59 (all non-revised files proven "
            "equal); zip CRC ok; no __pycache/keys/rawdata. Deliverables: "
            "releases/batch52_first_model/v6.38.1-nabs-final{,-flat.zip} and "
            "releases/batch52_second_model/v6.38.1-njoint-final{,-flat.zip}. Scientific scope "
            "unchanged (public HEK293 retrospective, GRCh37 reference proxy, local 293FT "
            "UNVALIDATED, 100-600 bp empirical, >600 UNSUPPORTED, Nabs predeclared first, pair "
            "FAIL preserved). CUDA runtime NOT replayed: code/config/weights exact-SHA unchanged, "
            "inheriting the batch59 independent runtime proof. Production pointer unchanged; no "
            "upload/publication; no email.")
    open(CHG + ".tmp", "w", encoding="utf-8").write(old + "\n" + line + "\n")
    os.replace(CHG + ".tmp", CHG)
    print(json.dumps({"registry": os.path.relpath(REG, ROOT), "batch_appends": len(reg["batch_appends"]),
                      "appended_at": now}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
