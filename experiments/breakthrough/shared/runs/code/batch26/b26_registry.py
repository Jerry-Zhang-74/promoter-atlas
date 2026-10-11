# -*- coding: utf-8 -*-
"""batch26: append-only registry + CHANGELOG entries (batch25 dataset freeze gap
closure + batch26 model record).  Never rewrites prior entries."""
from __future__ import annotations
import json, os, sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import b26_common as C

TOP = C.ROOT / "reports" / "breakthrough_20261002"
REG = TOP / "version_registry.json"
CHG = TOP / "CHANGELOG.txt"
LOCK = TOP / ".version_registry.lock"


class RegLock:
    def __enter__(self):
        for _ in range(120):
            try:
                self.fd = os.open(str(LOCK), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                os.write(self.fd, str(os.getpid()).encode())
                return self
            except FileExistsError:
                time.sleep(0.25)
        raise SystemExit("registry lock busy")

    def __exit__(self, *a):
        try:
            os.close(self.fd)
        finally:
            try:
                os.unlink(LOCK)
            except FileNotFoundError:
                pass


def shas():
    A = C.RUN26 / "attempts"
    return {
        "d_str_variant_batch25": C.sha256_file(C.ROOT / "data/breakthrough_20261002/batch25/str_variant_variants.tsv"),
        "b26_rows": C.sha256_file(C.DATA26 / "str_variant_rows.tsv"),
        "b26_encoded": C.sha256_file(C.DATA26 / "encoded.npz"),
        "b26_split": C.sha256_file(C.DATA26 / "split_manifest.tsv"),
        "A0_weights": C.sha256_file(A / "A0_ridgeC" / "ridge_weights.npy"),
        "A1_ckpt": C.sha256_file(A / "A1_cnn_true_seed20261026" / "ckpt.pt"),
        "A2_ckpt": C.sha256_file(A / "A2_cnn_true_seed20261027" / "ckpt.pt"),
        "A3_ckpt": C.sha256_file(A / "A3_cnn_shuffle_seed20261026" / "ckpt.pt"),
    }


def build_records():
    tm = C.read_json(C.REPORT26 / "test_metrics.json")
    tmt = tm["test_metrics"]
    s = shas()
    b25 = {
        "record_type": "dataset_freeze",
        "batch": "batch25",
        "version": "v6.18.1-external-evidence.rc1",
        "appended_by": "batch26 worker (parent-authorized append-only gap closure)",
        "appended_at": C.now_local(),
        "dataset": {
            "name": "GSE306816 hSTR1 HEK293T STR-variant MPRA frozen table",
            "path": "data/breakthrough_20261002/batch25/str_variant_variants.tsv",
            "sha256": s["d_str_variant_batch25"],
            "variants": 57674, "independent_groups_loci": 19742,
            "cell": "HEK293T", "biological_replicates": 3,
            "endpoint": "mean log2(cDNA/gDNA) per variant",
            "freeze_evidence": "reports/breakthrough_20261002/batch25/external_eval_freeze.json",
            "zero_model_fits_at_freeze": True,
            "task_scope": ("SEPARATE STR-variant task; NOT an autonomous-promoter cohort; H1/R1 were correctly "
                           "not applied (different assay context)."),
            "gap_closed": "batch25 produced no version_registry/CHANGELOG entry (its own review noted this under compliance item 2); this entry closes that gap without altering batch25 artifacts.",
        },
    }
    b26 = {
        "record_type": "model_candidate",
        "batch": "batch26",
        "candidate_version": "v6.19.0-strvariant.rc1",
        "internal_id": "b26_legnet_strvariant",
        "track": "task-native STR+flank reporter-activity predictor (HEK293T hSTR1 minP-GFP context)",
        "status": "experimental candidate; NOT promoted; production pointer unchanged (v6.3 main / v6.4 cellline pilot)",
        "parent_decision": "2026-10-03 17:43 Asia/Shanghai",
        "architecture": "batch02 trackB_grouped_cv LegNetInspired(num_targets=1, ch=64), read-only import, trained from scratch, no K562 weights",
        "n_parameters_trainable": 110145,
        "n_state_dict_entries": 111306,
        "input": "one-hot ACGT full native variable insert (108-168 nt), no padding, equal-length minibatches",
        "data_version": "d_str_variant_batch25",
        "rows_sha256": s["b26_rows"], "encoded_sha256": s["b26_encoded"],
        "split_version": "s_batch26_locus_holdout", "split_sha256": s["b26_split"],
        "split_rule": "groupid = min(sequence SHA)+sorted locus ids; hash('batch26_seed20261026|groupid')<0.70 train / <0.85 val / else test",
        "split_sizes": {"train_groups": 13228, "val_groups": 2841, "test_groups": 2840,
                        "train_rows": 40208, "val_rows": 8666, "test_rows": 8800},
        "artifacts": {
            "A0_ridgeC_weights_sha256": s["A0_weights"],
            "A1_cnn_true_seed20261026_ckpt_sha256": s["A1_ckpt"],
            "A2_cnn_true_seed20261027_ckpt_sha256": s["A2_ckpt"],
            "A3_cnn_shuffle_seed20261026_ckpt_sha256": s["A3_ckpt"],
            "selection_manifest": "reports/breakthrough_20261002/batch26/selection_manifest.json",
            "test_metrics": "reports/breakthrough_20261002/batch26/test_metrics.json",
            "model_card": "models/cards/promoter_atlas_v6_19_0_strvariant_rc1_card.txt",
        },
        "selected_primary": tm["selected_primary"],
        "test_metrics": {
            "C_ridge_spearman": tmt["A0_ridgeC"]["spearman"], "C_ridge_gbMAE": tmt["A0_ridgeC"]["group_balanced_mae"],
            "A1_spearman": tmt["A1_cnn_true_seed20261026"]["spearman"], "A1_gbMAE": tmt["A1_cnn_true_seed20261026"]["group_balanced_mae"],
            "A2_spearman": tmt["A2_cnn_true_seed20261027"]["spearman"], "A2_gbMAE": tmt["A2_cnn_true_seed20261027"]["group_balanced_mae"],
            "A3_shuffle_spearman": None, "A3_shuffle_note": "undefined (constant prediction)",
        },
        "predeclared_criteria": tm["predeclared_criteria"],
        "criteria_all_met": tm["criteria_all_met"],
        "variant_direction_support_met": tm["variant_direction_support_met"],
        "fit_budget": {"declared_max": 4, "observed_executions": 5, "compliance": "FAIL",
                       "evidence": "reports/breakthrough_20261002/batch26/fit_accounting_disclosure.json"},
        "scope_limitations": ("internal retrospective locus-holdout, NOT external/prospective validation; "
                              "minP-GFP fragment context, not autonomous promoter activity; not calibrated GFP; "
                              "residual near-homology uncertainty (locus-heldout, not strict family isolation)."),
        "decision": "reviewed candidate for a FUTURE engineering/transfer hypothesis only; no promotion, no transfer claim",
    }
    return b25, b26


def main():
    b25, b26 = build_records()
    with RegLock():
        reg = json.loads(REG.read_text(encoding="utf-8"))
        before = C.sha256_file(REG)
        assert "d_str_variant_batch25" not in reg.get("data_versions", {}), "already appended?"
        ap = reg.setdefault("batch_appends", [])
        if not any(a.get("batch") == "batch25" and a.get("record_type") == "dataset_freeze" for a in ap):
            ap.append(b25)
        if not any(a.get("batch") == "batch26" for a in ap):
            ap.append(b26)
        dv = reg.setdefault("data_versions", {})
        dv["d_str_variant_batch25"] = {
            "path": "data/breakthrough_20261002/batch25/str_variant_variants.tsv",
            "sha256": b25["dataset"]["sha256"],
            "note": "batch25 STR-variant freeze; read-only; appended by batch26 (gap closure).",
        }
        cds = reg.setdefault("candidates", [])
        if not any(c.get("candidate_version") == "v6.19.0-strvariant.rc1" for c in cds):
            cds.append({
                "candidate_version": "v6.19.0-strvariant.rc1",
                "internal_id": "b26_legnet_strvariant",
                "track": b26["track"], "status": b26["status"],
                "data_version": "d_str_variant_batch25",
                "split_version": "s_batch26_locus_holdout",
                "run_id": "batch26",
                "seeds": "true 20261026/20261027; shuffle control 20261026 (perm seed 20261028)",
                "checkpoint_sha256": {
                    "A0_ridgeC": b26["artifacts"]["A0_ridgeC_weights_sha256"],
                    "A1_cnn_true_seed20261026": b26["artifacts"]["A1_cnn_true_seed20261026_ckpt_sha256"],
                    "A2_cnn_true_seed20261027": b26["artifacts"]["A2_cnn_true_seed20261027_ckpt_sha256"],
                    "A3_cnn_shuffle_seed20261026": b26["artifacts"]["A3_cnn_shuffle_seed20261026_ckpt_sha256"],
                },
                "metrics": b26["test_metrics"],
                "predeclared_criteria": b26["predeclared_criteria"],
                "criteria_all_met": b26["criteria_all_met"],
                "fit_budget_compliance": "FAIL (5 observed executions vs max 4)",
                "creation_time": C.now_local(),
                "decision": b26["decision"],
            })
        reg["last_appended_at"] = C.now_local()
        reg["last_appended_by"] = "batch26 worker (append-only; all prior entries preserved)"
        reg["registry_sha256_before_batch26_append"] = before
        C.atomic_write_json(REG, reg)
        after = C.sha256_file(REG)
        n_prev, n_new = len(reg["batch_appends"]) - 2, len(reg["batch_appends"])
    lines = [
        "",
        "[%s] batch25 DATASET FREEZE appended to version_registry (gap closure): %s sha256 %s; 57674 variants / 19742 loci; zero model fits at freeze; separate STR-variant task, H1/R1 not applied. Prior entries preserved." %
        (C.now_local(), b25["dataset"]["path"], b25["dataset"]["sha256"]),
        "[%s] v6.19.0-strvariant.rc1 (b26_legnet_strvariant, task-native STR+flank minP-GFP reporter predictor) -> REVIEWED CANDIDATE (NOT promoted). Internal retrospective locus-holdout test 8800 rows / 2840 groups: C composition-ridge Spearman %.4f gbMAE %.4f; A1 true CNN Spearman %.4f gbMAE %.4f; A2 true CNN Spearman %.4f gbMAE %.4f; shuffled control Spearman UNDEFINED (constant prediction). predeclared criteria all_met=%s; variant-direction support met=%s. FIT BUDGET COMPLIANCE FAIL: 5 observed estimator executions vs declared max 4 (A0 ridge solve crashed then re-ran; see reports/breakthrough_20261002/batch26/fit_accounting_disclosure.json). Production pointer unchanged (v6.3 main / v6.4 cellline pilot)." %
        (C.now_local(), b26["test_metrics"]["C_ridge_spearman"], b26["test_metrics"]["C_ridge_gbMAE"],
         b26["test_metrics"]["A1_spearman"], b26["test_metrics"]["A1_gbMAE"],
         b26["test_metrics"]["A2_spearman"], b26["test_metrics"]["A2_gbMAE"],
         str(b26["criteria_all_met"]), str(b26["variant_direction_support_met"])),
    ]
    with open(CHG, "a", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(lines) + "\n")
        f.flush(); os.fsync(f.fileno())
    print("registry before/after:", before[:16], after[:16])
    print("appends now:", len(reg["batch_appends"]))


if __name__ == "__main__":
    main()
