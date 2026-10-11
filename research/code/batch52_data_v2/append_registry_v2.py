# -*- coding: utf-8 -*-
"""Atomic, append-only version registry + changelog finalization for batch52."""
from __future__ import annotations
import datetime as dt, hashlib, json, os

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
REG = os.path.join(ROOT, "reports", "breakthrough_20261002", "version_registry.json")
CHG = os.path.join(ROOT, "reports", "breakthrough_20261002", "CHANGELOG.txt")
REP = os.path.join(ROOT, "reports", "breakthrough_20261002", "batch52")
MODELS = os.path.join(ROOT, "data", "breakthrough_20261002", "batch52", "corrected", "models")
B52 = os.path.join(ROOT, "runs", "breakthrough_20261002", "batch52")
VERSION = "v6.38.0-full-hek-nested-multitask.rc1"
APPENDER = "batch52 worker"


def J(p):
    return json.load(open(p, encoding="utf-8")) if os.path.exists(p) else None


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def summary_line(final):
    if not final:
        return "held-out TEST metrics not available"
    parts = []
    for arm in ("R0", "Nabs", "Njoint"):
        if arm in final["arms"]:
            ac = final["arms"][arm]["activity"]; pr = final["arms"][arm].get("pair", {})
            parts.append("%s rho=%.4f gMAE=%.4f top10=%.4f pairGroupMAE=%s macroSign=%s" % (
                arm, ac["rho"], ac["group_mae"], ac["top10_precision"],
                ("%.4f" % pr["group_mean_mae"]) if pr else "-",
                ("%.4f" % pr["macro_sign_accuracy"]) if pr else "-"))
    return "; ".join(parts) + " | verdict " + str(final.get("verdict"))


def main():
    now = dt.datetime.now().astimezone().isoformat()
    reg = json.load(open(REG, encoding="utf-8"))
    if any(e.get("batch") == "batch52" for e in reg["batch_appends"]):
        raise SystemExit("batch52 already appended; refusing to double-append")
    final = J(os.path.join(REP, "final_test_metrics.json"))
    binfo = J(os.path.join(ROOT, "data", "breakthrough_20261002", "batch52", "corrected",
                           "cohort_v2", "build_report.json"))
    status = J(os.path.join(MODELS, "coordinator_status.json"))
    ledger = [json.loads(l) for l in open(os.path.join(B52, "exec_gate_ledger.jsonl"), encoding="utf-8") if l.strip()]
    nres = sum(1 for e in ledger if e["kind"] == "reserved")
    ncomp = sum(1 for e in ledger if e["kind"] == "completed")
    b51 = {
        "record_type": "historical_changelog_backfill",
        "batch": "batch51",
        "version": "v6.37.0-hek-reference-nested-cohort.rc1",
        "kind": "MISSING WORKER PHASE CHANGELOG ENTRY, backfilled once by the next worker (batch52) during normal finalization; batch51 report/cohort bytes NOT rewritten.",
        "appended_by": APPENDER + " (backfill of a batch51 omission)",
        "appended_at": now,
        "compliance_fails_retained": [
            "batch51 worker did not append its own changelog",
            "precision/hardcoded fields retained in the b51 report",
            "b51 asserted a Log2RPM formula for the Log2Norm_HEK293 column (unsupported)",
        ],
        "delivered": {
            "source_bed_sha256": "1bbe5db71a5f4a92a281d19567b28ca3751f93d62959e6d61bd41188dd550d96",
            "cohort": "coordinate-only, 500-component pilot cap, 61254 endpoints, 9725 pairs",
            "split_components": {"TRAIN": 349, "DEV": 72, "TEST": 79},
            "gate_failed": "correct_assembly_representation",
            "overall_ready": False,
        },
        "zero_fit": True,
        "production_pointer_unchanged": "v6.3 main / v6.4 cellline pilot",
    }
    return now, reg, final, binfo, status, nres, ncomp, b51


def b52_record(now, binfo, final, status, nres, ncomp):
    return {
        "record_type": "trained_model_batch",
        "batch": "batch52",
        "version": VERSION,
        "kind": "THREE fitted estimators (R0 composition ridge, Nabs random-init CNN, Njoint same CNN + nested-difference loss) on the corrected full HEK293 cohort with author target Log2RPM_HEK293; one final held-out TEST evaluation; NOT promoted.",
        "appended_by": APPENDER,
        "appended_at": now,
        "production_pointer_unchanged": "v6.3 main / v6.4 cellline pilot",
        "target_correction": "Log2Norm_HEK293 SUPERSEDED by author-documented Log2RPM_HEK293 for all models and deltas",
        "source": {
            "bed": "data/breakthrough_20261002/batch51/GSE301246_PARM_Training_Library_v2_MH4_normalized_fold0.bed.gz",
            "sha256": "1bbe5db71a5f4a92a281d19567b28ca3751f93d62959e6d61bd41188dd550d96",
        },
        "cohort": {"dir": "data/breakthrough_20261002/batch52/corrected/cohort_v2",
                   "endpoints": (binfo or {}).get("n_endpoints"),
                   "endpoint_counts": (binfo or {}).get("endpoint_counts"),
                   "group_counts": (binfo or {}).get("group_counts"),
                   "pair_counts": (binfo or {}).get("pair_counts"),
                   "datahash": (binfo or {}).get("datahash")},
        "fit_ledger": {"ledger": "runs/breakthrough_20261002/batch52/exec_gate_ledger.jsonl",
                       "reserved": nres, "completed": ncomp, "max": 4, "remaining": 0},
        "models": {arm: {"selected": (status or {}).get("arms", {}).get(arm, {}).get("selected"),
                         "sha256": sha256_file(os.path.join(MODELS, arm, arm + "_selected.ckpt"))
                         if os.path.exists(os.path.join(MODELS, arm, arm + "_selected.ckpt")) else None}
                   for arm in ("Nabs", "Njoint")},
        "heldout_test": summary_line(final),
        "verdict": (final or {}).get("verdict"),
        "outputs": {
            "report": "reports/breakthrough_20261002/batch52/REPORT.md",
            "metrics": "reports/breakthrough_20261002/batch52/final_test_metrics.json",
            "result_zh": "reports/breakthrough_20261002/batch52/RESULT_CN.txt",
            "figures": "reports/breakthrough_20261002/batch52/figures",
            "card": "models/cards/promoter_atlas_v6_38_0_full_hek_nested_multitask_rc1_card.txt",
            "b51_closure": "reports/breakthrough_20261002/batch52/B51_CLOSURE.txt",
        },
    }


def changelog_lines(now, binfo, final):
    a = ("[" + now + "] batch51 v6.37.0-hek-reference-nested-cohort.rc1 -> HISTORICAL CHANGELOG BACKFILL "
         "(recorded late by the batch52 worker; the batch51 worker omitted its own entry; b51 bytes preserved). "
         "Coordinate-only cohort from authentic GSE301246 v2_MH4 fold0 HEK293 (sha256 1bbe5db7...d96), "
         "500-component pilot cap, 61254 endpoints, 9725 adjacent nested pairs, split components TRAIN 349 / "
         "DEV 72 / TEST 79. Sequence representation BLOCKED (author hg19 vs local GRCh38): OVERALL_READY=False, "
         "gate_failed=correct_assembly_representation. Retained compliance FAILs: missing changelog; "
         "precision/hardcoded fields; unsupported Log2RPM formula claim for the Log2Norm_HEK293 column. "
         "Zero fit/GPU. Production pointer unchanged (v6.3 main / v6.4 cellline pilot).")
    b = ("[" + now + "] batch52 " + VERSION + " -> THREE fitted estimators delivered on the corrected full "
         "HEK293 cohort. TARGET CORRECTION applied: author-documented Log2RPM_HEK293 used directly for all "
         "models and paired deltas (Log2Norm_HEK293 superseded, never a fit target). Cohort rebuilt from the "
         "same authentic source after the label-free split was frozen: 761655 endpoints, counts "
         + json.dumps((binfo or {}).get("endpoint_counts")) + ", union groups "
         + json.dumps((binfo or {}).get("group_counts")) + ", nested pairs "
         + json.dumps((binfo or {}).get("pair_counts")) + "; Ensembl GRCh37 (= author hg19 contract) whole-genome "
         "reference sha256 0a43b56d...47865, reference PROXY for donor HG02601 (not a donor clone; no local "
         "293FT claim). Arms: R0 composition ridge (342 features, alpha 100); Nabs random-init CNN absolute "
         "loss; Njoint the SAME CNN init/arch/optimizer/seed + 0.5 x group-weighted SmoothL1 nested difference. "
         "All eligible TRAIN rows of complete fold0, 6-epoch cap chosen once after Nabs epoch 0, batch 32, one "
         "RTX5060 cuda:0, CPU 4 threads. exec_gate MAX4 ledger fully consumed 4/4 (1 historical ungated "
         "wrong-target R0 + corrected R0 + Nabs + Njoint); no reserve, no further fits launched. Single final "
         "held-out TEST evaluation with frozen DEV-selected weights: " + summary_line(final)
         + ". Investigated nested-multitask target: "
         + json.dumps((final or {}).get("investigation_target", {}), ensure_ascii=False)
         + ". NO production promotion. Production pointer unchanged (v6.3 main / v6.4 cellline pilot).")
    return a, b


def run():
    now, reg, final, binfo, status, nres, ncomp, b51 = main()
    reg["batch_appends"].append(b51)
    reg["batch_appends"].append(b52_record(now, binfo, final, status, nres, ncomp))
    reg["last_appended_at"] = now
    reg["last_appended_by"] = APPENDER
    tmp = REG + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(reg, f, ensure_ascii=False, indent=2)
    os.replace(tmp, REG)
    with open(CHG, "r", encoding="utf-8") as f:
        old = f.read()
    if not old.endswith("\n"):
        old += "\n"
    a, b = changelog_lines(now, binfo, final)
    tmpc = CHG + ".tmp"
    with open(tmpc, "w", encoding="utf-8") as f:
        f.write(old + "\n" + a + "\n\n" + b + "\n")
    os.replace(tmpc, CHG)
    print(json.dumps({"registry": REG, "batch_appends": len(reg["batch_appends"]),
                      "changelog": CHG, "appended_at": now}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    run()
