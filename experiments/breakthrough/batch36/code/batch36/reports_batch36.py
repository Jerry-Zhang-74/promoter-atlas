"""Batch36 Part C: two descriptive coverage figures + status/selfcheck/remaining/proposal/README."""
from __future__ import annotations
import csv, hashlib, json, os, collections, datetime, sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = "E:/promoter_atlas"
OUTD = os.path.join(ROOT, "data/breakthrough_20261002/batch36/mapt")
REPD = os.path.join(ROOT, "reports/breakthrough_20261002/batch36")
FIGD = os.path.join(REPD, "figures")
os.makedirs(FIGD, exist_ok=True)
ARMS = ["SAT_HEK293FT", "SAT_NEURON", "KV_SNV_HEK293FT", "KV_INDEL_HEK293FT",
        "KV_SNV_NEURON", "KV_INDEL_NEURON"]


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


# ---- figure 1: per-arm coverage ----
n = {}; nsig = {}
for arm in ARMS:
    rows = list(csv.DictReader(open(os.path.join(OUTD, "cohort_" + arm + ".tsv"), encoding="utf-8"), delimiter="\t"))
    n[arm] = len(rows)
    nsig[arm] = sum(1 for r in rows if r["adjP_Value"] not in ("", "NA") and float(r["adjP_Value"]) <= 0.05)
fig, ax = plt.subplots(figsize=(9, 4.5))
x = range(len(ARMS))
ax.bar([i - 0.2 for i in x], [n[a] for a in ARMS], width=0.4, label="all variants (retained)")
ax.bar([i + 0.2 for i in x], [nsig[a] for a in ARMS], width=0.4, label="adj.P<=0.05 (flag, not filtered)")
ax.set_xticks(list(x)); ax.set_xticklabels(ARMS, rotation=30, ha="right", fontsize=8)
ax.set_ylabel("variant count"); ax.set_title("Batch36 recovered MAPT MPRA label coverage per arm")
ax.legend(fontsize=8); fig.tight_layout()
f1 = os.path.join(FIGD, "batch36_cohort_coverage.png"); fig.savefig(f1, dpi=130); plt.close(fig)

# ---- figure 2: logFC distribution SAT_HEK vs SAT_NEURON ----
fig, ax = plt.subplots(figsize=(8, 4.2))
for arm, col in [("SAT_HEK293FT", "tab:blue"), ("SAT_NEURON", "tab:orange")]:
    vals = []
    for r in csv.DictReader(open(os.path.join(OUTD, "cohort_" + arm + ".tsv"), encoding="utf-8"), delimiter="\t"):
        try:
            vals.append(float(r["logFC"]))
        except (ValueError, TypeError):
            pass
    ax.hist(vals, bins=60, alpha=0.55, label=arm + " (n=%d)" % len(vals), color=col)
ax.set_xlabel("bcalm logFC (variant vs reference control)"); ax.set_ylabel("count")
ax.set_title("Measured variant effects, MAPT saturation region (same construct, two cell contexts)")
ax.legend(fontsize=8); fig.tight_layout()
f2 = os.path.join(FIGD, "batch36_logfc_by_cell.png"); fig.savefig(f2, dpi=130); plt.close(fig)
print("figures", os.path.basename(f1), os.path.basename(f2))

# ---- remaining work ----
remaining = {
    "batch": "batch36", "version": "v6.25.0-293ft-label-recovery.rc1", "generated_at": now(),
    "resolved_this_batch": [
        "MEASURED per-variant labels recovered (bioRxiv supplement media-1.xlsx, tables s11/s13/s15 etc.) and joined 1:1 to the frozen 36000-record design FASTA.",
        "15+15 constant flanks vs 219 bp genomic variable insert resolved empirically from the author 2000 reference oligos; variant centred at design offset 124.",
        "All arms retained with every row (no adjp/logFC filtering); cell/readout/sequence-handling/eligibility frozen before joining outcomes."],
    "unresolved_or_limited": [
        "Sat arm is a SINGLE 2 kb locus -> no train/val/test split is possible; only a within-locus variant diagnostic, never a new-family autonomous promoter.",
        "The measured quantity is CRE/enhancer reporter activity in a fixed lentiMPRA vector (bcalm logFC vs reference), NOT autonomous promoter activity.",
        "HEK293FT arm = plasmid transfection/activity host; neuron arm = lentiviral transduction; the two arms are separate contexts and are NOT pooled.",
        "Study is a PREPRINT; labels are bcalm estimates (logFC, t, P, adj.P), no per-replicate raw counts or SE column in the tables.",
        "No orientation field is present in the tables; orientation is UNKNOWN (multiple orientations, if any, are already aggregated by bcalm = label definition)."],
    "blocked_external": "None. The supplement was retrieved from the official bioRxiv source; PMC serves an anti-bot proof-of-work challenge which was NOT bypassed (not needed).",
    "not_blocked": "Lack of a novel-family split does NOT block a bounded within-locus diagnostic or the fixed PARM interface.",
}


def write_json(name, obj):
    p = os.path.join(REPD, name)
    json.dump(obj, open(p, "w", encoding="utf-8"), indent=2, default=str)
    print("wrote", name)


# ---- selfcheck / status ----
readiness = list(csv.DictReader(open(os.path.join(REPD, "arm_readiness.tsv"), encoding="utf-8"), delimiter="\t"))
freeze = json.load(open(os.path.join(REPD, "cohort_freeze_audit.json"), encoding="utf-8"))
selfcheck = {
    "batch": "batch36", "version": "v6.25.0-293ft-label-recovery.rc1", "generated_at": now(),
    "zero_fit_acknowledged": True, "zero_model_scoring": True, "zero_gpu_inference": True,
    "cpu_only": True, "downloads_bytes_total": 73418975, "download_cap_bytes": 104857600,
    "targeted_queries_used": 4,
    "targeted_queries": [
        "Europe PMC search core (metadata/suppl availability; 200 OK)",
        "Europe PMC supplementaryFiles zip (too slow, aborted; no data used)",
        "PMC article page + supplement URL (anti-bot POW challenge; NOT bypassed)",
        "bioRxiv official supplement media-1.xlsx (73.4 MB, 200 OK, sha bc6e0585...)"],
    "media_xlsx_sha256": freeze["source"]["xlsx_sha256"],
    "twenty_row_all_pass": freeze["twenty_row_all_pass"],
    "join_all_arms_100pct": all(v["matched"] == v["n_rows"] and v["unmatched"] == 0 for v in freeze["join_audit"].values()),
    "no_many_to_many_expansion": True,
    "utf8": True,
}

proposal = {
    "batch": "batch36", "version": "v6.25.0-293ft-label-recovery.rc1", "generated_at": now(),
    "proposed_next": "batch37 (bounded, PARENT APPROVAL REQUIRED, ZERO fit by default)",
    "title": "Zero-fit frozen variant-effect diagnostic on the recovered MAPT cohort, with predeclared split and endpoints",
    "rationale": "Labels are now recovered and joined. The single 2 kb saturation region cannot be split; the known-variant HEK293FT arm spans ~48 chr17 CRE clusters and CAN support a parent-region/group split.",
    "predeclared_before_any_scoring": {
        "reference_and_delta": "per-position reference oligo (maptprom_chr17_<pos>_referenceseq); delta = measured bcalm logFC already reported by the authors",
        "input_normalization": "uppercase ACGT; strip the verified 15+15 constant flanks; 219 bp variable insert; variant at insert centre",
        "length": "219 bp (deletion alleles 214 bp = 219-5)",
        "baseline": "a single FROZEN published model (no fitting) OR the trivial zero-effect baseline; chosen before scoring",
        "known_exposures": "PromoterAI supplement rows are computed scores on the SAME region (do NOT use as ground truth; declare them as exposed competitors). Pretraining exposure of any external model is UNKNOWN.",
        "fixed_endpoints": "KV multiregion arm: Spearman(pred, measured logFC) under grouped-by-region CV + region-level bootstrap; SAT single locus: within-locus direction/sign agreement ONLY (no generalization claim).",
        "selection": "no architecture search, no per-construct calibration, no test re-use"},
    "do_not": ["random SNV split", "treat the 2000 positions as 2000 families",
               "call these constructs autonomous promoters", "use PromoterAI rows as measured truth",
               "autonomously train or score without approval"],
    "needs_parent_approval": True,
}
write_json("worker_selfcheck.json", selfcheck)
write_json("remaining_work.json", remaining)
write_json("next_batch_proposal.json", proposal)
write_json("status.json", {
    "batch": "batch36", "version": "v6.25.0-293ft-label-recovery.rc1", "phase": "worker",
    "state": "complete", "generated_at": now(),
    "verdict": "RUN_OK_LABELS_RECOVERED", "zero_fit": True, "gpu": False,
    "arms": {r["arm"]: r["verdict"] for r in readiness},
    "notes": "MEASURED labels recovered and frozen; within-locus/no-split constraints recorded; no training."})
print("done reports")

