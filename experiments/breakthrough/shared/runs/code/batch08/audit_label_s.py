#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Batch08 label audit: legacy Score_S definition / mask / ties / loss-scale.

Read-only over all canonical source tables/scripts. Writes ONLY to
  data/breakthrough_20261002/batch08
  models/breakthrough_20261002/batch08
  reports/breakthrough_20261002/batch08
No GPU, no training, no edits to historical labels.

Candidate version: v6.9.0-labelaudit.rc1  (label-audit track, no model promotion)
"""
import csv
import hashlib
import json
import math
import statistics
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[4]
DATA_OUT = ROOT / "data" / "breakthrough_20261002" / "batch08"
MODEL_OUT = ROOT / "models" / "breakthrough_20261002" / "batch08"
REPORT_OUT = ROOT / "reports" / "breakthrough_20261002" / "batch08"

CANON = ROOT / "labels" / "build" / "gene_expression_label_table.tsv"
EXPR = ROOT / "labels" / "build" / "gene_expression_table.tsv"
V6_IN = ROOT / "labels" / "build" / "promoter_atlas_v6_input.tsv"
SPLIT_CORE = ROOT / "labels" / "releases" / "promoter_split_core.tsv"
GENE_SPLIT = ROOT / "labels" / "releases" / "gene_split_manifest.tsv"
SRC_BUILDER = ROOT / "runs" / "scripts" / "build_candidate_e_labels.py"
SRC_TRAIN = ROOT / "runs" / "scripts" / "train_promoter_atlas_v6.py"
SRC_V6INPUT = ROOT / "runs" / "scripts" / "build_v6_training_input.py"
SPEC_DEF = ROOT / "labels" / "spec" / "label_definition.md"
SPEC_SCHEMA = ROOT / "labels" / "spec" / "label_schema.tsv"

DETECTABLE_TPM = 1.0
B_MIN = 0.30
MIN_DET_TISSUES = 3
CANDIDATE_VERSION = "v6.9.0-labelaudit.rc1"
TRACK = "label_audit_S"

def sha256(p: Path) -> str:
    with p.open("rb") as fh:
        return hashlib.file_digest(fh, "sha256").hexdigest()

def fhash(p: Path) -> dict:
    st = p.stat()
    return {"path": str(p.resolve()), "bytes": st.st_size, "sha256": sha256(p),
            "mtime_ns": st.st_mtime_ns}

def base_gene_id(v: str) -> str:
    return (v or "").split(".")[0]

def median(xs):
    return statistics.median(xs) if xs else 0.0

def mad(xs):
    if not xs:
        return 0.0
    m = median(xs)
    return median([abs(x - m) for x in xs])

def collapse_duplicate(values):
    nz = [x for x in values if x > 0.0]
    zc = sum(1 for x in values if x == 0.0)
    if nz and zc > 0:
        return max(nz), "prefer_nonzero_max"
    return statistics.fmean(values), "mean"

def rankdata_average(x):
    x = np.asarray(x, dtype=np.float64)
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(len(x), dtype=np.float64)
    i = 0
    n = len(x)
    while i < n:
        j = i + 1
        while j < n and x[order[j]] == x[order[i]]:
            j += 1
        ranks[order[i:j]] = 0.5 * (i + j - 1) + 1.0
        i = j
    return ranks

def spearman_np(x, y):
    if len(x) < 2:
        return float("nan")
    rx = rankdata_average(np.asarray(x, dtype=np.float64))
    ry = rankdata_average(np.asarray(y, dtype=np.float64))
    rx -= rx.mean(); ry -= ry.mean()
    d = math.sqrt((rx ** 2).sum() * (ry ** 2).sum())
    return float((rx * ry).sum() / d) if d else float("nan")

def main():
    t0 = time.time()
    for d in (DATA_OUT, MODEL_OUT, REPORT_OUT):
        d.mkdir(parents=True, exist_ok=True)

    sources = {p.name: fhash(p) for p in
               [CANON, EXPR, V6_IN, SPLIT_CORE, GENE_SPLIT, SRC_BUILDER, SRC_TRAIN,
                SRC_V6INPUT, SPEC_DEF, SPEC_SCHEMA]}

    g = pd.read_csv(CANON, sep="\t", low_memory=False)
    N = len(g)
    defined = g["Score_S"].notna()
    M = int(defined.sum())
    a = (N - M) / (N - 1)
    b = (M - 1) / (N - 1)

    print("recomputing full-precision S_raw from", EXPR.name, flush=True)
    expr = pd.read_csv(EXPR, sep="\t", usecols=["gene_id", "tissue", "mean_tpm"],
                       dtype={"gene_id": "str", "tissue": "str", "mean_tpm": np.float64},
                       low_memory=False)
    gene_tissue = defaultdict(lambda: defaultdict(list))
    for gene_id, tissue, mt in zip(expr["gene_id"], expr["tissue"], expr["mean_tpm"]):
        gb = base_gene_id(gene_id)
        gene_tissue[gb][tissue].append(float(mt))
    del expr

    full_s = {}
    for gene in sorted(gene_tissue):
        tm = gene_tissue[gene]
        collapsed = []
        for tissue in sorted(tm):
            vals = tm[tissue]
            collapsed.append(collapse_duplicate(vals)[0] if len(vals) > 1 else vals[0])
        tc = len(collapsed)
        detectable = [x for x in collapsed if x >= DETECTABLE_TPM]
        dtc = len(detectable)
        b_raw = dtc / tc if tc else 0.0
        if b_raw >= B_MIN and dtc >= MIN_DET_TISSUES:
            full_s[gene] = -mad([math.log2(1.0 + x) for x in detectable])
    print("full-precision S_raw genes: %d (canonical defined: %d)" % (len(full_s), M), flush=True)

    g["S_raw_full"] = g["gene_id"].map(full_s)
    g["S_raw_6dec"] = g["S_raw"]
    g["S_raw_6dec_err"] = np.where(g["S_raw_full"].notna() & g["S_raw_6dec"].notna(),
                                   (g["S_raw_full"].astype(float) - g["S_raw_6dec"].astype(float)).abs(),
                                   np.nan)

    order_s = np.argsort(np.where(g["S_raw_full"].notna(),
                                  g["S_raw_full"].astype(float).to_numpy(),
                                  -np.inf), kind="mergesort")
    legacy_rank = np.empty(N, dtype=np.float64)
    legacy_rank[order_s] = np.arange(N) / (N - 1)
    g["Score_S_recomputed"] = legacy_rank
    g.loc[g["S_raw_full"].isna(), "Score_S_recomputed"] = np.nan
    score_s_err = (g.loc[defined, "Score_S"].astype(float) -
                   g.loc[defined, "Score_S_recomputed"].astype(float)).abs()
    max_score_s_err = float(score_s_err.max())

    g["S_affine"] = np.where(defined, (g["Score_S"].astype(float) - a) / b, np.nan)
    g["S_affine_exact"] = np.where(defined,
                                   (g["Score_S_recomputed"].astype(float) - a) / b, np.nan)
    g["S_affine_inverse_check"] = np.where(defined,
                                           a + b * g["S_affine"].astype(float), np.nan)

    s_full_defined = g.loc[defined, "S_raw_full"].astype(float).to_numpy()
    avg_rank = rankdata_average(s_full_defined)
    quantile = (avg_rank - 1.0) / (M - 1.0)
    g["S_defined_rank_int"] = np.nan
    g["S_defined_rank"] = np.nan
    g.loc[defined, "S_defined_rank_int"] = avg_rank
    g.loc[defined, "S_defined_rank"] = quantile

    tie_id_full = {}
    for gene, val in full_s.items():
        tie_id_full.setdefault(val, []).append(gene)
    full_tie_groups = {k: v for k, v in tie_id_full.items() if len(v) > 1}
    s6 = g.loc[defined, "S_raw_6dec"].astype(float).to_numpy()
    genes_def = g.loc[defined, "gene_id"].to_numpy()
    tie6 = defaultdict(list)
    for gene, val in zip(genes_def, s6):
        tie6[val].append(gene)
    tie6_groups = {k: v for k, v in tie6.items() if len(v) > 1}

    tie_rows = []
    for val, genes in sorted(full_tie_groups.items()):
        tie_rows.append({"precision": "full", "S_raw_value": repr(val),
                         "group_size": len(genes), "genes": "|".join(sorted(genes))})
    for val, genes in sorted(tie6_groups.items()):
        tie_rows.append({"precision": "6dec", "S_raw_value": "%.6f" % val,
                         "group_size": len(genes), "genes": "|".join(sorted(genes))})

    true_tie_genes = set()
    for genes in full_tie_groups.values():
        true_tie_genes.update(genes)
    n_full_tie_groups = len(full_tie_groups)
    n_full_tie_genes = len(true_tie_genes)
    n_6dec_tie_groups = len(tie6_groups)
    n_6dec_tie_genes = sum(len(v) for v in tie6_groups.values())
    rounding_groups = 0
    rounding_genes = 0
    for val, genes in tie6_groups.items():
        fp_vals = {full_s[gene] for gene in genes}
        if len(fp_vals) > 1:
            rounding_groups += 1
            rounding_genes += len(genes)
    true6_groups = n_6dec_tie_groups - rounding_groups

    v6 = pd.read_csv(V6_IN, sep="\t", low_memory=False)
    hum = v6[v6["species"] == "homo_sapiens"].copy()
    n_human = len(hum)
    n_labeled = int((hum["label_mask"] == 1).sum())
    hum["s_defined"] = (hum["label_mask"] == 1) & hum["Score_S"].notna()
    n_s_defined = int(hum["s_defined"].sum())
    n_s_missing_labeled = n_labeled - n_s_defined

    gsplit = pd.read_csv(GENE_SPLIT, sep="\t", low_memory=False)
    n_genes_split = len(gsplit)
    n_genes_split_s = int(gsplit["Score_S"].notna().sum())

    n_labeled_records = n_labeled
    n_unlabeled_records = n_human - n_labeled
    n_unique_labeled_genes = int(hum.loc[hum["label_mask"] == 1, "gene_id"].nunique())
    n_unique_s_genes = int(hum.loc[hum["s_defined"], "gene_id"].nunique())

    split_rows = []
    for split in ["train", "val", "test"]:
        sub = hum[hum["split"] == split]
        split_rows.append({
            "split": split,
            "records": int(len(sub)),
            "labeled_E": int((sub["label_mask"] == 1).sum()),
            "unlabeled": int((sub["label_mask"] != 1).sum()),
            "s_defined": int(sub["s_defined"].sum()),
            "s_missing_labeled": int(((sub["label_mask"] == 1) & ~sub["s_defined"]).sum()),
            "unique_genes_total": int(sub["gene_id"].nunique()),
            "unique_genes_S": int(sub.loc[sub["s_defined"], "gene_id"].nunique()),
        })

    gene_meta = g[["gene_id", "Score_S", "S_affine", "S_defined_rank"]].rename(
        columns={"Score_S": "legacy_S", "S_affine": "affine_S"})
    hum_j = hum[["gene_id", "split", "s_defined"]].merge(gene_meta, on="gene_id", how="left")
    mask_stats = []
    for split in ["train", "val", "test"]:
        sub = hum_j[(hum_j["split"] == split) & (hum_j["s_defined"])].drop_duplicates(subset="gene_id")
        for col in ["legacy_S", "affine_S", "S_defined_rank"]:
            v = sub[col].astype(float).to_numpy()
            mask_stats.append({
                "split": split, "label": col, "n_genes": len(v),
                "min": float(v.min()), "max": float(v.max()),
                "range": float(v.max() - v.min()), "mean": float(v.mean()),
                "sd": float(v.std(ddof=0)), "variance": float(v.var(ddof=0)),
            })

    es = g.loc[defined, ["gene_id", "Score_E", "Score_S", "S_affine", "S_defined_rank",
                         "B_raw", "Score_B", "Score_HKG"]].copy()
    es["Score_E"] = es["Score_E"].astype(float)
    es["Score_S"] = es["Score_S"].astype(float)
    es["S_affine"] = es["S_affine"].astype(float)
    es["S_defined_rank"] = es["S_defined_rank"].astype(float)
    es["ES_legacy"] = es["Score_E"] * es["Score_S"]
    es["ES_affine"] = es["Score_E"] * es["S_affine"]
    es["ES_rank"] = es["Score_E"] * es["S_defined_rank"]
    es["HKG_legacy"] = es["Score_HKG"].astype(float)
    es["HKG_affine"] = (es["Score_E"] * es["S_affine"] * es["Score_B"]).clip(lower=0.0) ** (1.0 / 3.0)

    sp_es_legacy_vs_affine = spearman_np(es["ES_legacy"], es["ES_affine"])
    sp_es_legacy_vs_rank = spearman_np(es["ES_legacy"], es["ES_rank"])
    topk = 1000
    tl = set(es.nlargest(topk, "ES_legacy")["gene_id"])
    ta = set(es.nlargest(topk, "ES_affine")["gene_id"])
    tr = set(es.nlargest(topk, "ES_rank")["gene_id"])
    overlap_la = len(tl & ta)
    overlap_lr = len(tl & tr)

    b_raw_vals = g["B_raw"].astype(float).to_numpy()
    b_counts = Counter(np.round(b_raw_vals, 12))
    n_b_distinct = len(b_counts)
    b_tie_genes = sum(c - 1 for c in b_counts.values() if c > 1)
    b_tie_groups = sum(1 for c in b_counts.values() if c > 1)
    max_b_group = max(b_counts.values())

    out_cols = ["gene_id", "tissue_count", "detectable_tissue_count", "B_gate_pass",
                "S_defined_main", "B_raw", "E_raw", "S_raw_6dec", "S_raw_full",
                "S_raw_6dec_err", "Score_E", "Score_S", "Score_B", "Score_HKG",
                "S_affine", "S_affine_exact", "S_affine_inverse_check",
                "S_defined_rank_int", "S_defined_rank", "S_eligibility_reason"]
    reasons = []
    for _, row in g.iterrows():
        if row["S_defined_main"] == 1:
            reasons.append("")
        elif row["B_gate_pass"] == 0:
            reasons.append("B_raw_below_0.30")
        elif row["detectable_tissue_count"] < MIN_DET_TISSUES:
            reasons.append("detectable_tissues_below_3")
        else:
            reasons.append("S_undefined_other")
    g["S_eligibility_reason"] = reasons
    g[out_cols].to_csv(DATA_OUT / "label_audit_all_genes.tsv", sep="\t", index=False,
                       float_format="%.10g")

    pd.DataFrame(tie_rows).to_csv(DATA_OUT / "tie_audit.tsv", sep="\t", index=False)
    pd.DataFrame(split_rows).to_csv(DATA_OUT / "pipeline_mask_counts.tsv", sep="\t", index=False)
    pd.DataFrame(mask_stats).to_csv(DATA_OUT / "mask_loss_stats.tsv", sep="\t", index=False)

    transform = {
        "candidate_version": CANDIDATE_VERSION,
        "track": TRACK,
        "scope": "S label audit only; no model training/promotion; no biological validation claim",
        "source_S_definition": {
            "E_raw": "mean over 54 tissues of log2(1+collapsed_mean_tpm)",
            "B_raw": "count(collapsed_mean_tpm >= 1.0) / 54",
            "S_raw": "-MAD(log2(1+collapsed_mean_tpm)) over detectable tissues (TPM>=1) only",
            "S_eligibility": "B_raw >= 0.30 AND detectable_tissue_count >= 3",
            "detectable_redundancy_note": "with 54 tissues, B_raw>=0.30 implies detectable>=17, so the detectable>=3 clause is redundant in practice",
            "Score_S": "stable ascending ordinal rank / (N-1) over ALL N genes; undefined S_raw entered as -inf then Score_S cleared",
            "tie_convention": "stable sort (equal raw values ordered by gene_id), NOT average-rank",
        },
        "affine_transform": {
            "N_canonical_rows": N,
            "M_defined": M,
            "a": a, "b": b,
            "formula": "S_affine = (legacy Score_S - a) / b",
            "inverse": "legacy Score_S = a + b * S_affine",
            "a_formula": "(N-M)/(N-1)",
            "b_formula": "(M-1)/(N-1)",
            "defined_domain": {
                "legacy_min": float(g.loc[defined, "Score_S"].astype(float).min()),
                "legacy_max": float(g.loc[defined, "Score_S"].astype(float).max()),
                "affine_min": float(g.loc[defined, "S_affine"].min()),
                "affine_max": float(g.loc[defined, "S_affine"].max()),
            },
            "endpoint_tolerance_note": "canonical Score_S stored to 6 decimals; stored min defined = 0.678811 vs exact a = 0.67881058538322",
            "missing_preserved": True,
            "ordering_preserved": True,
        },
        "S_defined_rank": {
            "formula": "average-rank ties on full-precision S_raw among M defined genes; quantile=(rank-1)/(M-1)",
            "M": M,
            "ties": "explicit average-rank (differs from legacy stable gene_id-order tie-break)",
            "degenerate_note": "M<=1 unsupported (division by zero); here M=18947>1",
        },
        "tie_audit_summary": {
            "full_precision_tie_groups": n_full_tie_groups,
            "full_precision_tie_genes": n_full_tie_genes,
            "sixdec_tie_groups": n_6dec_tie_groups,
            "sixdec_tie_genes": n_6dec_tie_genes,
            "sixdec_rounding_only_groups": rounding_groups,
            "sixdec_rounding_only_genes": rounding_genes,
            "sixdec_true_tie_groups": true6_groups,
            "unique_sixdec_values_among_defined": int(g.loc[defined, "S_raw_6dec"].nunique()),
        },
        "score_s_recomputed_max_abs_err": max_score_s_err,
        "pipeline_counts": {
            "canonical_genes": N,
            "canonical_S_defined": M,
            "human_records_v6_input": n_human,
            "human_labeled_E_records": n_labeled,
            "human_unlabeled_records": n_unlabeled_records,
            "human_unique_labeled_genes": n_unique_labeled_genes,
            "human_S_defined_records": n_s_defined,
            "human_S_missing_among_labeled_records": n_s_missing_labeled,
            "human_unique_S_genes": n_unique_s_genes,
            "gene_split_manifest_genes": n_genes_split,
            "gene_split_manifest_S_defined": n_genes_split_s,
            "gene_split_manifest_note": "separate 56161-gene reference manifest (symbol-annotated); not the v6-input record universe",
        },
        "es_sensitivity": {
            "n_S_defined_for_ES": int(len(es)),
            "topk": topk,
            "spearman_ES_legacy_vs_affine": sp_es_legacy_vs_affine,
            "spearman_ES_legacy_vs_rank": sp_es_legacy_vs_rank,
            "topk_overlap_legacy_vs_affine": overlap_la,
            "topk_overlap_legacy_vs_rank": overlap_lr,
            "note": "S rescaling changes E*S objective weighting; NOT biological validation",
        },
        "B_tie_artifact": {
            "B_raw_distinct_values": n_b_distinct,
            "B_tie_groups": b_tie_groups,
            "B_tied_genes_beyond_first": b_tie_genes,
            "max_B_group_size": max_b_group,
            "note": "Score_B uses stable gene_id-order tie-break for equal B_raw; left unchanged this batch",
        },
    }
    (MODEL_OUT / "transform_metadata.json").write_text(
        json.dumps(transform, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")

    tm_rows = [
        ("candidate_version", CANDIDATE_VERSION),
        ("track", TRACK),
        ("N_canonical_rows", N), ("M_defined", M),
        ("a", a), ("b", b),
        ("affine_formula", "S_affine=(Score_S-a)/b"),
        ("inverse_formula", "Score_S=a+b*S_affine"),
        ("S_raw_formula", "-MAD(log2(1+TPM)) over detectable tissues"),
        ("S_eligibility", "B_raw>=0.30 AND detectable>=3"),
        ("tie_convention", "legacy=stable gene_id order; S_defined_rank=average-rank"),
        ("max_score_s_recompute_err", max_score_s_err),
        ("full_precision_tie_groups", n_full_tie_groups),
        ("sixdec_tie_groups", n_6dec_tie_groups),
        ("sixdec_rounding_only_groups", rounding_groups),
        ("human_S_defined_records", n_s_defined),
    ]
    with (MODEL_OUT / "transform_metadata.tsv").open("w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh, delimiter="\t")
        w.writerow(["key", "value"])
        w.writerows(tm_rows)

    with (REPORT_OUT / "sha_registry.tsv").open("w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh, delimiter="\t")
        w.writerow(["source", "path", "bytes", "sha256", "mtime_ns", "role"])
        for name, info in sources.items():
            w.writerow([name, info["path"], info["bytes"], info["sha256"], info["mtime_ns"], "read-only-source"])

    summary = {
        "candidate_version": CANDIDATE_VERSION,
        "N": N, "M": M, "a": a, "b": b,
        "score_s_defined_range": [float(g.loc[defined, "Score_S"].astype(float).min()),
                                  float(g.loc[defined, "Score_S"].astype(float).max())],
        "s_raw_defined_range": [float(g.loc[defined, "S_raw_6dec"].astype(float).min()),
                                float(g.loc[defined, "S_raw_6dec"].astype(float).max())],
        "affine_range": [float(g.loc[defined, "S_affine"].min()), float(g.loc[defined, "S_affine"].max())],
        "max_score_s_recompute_err": max_score_s_err,
        "s_raw_max_6dec_err": float(g["S_raw_6dec_err"].max()),
        "tie_audit_summary": transform["tie_audit_summary"],
        "pipeline_counts": transform["pipeline_counts"],
        "es_sensitivity": transform["es_sensitivity"],
        "B_tie_artifact": transform["B_tie_artifact"],
        "split_rows": split_rows,
        "mask_stats": mask_stats,
        "elapsed_s": round(time.time() - t0, 2),
    }
    (REPORT_OUT / "audit_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print("DONE elapsed", round(time.time() - t0, 2), "s", flush=True)

if __name__ == "__main__":
    main()
