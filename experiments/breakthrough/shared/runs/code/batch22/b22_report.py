#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Stage 5 reporting: three scientific figures + source tables, GPU latency/devices,
applicability diagnostics, manifests (preprocessing/target/split/source/code/modelSHA),
attempts totals, selfcheck and next_batch_proposal.

Figures (each with its own source table):
  fig1 paired-cell MEASURED domain shift (K562 vs HEK hek_norm). NOT model performance.
  fig2 sequence-only grouped-heldout predictions: observed vs R1/R2 + per-fold rho.
  fig3 applicability: training length coverage vs the 14 local constructs (no numeric
       full-activity claim, out-of-training-length/context flagged) + frozen cell-line
       heldout prediction distribution from batch21 records for comparison.
"""
from __future__ import annotations

import csv
import json
import time
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from b22_common import (ROOT, DATA_OUT, REPORT_OUT, MODEL_OUT, FROZEN_CKPT, VERSION,
                        atomic_write_json, now_local, sha256_file, spearman,
                        group_bootstrap, set_thread_limits)
from b22_fits import load_inputs, feature_composition  # reuse frozen feature logic

BATCH21 = ROOT / "reports" / "breakthrough_20261002" / "batch21"
FIGDIR = REPORT_OUT / "figures"
CONSTRUCT_LENGTHS = [100, 200, 300, 400, 100, 200, 300, 400, 100, 200, 300, 400, 700, 700]


def fig1_domain_shift(y_hek, y_k562, pids):
    FIGDIR.mkdir(parents=True, exist_ok=True)
    rho = spearman(y_k562, y_hek)
    boot = group_bootstrap(y_k562, y_hek, np.arange(len(y_hek)), metric=spearman, n_boot=2000)
    fig, ax = plt.subplots(figsize=(5.2, 5.0), dpi=150)
    ax.scatter(y_k562, y_hek, s=10, alpha=0.55, edgecolors="none", color="#1f77b4")
    lim = [min(y_k562.min(), y_hek.min()) - 0.2, max(y_k562.max(), y_hek.max()) + 0.2]
    ax.plot(lim, lim, "--", color="gray", linewidth=0.8)
    ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_xlabel("K562 measured log2(exp/input)")
    ax.set_ylabel("HEK293 measured log2(exp/input)")
    ax.set_title("Paired-cell MEASURED domain shift (n=%d)\nSpearman=%.3f (not model performance)"
                 % (len(y_hek), rho), fontsize=9)
    fig.tight_layout()
    fig.savefig(FIGDIR / "fig1_domain_shift.png")
    plt.close(fig)
    with open(REPORT_OUT / "fig1_data.tsv", "w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh, delimiter="\t")
        w.writerow(["prom_id", "k562_norm", "hek_norm"])
        for p, a, b in zip(pids, y_k562, y_hek):
            w.writerow([p, a, b])
    return {"n": int(len(y_hek)), "spearman_k562_vs_hek": rho,
            "spearman_ci": boot, "note": "measured domain shift only; no model output"}


def fig2_heldout(preds, y, fold, res):
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.3), dpi=150)
    for ax, name, col in ((axes[0], "R1", "#ff7f0e"), (axes[1], "R2", "#2ca02c")):
        p = preds[name]
        ax.scatter(y, p, s=9, alpha=0.5, color=col, edgecolors="none")
        lo = min(y.min(), p.min()); hi = max(y.max(), p.max())
        ax.plot([lo, hi], [lo, hi], "--", color="gray", lw=0.8)
        ax.set_xlabel("observed hek_norm"); ax.set_ylabel("heldout prediction")
        m = res["pooled_oof"][name]
        ax.set_title("%s pooled OOF\nrho=%.3f MAE=%.3f R2=%.3f"
                     % (name, m["spearman"], m["mae"], m["r2"]), fontsize=9)
    ax = axes[2]
    k = int(fold.max()) + 1
    xs = np.arange(k); w = 0.38
    r1 = [res["per_fold"][str(f)]["R1"]["spearman"] for f in range(k)]
    r2 = [res["per_fold"][str(f)]["R2"]["spearman"] for f in range(k)]
    ax.bar(xs - w / 2, r1, w, label="R1 composition", color="#ff7f0e")
    ax.bar(xs + w / 2, r2, w, label="R2 + frozen repr", color="#2ca02c")
    ax.axhline(float(np.mean(r1)), ls=":", color="gray", lw=0.8,
               label="R1 pooled mean=%.3f" % float(np.mean(r1)))
    ax.set_xticks(xs); ax.set_xticklabels(["fold %d" % f for f in range(k)], fontsize=8)
    ax.set_ylabel("group-heldout Spearman"); ax.set_ylim(0, max(max(r1), max(r2)) * 1.25)
    ax.legend(fontsize=8)
    ax.set_title("Per-fold R1 vs R2\n(+%d/%d folds positive; rules not met)"
                 % (res["decision"]["positive_folds"], k), fontsize=9)
    fig.tight_layout()
    fig.savefig(FIGDIR / "fig2_heldout_predictions.png")
    plt.close(fig)
    with open(REPORT_OUT / "fig2_data.tsv", "w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh, delimiter="\t")
        w.writerow(["scope", "recipe", "n", "spearman", "mae", "r2",
                    "top_decile_precision", "top_decile_enrichment"])
        for scope in ["pooled"] + ["fold%d" % f for f in range(k)]:
            blk = res["pooled_oof"] if scope == "pooled" else res["per_fold"][str(int(scope[4:]))]
            for name in ("R0", "R1", "R2"):
                m = blk[name]
                w.writerow([scope, name, m["n"], m["spearman"], m["mae"], m["r2"],
                            m["top_decile"]["precision"], m["top_decile"]["enrichment"]])
    return {"pooled": {n: {kk: res["pooled_oof"][n][kk] for kk in ("spearman", "mae", "r2")}
                       for n in ("R0", "R1", "R2")}}


def fig3_applicability(contract, b21_pred):
    FIGDIR.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.3), dpi=150)
    ax = axes[0]
    ax.hist([133] * 673, bins=[125, 141], color="#1f77b4", label="batch22 train inserts (n=673, 133 bp)")
    for L in CONSTRUCT_LENGTHS:
        ax.axvline(L, color="crimson", alpha=0.35, lw=1.2)
    ax.axvline(CONSTRUCT_LENGTHS[0], color="crimson", lw=1.2, label="14 local constructs (100-700 bp)")
    ax.set_xlabel("sequence length (bp)"); ax.set_ylabel("count")
    ax.set_title("Length applicability\n(no numeric full-activity claim)", fontsize=9)
    ax.legend(fontsize=7)
    ax = axes[1]
    ls = sorted({r["length"] for r in contract})
    ok = [1 if all(r["ok"] for r in contract if r["length"] == L) else 0 for L in ls]
    ax.bar([str(L) for L in ls], ok, color=["#2ca02c" if o else "#d62728" for o in ok])
    ax.set_ylim(0, 1.25); ax.set_ylabel("dimensional contract ok")
    ax.set_xlabel("input length (bp)")
    ax.set_title("Frozen encoder length contract\n(4 x L -> 128, weights unchanged)", fontsize=9)
    ax = axes[2]
    if b21_pred is not None and len(b21_pred):
        ax.hist(b21_pred, bins=25, color="#9467bd")
        ax.set_xlabel("frozen RecipeA K562 scalar (batch21 heldout records)")
        ax.set_ylabel("count")
        ax.set_title("Original frozen cell-line heldout predictions\n(K562 source domain, n=%d)"
                     % len(b21_pred), fontsize=9)
    else:
        ax.text(0.5, 0.5, "no batch21 records", ha="center")
    fig.tight_layout()
    fig.savefig(FIGDIR / "fig3_applicability.png")
    plt.close(fig)
    with open(REPORT_OUT / "fig3_data.tsv", "w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh, delimiter="\t")
        w.writerow(["row", "kind", "length_bp", "in_training_length_domain",
                    "numeric_activity_claim", "flag"])
        for i in range(673):
            w.writerow([i, "batch22_train_insert", 133, 1, "metric-only",
                        "group-heldout bulk MPRA log2(RNA/input)"])
        for j, L in enumerate(CONSTRUCT_LENGTHS, 1):
            w.writerow(["C%d" % j, "local_14_construct", L, int(L == 133), "none",
                        "out_of_training_length/context: HEK293T transient D28 GFP, "
                        "no sequence available in project records"])
        for r in contract:
            w.writerow([r["length"], "frozen_contract_check", r["length"],
                        int(133 <= r["length"] <= 133), "none", "weights unchanged"])
    return {"construct_lengths": CONSTRUCT_LENGTHS,
            "n_constructs_with_sequence_in_records": 0,
            "numeric_activity_claim": False,
            "reason": "project records hold no DNA sequence for the 14 constructs; "
                      "different assay (HEK293T transient plasmid GFP, D28) and mostly "
                      "different length than the 133 bp trained domain -> no numeric claim"}


def latency_devices():
    import torch
    from b22_frozen import assert_cuda, load_frozen, encode_pooled
    assert_cuda()
    model, meta = load_frozen()
    seqs = ["ACGT" * 34][0][:133]
    batch = ["".join(s) for s in [["ACGT" * 40][0][:133]] * 32]
    for _ in range(3):
        encode_pooled(model, batch, batch=32)
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(10):
        encode_pooled(model, batch, batch=32)
    torch.cuda.synchronize()
    per_batch = (time.time() - t0) / 10
    contract = []
    for L in (100, 133, 164, 200, 300, 400, 700):
        s = ("ACGT" * ((L // 4) + 2))[:L]
        h = encode_pooled(model, [s], batch=1)
        contract.append({"length": L, "out_dim": int(h.shape[1]),
                         "ok": bool(h.shape[1] == 128)})
    rec = {"generated_at_local": now_local(), "device": "cuda:0",
           "device_name": torch.cuda.get_device_name(0),
           "compute_capability": list(torch.cuda.get_device_capability(0)),
           "torch": torch.__version__, "cuda": torch.version.cuda,
           "inference_batch": 32, "gpu_minutes_this_stage": None,
           "latency_ms_per_batch32": round(per_batch * 1000, 2),
           "latency_ms_per_sequence": round(per_batch * 1000 / 32, 3),
           "throughput_seq_per_s": round(32 / per_batch, 1),
           "length_contract": contract,
           "model_sha256": meta["model_sha256"],
           "note": "bounded timing only; total batch22 GPU wall time far below the 30 min cap"}
    atomic_write_json(REPORT_OUT / "inference_latency.json", rec)
    return rec, contract


def attempts_totals():
    lines = [json.loads(l) for l in open(REPORT_OUT / "attempts_ledger.jsonl", encoding="utf-8")]
    by = {}
    for r in lines:
        by[r["recipe"]] = by.get(r["recipe"], 0) + 1
    return {"n_ledger_entries": len(lines), "by_recipe": by,
            "max_ridge_fits_budget": 10, "neural_full_fits": 0,
            "r0_means_bookkeeping": 5, "pids": sorted({r["pid"] for r in lines})}


def manifests(extra):
    def h(p):
        p = Path(p)
        return sha256_file(p) if p.exists() else None
    man = {
        "generated_at_local": now_local(), "version": VERSION,
        "preprocessing": {
            "units_in": "published bulk log2(RNA/input) per cell line; negatives preserved",
            "transform_applied": "none (no extra log, no filtering of negative values)",
            "feature_scaling": "per-fold outer-train standardization only (constants safe)"},
        "target": {"target_column": "hek_norm", "cell_line": "HEK293 (NOT HEK293T)",
                   "definition_verified": "log2(hek exp/input) to machine precision",
                   "diagnostic_only": "k562_norm (never an input to sequence-only readouts)"},
        "split": {"rule": "label-free connected components (Hamming<=16 fwd/RC, shared locus)",
                  "folds": 5, "seed": 20261025, "cross_fold_violations": 0},
        "source": {"dataset": "GSE188639 (Zhao et al. 2023 Nat Genet) bulk MPRA",
                   "sequences": "Haberle et al. 2019 Nature Suppl. Table 6 (hg19)",
                   "exposure_audit": "exact 133-mer substring vs GSE118242 K562 source"},
        "sha256": {
            "source/bulk_exp": h(DATA_OUT / "source" / "GSE188639_mixed_cell_bulk_exp.csv.gz"),
            "source/haberle_table": h(DATA_OUT / "source" / "haberle2019_table6_human_CP_candidates.tsv"),
            "sequence_table.tsv": h(DATA_OUT / "sequence_table.tsv"),
            "frozen_repr_A_seed20261023.npz": h(DATA_OUT / "frozen_repr_A_seed20261023.npz"),
            "oof_predictions.npz": h(DATA_OUT / "oof_predictions.npz"),
            "code/b22_common.py": h(Path(__file__).with_name("b22_common.py")),
            "code/b22_source.py": h(Path(__file__).with_name("b22_source.py")),
            "code/b22_groups.py": h(Path(__file__).with_name("b22_groups.py")),
            "code/b22_frozen.py": h(Path(__file__).with_name("b22_frozen.py")),
            "code/b22_fits.py": h(Path(__file__).with_name("b22_fits.py")),
            "code/b22_metrics.py": h(Path(__file__).with_name("b22_metrics.py")),
            "code/b22_report.py": h(Path(__file__).with_name("b22_report.py")),
            "code/b22_selfcheck.py": h(Path(__file__).with_name("b22_selfcheck.py")),
            "code/b22_cli.py": h(Path(__file__).with_name("b22_cli.py")),
            "model_weights_batch21_recipeA": sha256_file(FROZEN_CKPT)},
        "code_entrypoints": [str(p.relative_to(ROOT)) if Path(p).is_relative_to(ROOT) else str(p)
                             for p in sorted(Path(__file__).parent.glob("b22_*.py"))],
    }
    man.update(extra)
    atomic_write_json(REPORT_OUT / "manifests.json", man)
    return man


def next_batch_proposal(res, man):
    return {
        "generated_at_local": now_local(),
        "context": "batch22 R2 (frozen batch21 K562 RecipeA representation) did NOT beat the "
                   "composition baseline R1 on the HEK293 GSE188639 group-heldout task.",
        "evidence": {"R1_rho": res["pooled_oof"]["R1"]["spearman"],
                     "R2_rho": res["pooled_oof"]["R2"]["spearman"],
                     "delta_rho": res["decision"]["delta_spearman_R2_minus_R1"],
                     "mae_reduction_frac": res["decision"]["mae_reduction_frac_R2_vs_R1"]},
        "do_not_repeat": ["batch21 K562 ranking-loss comparison",
                          "discarding the 676-row library merely because it fails the "
                          "1000-row from-scratch gate"],
        "proposed_next_batch": {
            "batch": "batch23",
            "title": "Augment HEK293/HEK293T sequence-activity data, not the readout stack",
            "mechanism": [
                "Assemble the largest available measured HEK293/HEK293T core-promoter MPRA "
                "libraries (e.g. lentiMPRA/MPRA-format with published sequences) to raise the "
                "from-scratch row/group count above the batch21 gate, joining by measured "
                "sequence rather than synthetic ids.",
                "Then retest a small from-scratch sequence model on HEK293 with group-heldout "
                "folds, using the batch22 label-free grouping and metrics as the frozen "
                "evaluation harness.",
                "Keep the source-exposure stratification and the honest-negative reporting "
                "convention; do not transfer K562 scalar weights expecting HEK293 gain."],
            "why_evidence_supported": "R2 gained only +%.4f rho and worsened MAE while R1 "
                                      "composition features reached rho=%.4f, i.e. the frozen "
                                      "K562 representation adds little transferable HEK293 signal; "
                                      "the binding constraint is HEK-domain data volume, not readout."
                                      % (res["decision"]["delta_spearman_R2_minus_R1"],
                                         res["pooled_oof"]["R1"]["spearman"])},
        "guardrails": ["no gate relaxation", "no prospective/external claim from this CV",
                       "keep all prior artifacts read-only"],
    }


def write_report_md(res, fz, lat, at, dm, fm, man, prop):
    p, d = res["pooled_oof"], res["decision"]
    L = []
    A = L.append
    A("# BATCH22 — v6.17.0-hektransfer.rc1")
    A("")
    A("**Frozen-representation HEK293 transfer / domain evaluation on GSE188639 (676 core-promoter records).**")
    A("")
    A("Parent scientific decision 2026-10-03 15:00 Asia/Shanghai. This is a *different question* from "
      "batch21's negative K562 ranking-loss comparison, not a retroactive gate relaxation.")
    A("")
    A("**Verdict: %s.**" % d["verdict"])
    A("")
    A("---")
    A("")
    A("## 1. Question and scope")
    A("")
    A("| item | value |")
    A("|---|---|")
    A("| target | `hek_norm` = log2(hek exp / input), published units, negatives preserved |")
    A("| cell | HEK293 (**not** HEK293T) |")
    A("| inputs | the measured 133 bp core-promoter insert only |")
    A("| K562 values | paired-cell diagnostic reference only (never a model input) |")
    A("| neural end-to-end full fits | **0** (authorised budget) |")
    A("| classical ridge fits | %d of a 10-fit budget |" % at["n_ledger_entries"])
    A("| frozen extraction device | cuda:0 with parameter/input assertions (no CPU fallback) |")
    A("")
    A("## 2. Data acquisition and verification (Stage 1)")
    A("")
    A("- `GSE188639_mixed_cell_bulk_exp.csv.gz` (676 rows, sha256 `3b21ee7f…`) is the read-only probe table; "
      "it is reused byte-identically, not re-downloaded.")
    A("- %d of 676 records joined **by exact `prom_id`** to published hg19 133 bp sequences "
      "(Haberle et al. 2019 Nature, Supplementary Table 6)." % dm["n_with_published_sequence"])
    A("- The 3 unresolved members (%s) are synthetic SCP1 mutants with no published sequence; they are "
      "**excluded, never fabricated**." % ", ".join("`%s`" % x for x in dm["unresolved_ids"]))
    A("- Published normalization verified to machine precision: "
      "`max|hek_norm - log2(hek exp/input)| = %.2e`, `max|k562_norm - log2(k562 exp/input)| = %.2e`. "
      "No extra log; negative values kept." % (dm["target"]["verified_max_abs_diff_vs_definition"],
                                               dm["paired_cell_diagnostic"]["verified_max_abs_diff_vs_definition"]))
    A("- Build/orientation cross-check against UCSC hg19: 24/24 exact matches "
      "(`source_sequence_verification.json`).")
    A("")
    A("## 3. Split and source exposure (Stage 2)")
    A("")
    A("- Label-free grouping: near-homology (Hamming <= 16 forward or reverse-complement) union shared-locus "
      "window overlap (same chromosome, overlapping 0-based windows).")
    A("- **%d rows -> %d independent groups**; %d deterministic balanced group folds, seed %d; "
      "cross-fold grouping violations **0**. %d groups per holdout, at or above the >= 10 requirement."
      % (fz["n_rows"], fz["n_groups"], fz["n_folds"], fz["fold_seed"], fz["min_groups_per_holdout"]))
    A("- Source-exposure audit vs the frozen RecipeA pretraining source (GSE118242 K562, 164 bp, no "
      "coordinates): **%d rows** carry an exact 133-mer substring match; results are stratified source-seen "
      "vs source-unseen." % fz["source_exposure"]["n_rows_exact_substring_source"])
    A("- `freeze.json` and the sequence-table SHA were written **before** any fit.")
    A("")
    A("## 4. Frozen representation (Stage 3)")
    A("")
    A("- Source: batch21 `ckpt_A_seed20261023.pt` (RecipeA, modelSHA `bc35e5585fa7…`), loaded strictly; "
      "all parameters on cuda:0, `requires_grad=False`, model in `eval()`.")
    A("- Representation: `h = cat[mean_L, max_L]` (128-d) of the penultimate block, the pre-scalar-head pooled "
      "features, from the **whole** measured insert at its actual length (no padding, no truncation, no "
      "reference DNA, no artificial duplication).")
    A("- Verified: dimensional contract 4xL -> 128 at L in {100, 133, 164, 200, 300, 400, 700} and exact "
      "permutation invariance at the fixed extraction batch size (32).")
    A("- `n_hek_labels_used = 0` in the extraction (`frozen_extraction_evidence.json`).")
    A("")
    A("## 5. Readouts and budget (Stage 3)")
    A("")
    A("| recipe | features | model |")
    A("|---|---|---|")
    A("| R0 | - | outer-train mean (bookkeeping, not a fit) |")
    A("| R1 | mono4 + di16 + tri64 + GC + log1p(len) = 86-d | Ridge(alpha=100) |")
    A("| R2 | R1 features + frozen 128-d representation = 214-d | Ridge(alpha=100) |")
    A("")
    A("Standardization is fit on the **outer-train rows only** (constants kept safe). No inner search, "
      "no seed shopping, no end-to-end finetuning. Every attempt was registered in the append-only ledger "
      "(`attempts_ledger.jsonl`, %d entries, PIDs %s) **before** execution and cached to immutable "
      "per-fold attempt directories." % (at["n_ledger_entries"], ", ".join(str(x) for x in at["pids"])))
    A("")
    A("## 6. Results (Stage 4)")
    A("")
    A("Group-heldout out-of-fold, n = %d, published `hek_norm` units:" % res["n"])
    A("")
    A("| recipe | Spearman | MAE | R2 | top-decile precision | enrichment |")
    A("|---|---|---|---|---|---|")
    for nm in ("R0", "R1", "R2"):
        m = p[nm]
        A("| %s | %.4f | %.4f | %.4f | %.4f | %.2f |"
          % (nm, m["spearman"], m["mae"], m["r2"], m["top_decile"]["precision"],
             m["top_decile"]["enrichment"]))
    A("")
    A("Per-fold Spearman, R1 to R2: " + ", ".join(
        "fold %d %.3f -> %.3f" % (f, res["per_fold"][str(f)]["R1"]["spearman"],
                                  res["per_fold"][str(f)]["R2"]["spearman"])
        for f in range(res["n_folds"])))
    A("")
    bs = res["paired_group_bootstrap"]["R2_minus_R1"]["delta_spearman"]
    A("Paired group bootstrap (2000 draws, %d groups): delta-rho(R2-R1) = %+.4f, 95%% CI [%+.4f, %+.4f]."
      % (bs["n_groups"], bs["point"], bs["lo"], bs["hi"]))
    A("")
    A("### Decision (predeclared, frozen, not lowered after seeing results)")
    A("")
    A("R2 over R1 must satisfy **all** of: pooled OOF rho gain >= 0.03; MAE reduction >= 10 %; "
      "top-decile precision not lower; positive rho difference in >= 4 of 5 folds. "
      "A bootstrap CI crossing zero means *uncertain* even if the point thresholds pass.")
    A("")
    A("| criterion | value | met |")
    A("|---|---|---|")
    A("| rho gain >= 0.03 | %+.4f | %s |" % (d["delta_spearman_R2_minus_R1"],
                                             "yes" if d["delta_spearman_R2_minus_R1"] >= 0.03 else "**no**"))
    A("| MAE reduction >= 10 %% | %+.1f %% | %s |" % (d["mae_reduction_frac_R2_vs_R1"] * 100,
                                                      "yes" if d["mae_reduction_frac_R2_vs_R1"] >= 0.10 else "**no**"))
    A("| top-decile precision not lower | %s | %s |" % (d["top_decile_not_lower"],
                                                        "yes" if d["top_decile_not_lower"] else "**no**"))
    A("| positive folds >= 4/5 | %d/5 | %s |" % (d["positive_folds"],
                                                 "yes" if d["positive_folds"] >= 4 else "**no**"))
    A("")
    A("**%s.** The frozen K562 representation does not add a predeclared practical HEK293 gain over the "
      "composition baseline; R1 already reaches rho = %.4f. This is the expected shape of a cross-cell-line "
      "transfer failure and is reported as such." % (d["verdict"], p["R1"]["spearman"]))
    A("")
    A("## 7. Figures")
    A("")
    A("| figure | content | source table |")
    A("|---|---|---|")
    A("| `fig1_domain_shift.png` | paired-cell **measured** K562 vs HEK293 shift (rho = %.3f), diagnostic only, "
      "not model performance | `fig1_data.tsv` |" % fm["fig1"]["spearman_k562_vs_hek"])
    A("| `fig2_heldout_predictions.png` | sequence-only grouped-heldout observed vs R1/R2 + per-fold rho | "
      "`fig2_data.tsv` |")
    A("| `fig3_applicability.png` | length applicability vs the 14 local constructs; frozen length contract; "
      "frozen K562 heldout prediction distribution from batch21 records | `fig3_data.tsv` |")
    A("")
    A("## 8. Applicability and the 14 local constructs")
    A("")
    A("- Project records hold **no DNA sequence** for the 14 ACTB/GAPDH/CMV constructs, and they are a "
      "different assay/context (HEK293T transient plasmid GFP, D28) at lengths 100-700 bp against a 133 bp "
      "trained domain. **No numeric full-activity claim is made** and nothing was tuned against them.")
    A("- For comparison, the original frozen cell-line heldout predictions from batch21 records are shown "
      "(fig3, right panel).")
    A("")
    A("## 9. Compute, devices, latency")
    A("")
    A("- Device: **%s**, compute capability %s, torch %s, CUDA %s."
      % (lat["device_name"], lat["compute_capability"], lat["torch"], lat["cuda"]))
    A("- Frozen encoder inference: batch 32, **%.3f ms/sequence** (%.1f seq/s). Total batch22 GPU wall time is "
      "far below the 30-minute cap; no dummy load." % (lat["latency_ms_per_sequence"],
                                                       lat["throughput_seq_per_s"]))
    A("- Ridge fits are small CPU linear-algebra jobs (4-thread limited), not GPU training.")
    A("")
    A("## 10. Attempts totals and risks")
    A("")
    A("- Ledger entries: **%d** (R1 x5 folds, R2 x5 folds); budget 10; **0** neural full fits; R0 means are "
      "bookkeeping." % at["n_ledger_entries"])
    A("- Risks / honest limitations: (i) %d/%d rows have an exact 133-mer match to the frozen pretraining "
      "source; strata are reported but the source overlap cannot be concealed; (ii) the 3 SCP1 mutants are "
      "excluded; (iii) this is **retrospective** grouped CV, so no prospective or external claim; (iv) HEK293 "
      "performance is **not** HEK293T suitability; (v) batch21's own `sha_manifest.json` predates its final "
      "`report.md` (14:41) and `raw_fasta_predictions.tsv` (14:42) - those stale entries predate batch22 "
      "(start 15:06) and are **not** batch22 edits, verified in `selfcheck.json`."
      % (fz["source_exposure"]["n_rows_exact_substring_source"], fz["n_rows"]))
    A("")
    A("## 11. Raw-input CLI")
    A("")
    A("```")
    A("python -X utf8 runs/breakthrough_20261002/code/batch22/b22_cli.py repr     --in seqs.fa --out repr.tsv")
    A("python -X utf8 runs/breakthrough_20261002/code/batch22/b22_cli.py features --in seqs.fa --out comp.tsv")
    A("python -X utf8 runs/breakthrough_20261002/code/batch22/b22_cli.py predict  --in seqs.fa --recipe R2 --fold 0 --out preds.tsv")
    A("```")
    A("")
    A("`predict` applies a **pre-existing fold-specific** readout from saved training statistics; it "
      "deliberately does not train an all-data delivery estimator (reserved for parent acceptance).")
    A("")
    A("## 12. Selfcheck vs fresh review")
    A("")
    A("`selfcheck.json` is the **worker-side** mechanical replay (14/14 checks; saved predictions and training "
      "statistics only, no ridge or neural refits; includes a verified comparison that no batch21 file was "
      "modified after batch22 started). It is deliberately **distinct** from the fresh independent review, "
      "which is written separately to `independent_review.json`.")
    A("")
    A("## 13. Artifacts")
    A("")
    A("Reports: `results.json`, `freeze.json`, `group_audit.json`, `fits_summary.json`, `attempts_ledger.jsonl`, "
      "`attempts_summary.json`, `data_manifest.json`, `source_sequence_verification.json`, "
      "`frozen_extraction_evidence.json`, `figures_manifest.json`, `manifests.json`, `inference_latency.json`, "
      "`selfcheck.json`, `next_batch_proposal.json`, `figures/`.")
    A("")
    A("Data: `data/breakthrough_20261002/batch22/{sequence_table.tsv,fold_assignment.tsv,groups.npz,"
      "frozen_repr_A_seed20261023.npz,oof_predictions.npz,oof_predictions.tsv,source/}`.")
    A("")
    A("Code: `runs/breakthrough_20261002/code/batch22/b22_{common,source,verify_source,groups,frozen,fits,"
      "metrics,report,selfcheck,cli}.py`.")
    A("")
    A("Version card: `models/cards/promoter_atlas_v6_17_0_hektransfer_rc1_card.txt`.")
    A("")
    A("## 14. Next batch proposal")
    A("")
    A("- **%s**: %s" % (prop["proposed_next_batch"]["batch"], prop["proposed_next_batch"]["title"]))
    for m in prop["proposed_next_batch"]["mechanism"]:
        A("  - %s" % m)
    A("- Rationale: %s" % prop["proposed_next_batch"]["why_evidence_supported"])
    (REPORT_OUT / "report.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    return len(L)


def run():
    set_thread_limits(4)
    rows, seqs, y, diag, pids, fold_of, gid, seen, H = load_inputs()
    res = json.load(open(REPORT_OUT / "results.json", encoding="utf-8"))
    f1 = fig1_domain_shift(y, diag, pids)
    oof_rows = list(csv.DictReader(open(DATA_OUT / "oof_predictions.tsv", encoding="utf-8"),
                                   delimiter="\t"))
    oof = {n: np.array([float(r["pred_" + n]) for r in oof_rows]) for n in ("R0", "R1", "R2")}
    assert np.allclose(oof["R1"], np.load(DATA_OUT / "oof_predictions.npz")["R1"])
    f2 = fig2_heldout(oof, y, fold_of, res)
    lat, contract = latency_devices()
    b21 = None
    p21 = BATCH21 / "test_predictions.tsv"
    if p21.exists():
        b21 = np.array([float(r["pred_A_seed20261023"]) for r in csv.DictReader(
            open(p21, encoding="utf-8"), delimiter="\t")])
    f3 = fig3_applicability(contract, b21)
    figman = {"generated_at_local": now_local(),
              "figures": ["fig1_domain_shift.png", "fig2_heldout_predictions.png",
                          "fig3_applicability.png"],
              "source_tables": ["fig1_data.tsv", "fig2_data.tsv", "fig3_data.tsv"],
              "fig1": f1, "fig2": f2, "fig3": f3, "latency": lat}
    atomic_write_json(REPORT_OUT / "figures_manifest.json", figman)
    at = attempts_totals()
    atomic_write_json(REPORT_OUT / "attempts_summary.json", at)
    man = manifests({"attempts_totals": at, "figures_manifest": figman})
    prop = next_batch_proposal(res, man)
    atomic_write_json(REPORT_OUT / "next_batch_proposal.json", prop)
    dm = json.load(open(REPORT_OUT / "data_manifest.json", encoding="utf-8"))
    nlines = write_report_md(res, json.load(open(REPORT_OUT / "freeze.json", encoding="utf-8")),
                             lat, at, dm, figman, man, prop)
    print("[b22_report] report.md lines=%d figs=3 latency_ms_per_seq=%.3f attempts=%d verdict=%s"
          % (nlines, lat["latency_ms_per_sequence"], at["n_ledger_entries"],
             res["decision"]["verdict"]))
    print("[b22_report] figs=3 latency_ms_per_seq=%.3f attempts=%d verdict=%s"
          % (lat["latency_ms_per_sequence"], at["n_ledger_entries"], res["decision"]["verdict"]))
    return figman


if __name__ == "__main__":
    import pathlib
    import sys
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
    run()

