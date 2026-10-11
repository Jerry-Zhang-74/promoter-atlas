#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""batch23 figures: (1) actual construct lengths / applicability vs the training
length; (2) descriptive diagnostic ranking of the untuned model score vs the local
HEK293T D28 endpoint. Source data tables are written alongside. ZERO fits."""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
from b23_common import (ROOT, DATA_OUT, REPORT_OUT, TRAIN_LEN, now_local,  # noqa: E402
                        sha256_file, atomic_write_json, spearman)
from b23_ready import main as ready_main  # noqa: E402

FIGS = ROOT / "figures" / "batch23"


def read_tsv(p):
    with open(p, encoding="utf-8-sig", newline="") as fh:
        return list(csv.DictReader(fh, delimiter="\t"))


def main():
    FIGS.mkdir(parents=True, exist_ok=True)
    ready_main()  # ensure data_ready/next_batch_proposal present (idempotent)
    recs = read_tsv(DATA_OUT / "local_diagnostic_scores.tsv")
    loc = {r["construct_id"]: r for r in read_tsv(DATA_OUT / "local_constructs.tsv")}
    cids = [r["construct_id"] for r in recs]
    L = np.array([int(r["sequence_length_bp"]) for r in recs])
    sup = np.array([int(r["length_supported_by_training"]) for r in recs])
    y = np.array([float(r["observed_target_log10_gfp"]) for r in recs])
    p2 = np.array([float(r["R2_mean_output"]) for r in recs])
    p1 = np.array([float(r["R1_mean_output"]) for r in recs])

    # ---------------- Figure 1: lengths / applicability ----------------
    order = np.argsort(L)
    fig, ax = plt.subplots(figsize=(8.6, 4.4), dpi=150)
    xs = np.arange(len(order))
    cols = ["#2b7bba" if sup[i] else "#c0392b" for i in order]
    ax.bar(xs, L[order], color=cols, edgecolor="black", linewidth=0.5)
    ax.axhline(TRAIN_LEN, color="black", ls="--", lw=1.2,
               label="training length = %d bp" % TRAIN_LEN)
    ax.set_xticks(xs)
    ax.set_xticklabels([cids[i] for i in order], rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("actual insert length (bp)")
    ax.set_title("batch23: actual local construct lengths vs frozen training length\n"
                 "(0/14 assay barcode verified; blue = length==training)", fontsize=9)
    ax.legend(fontsize=8, loc="upper left")
    for k, i in enumerate(order):
        ax.text(k, L[i] + 25, str(L[i]), ha="center", fontsize=7)
    fig.tight_layout()
    fig.savefig(FIGS / "fig_construct_lengths_applicability.png")
    plt.close(fig)

    with open(FIGS / "fig_construct_lengths_data.tsv", "w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh, delimiter="\t")
        w.writerow(["construct_id", "family", "sequence_length_bp", "training_length_bp",
                    "length_supported_by_training", "assay_mapping_certainty",
                    "assay_to_plasmid_barcode_verified"])
        for r in recs:
            w.writerow([r["construct_id"], r["family"], r["sequence_length_bp"], r["training_length_bp"],
                        r["length_supported_by_training"], r["assay_mapping_certainty"],
                        r["assay_to_plasmid_barcode_verified"]])

    # ---------------- Figure 2: diagnostic ranking ----------------
    rho2 = spearman(p2, y); rho1 = spearman(p1, y)
    fig, ax = plt.subplots(figsize=(5.6, 5.2), dpi=150)
    ax.scatter(y, p2, s=46, c="#2b7bba", edgecolor="black", linewidth=0.6, zorder=3)
    for i, c in enumerate(cids):
        ax.annotate(c, (y[i], p2[i]), fontsize=7, xytext=(3, 3),
                    textcoords="offset points")
    ax.set_xlabel("observed log10 relative GFP (HEK293T D28 endpoint)")
    ax.set_ylabel("exploratory_model_score\n(batch22 hek_norm log2 scale; UNSUPPORTED)")
    ax.set_title("batch23 descriptive OOD diagnostic (n=14)\n"
                 "Spearman R2=%.3f  R1=%.3f; NO calibration, NO cross-scale claim" % (rho2, rho1),
                 fontsize=8)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIGS / "fig_diagnostic_ranking.png")
    plt.close(fig)

    with open(FIGS / "fig_diagnostic_ranking_data.tsv", "w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh, delimiter="\t")
        w.writerow(["construct_id", "observed_log10_gfp", "R1_mean_output", "R2_mean_output",
                    "R2_fold_spread_sd"])
        for r in recs:
            w.writerow([r["construct_id"], r["observed_target_log10_gfp"], r["R1_mean_output"],
                        r["R2_mean_output"], r["R2_fold_spread_sd"]])

    atomic_write_json(REPORT_OUT / "figures_manifest.json", {
        "generated_at_local": now_local(),
        "figures": {
            str((FIGS / "fig_construct_lengths_applicability.png").relative_to(ROOT)):
                sha256_file(FIGS / "fig_construct_lengths_applicability.png"),
            str((FIGS / "fig_diagnostic_ranking.png").relative_to(ROOT)):
                sha256_file(FIGS / "fig_diagnostic_ranking.png"),
        },
        "source_tables": {
            str((FIGS / "fig_construct_lengths_data.tsv").relative_to(ROOT)):
                sha256_file(FIGS / "fig_construct_lengths_data.tsv"),
            str((FIGS / "fig_diagnostic_ranking_data.tsv").relative_to(ROOT)):
                sha256_file(FIGS / "fig_diagnostic_ranking_data.tsv"),
        },
        "n_constructs": len(recs),
        "spearman_R2": rho2, "spearman_R1": rho1,
    })
    print("[b23_figs] ok n=%d rhoR2=%.4f rhoR1=%.4f" % (len(recs), rho2, rho1))


if __name__ == "__main__":
    main()
