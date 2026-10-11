#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Batch37 step 4 - three figures max (CPU only)."""
import csv, json, os, hashlib
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUTDIR = os.path.join(REPO, "data/breakthrough_20261002/batch37")
RPT = os.path.join(REPO, "reports/breakthrough_20261002/batch37")
FIG = os.path.join(REPO, "figures/batch37")
os.makedirs(FIG, exist_ok=True)

def _rank(a):
    order = np.argsort(a, kind="mergesort"); r = np.empty(len(a), float)
    r[order] = np.arange(len(a), dtype=float); out = r.copy(); i = 0; s = np.sort(a)
    while i < len(a):
        j = i
        while j + 1 < len(a) and s[j + 1] == s[i]: j += 1
        if j > i:
            avg = (i + j) / 2.0
            for k in range(i, j + 1): out[order[k]] = avg
        i = j + 1
    return out

def load():
    orders = {}
    with open(os.path.join(OUTDIR, "unique_scores.tsv"), encoding="utf-8") as f:
        for r in csv.DictReader(f, delimiter="\t"): orders[r["seq_id"]] = float(r["mean_score"])
    pairs = []
    with open(os.path.join(OUTDIR, "paired_inputs.tsv"), encoding="utf-8") as f:
        for r in csv.DictReader(f, delimiter="\t"):
            lab = r["label_logFC"]; lab = float(lab) if lab not in ("", "None") else float("nan")
            a = hashlib.sha256(r["alt_insert"].encode()).hexdigest()[:20]
            rf = hashlib.sha256(r["ref_insert"].encode()).hexdigest()[:20]
            pairs.append({"arm": r["arm"], "y": lab, "d": orders[a] - orders[rf], "pos": int(r["pos"]),
                          "status": r["pair_status"], "vclass": r["variant_class"], "group": int(r["group_id"])})
    return pairs

def fig1(pairs):
    sat = [p for p in pairs if p["arm"] == "SAT" and np.isfinite(p["y"])]
    kv = [p for p in pairs if p["arm"] == "KV" and np.isfinite(p["y"])]
    m = json.load(open(os.path.join(RPT, "metrics.json")))
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.6))
    for a, sub, name in [(ax[0], sat, "SAT (single 2kb locus)"), (ax[1], kv, "KV (known variants, multi-region)")]:
        rho = m["sat_all"]["spearman_d_logFC"] if name.startswith("SAT") else m["kv_all"]["spearman_d_logFC"]
        x = [p["d"] for p in sub]; y = [p["y"] for p in sub]
        a.scatter(x, y, s=4, alpha=0.25, color=("#c0392b" if name.startswith("SAT") else "#2c7fb8"))
        a.axhline(0, color="k", lw=0.5); a.axvline(0, color="k", lw=0.5)
        a.set_xlabel("predicted effect d = mean_score(alt) - mean_score(ref)")
        a.set_ylabel("measured bcalm logFC (author)")
        a.set_title("%s\nSpearman rho = %.3f (descriptive)" % (name, rho), fontsize=9)
    fig.suptitle("Batch37: frozen PARM HEK293 score vs matched variant-reference effect (ZERO-fit)", fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out = os.path.join(FIG, "fig1_paired_effect_scatter.png"); fig.savefig(out, dpi=140); plt.close(fig)
    return out

def fig2(pairs):
    import math
    kv = [p for p in pairs if p["arm"] == "KV" and np.isfinite(p["y"])]
    g = {}
    for p in kv: g.setdefault(p["group"], []).append(p)
    rhos = []
    for gi, ps in g.items():
        y = np.array([p["y"] for p in ps]); d = np.array([p["d"] for p in ps])
        if len(ps) >= 10 and not np.all(d == d[0]):
            rx = _rank(d); ry = _rank(y); rx = rx - rx.mean(); ry = ry - ry.mean()
            den = math.sqrt((rx * rx).sum() * (ry * ry).sum())
            if den > 0: rhos.append((rx * ry).sum() / den)
    m = json.load(open(os.path.join(RPT, "metrics.json")))
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.6))
    ax[0].hist(rhos, bins=25, color="#2c7fb8", alpha=0.8)
    ax[0].axvline(0, color="k", lw=0.8)
    ax[0].set_xlabel("per-group Spearman rho"); ax[0].set_ylabel("n groups")
    ax[0].set_title("KV per-group rho (n=%d groups, >=10 pairs)" % len(rhos), fontsize=9)
    labels = ["model\n(all)", "model\n(valid)", "always+", "always-"]
    vals = [m["kv_all"]["model_strict_sign_accuracy"], m["kv_valid_only"]["model_strict_sign_accuracy"],
            m["kv_all"]["fixed_always_positive_accuracy"], m["kv_all"]["fixed_always_negative_accuracy"]]
    ax[1].bar(labels, vals, color=["#2c7fb8", "#7fb3d5", "#bdbdbd", "#bdbdbd"])
    ax[1].axhline(0.5, color="k", ls="--", lw=0.8)
    ax[1].set_ylim(0, 1); ax[1].set_ylabel("strict sign accuracy")
    ax[1].set_title("KV direction accuracy vs fixed baselines\n(predicted ties wrong, true zeros excluded)", fontsize=9)
    for i, v in enumerate(vals): ax[1].text(i, v + 0.01, "%.3f" % v, ha="center", fontsize=8)
    fig.suptitle("Batch37: KV region-level effect direction (exploratory groups, not verified families)", fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out = os.path.join(FIG, "fig2_kv_region_signerrors.png"); fig.savefig(out, dpi=140); plt.close(fig)
    return out

def fig3(pairs):
    m = json.load(open(os.path.join(RPT, "metrics.json")))
    fig, ax = plt.subplots(figsize=(10.5, 6.4)); ax.axis("off")
    lines = [
        "BATCH37 SCOPE / EVIDENCE CARD  (v6.26.0-frozen-variant-diagnostic.rc1)",
        "Target chassis = 293FT. Model source cell = HEK293 (PARM, 5-fold equal mean).",
        "Task: variant-induced CRE-effect transfer diagnostic. NOT autonomous-promoter reliability.",
        "",
        "RESULT (ZERO-fit; no calibration, no fitting, no fold/orientation choice):",
        "  KV known-variant SNVs (multi-region):  pooled Spearman(d, logFC) = %.3f  (all %d rows)" % (
            m["kv_all"]["spearman_d_logFC"], m["kv_all"]["n_rows"]),
        "     valid pairs only (n=%d):            rho = %.3f ; group-bootstrap 95%% CI [%.3f, %.3f]" % (
            m["kv_valid_only"]["n_rows"], m["kv_valid_only"]["spearman_d_logFC"],
            m["kv_bootstrap"]["rho_ci95"][0], m["kv_bootstrap"]["rho_ci95"][1]),
        "  SAT single-locus saturation (descriptive): rho = %.3f  (SNV %.3f / deletion %.3f)" % (
            m["sat_all"]["spearman_d_logFC"], m["sat_snv"]["spearman_d_logFC"], m["sat_del"]["spearman_d_logFC"]),
        "  deltaGC zero-fit baseline rho: KV %.3f / SAT %.3f" % (
            m["kv_all"]["spearman_deltaGC_logFC"], m["sat_all"]["spearman_deltaGC_logFC"]),
        "",
        "AUDIT (frozen, disclosed):",
        "  KV: %d/%d rows DEGENERATE (author alt oligo identical to reference; d=0) -> allele join %.1f%%." % (
            m["kv_all"]["n_pred_ties_zero_d"], m["kv_all"]["n_rows"],
            100.0 * (1 - m["kv_all"]["n_pred_ties_zero_d"] / m["kv_all"]["n_rows"])),
        "  SAT: 100%% allele-consistent; 6000 SNV + 2000 deletion kept as separate strata.",
        "  NaN labels excluded from numeric metric only: KV 17, SAT 3.",
        "  KV 2kb-gap clusters = 48 EXPLORATORY components (NOT verified independent families).",
        "  %d KV variants fall in the SAT 2kb region; SAT/KV arms never pooled." % m["sat_kv_overlap"]["kv_variants_overlapping_sat_region"],
        "",
        "DECISION RULE (operational only): a SEPARATE variant-path is warranted iff",
        "  KV rho>=0.20 AND bootstrap lower>0 AND macro sign>both fixed baselines AND no audit failure.",
        "  conditions met = %s   => VERDICT: LIMITED." % json.dumps(m["decision_rule"]["conditions"]),
        "  (no model promotion; no alteration of the old HEK MAE gate.)",
        "",
        "UNCHANGED: local 293FT D28 direction rule = FAIL (macro 0.40, n=21).",
        "  over-length (>600bp) constructs remain UNSUPPORTED (2000bp NOT clipped/tiled).",
        "  No 293FT new-family validation; external-model pretraining exposure UNKNOWN.",
    ]
    ax.text(0.01, 0.99, "\n".join(lines), va="top", ha="left", fontsize=9.0, family="monospace")
    fig.tight_layout()
    out = os.path.join(FIG, "fig3_scope_evidence_card.png"); fig.savefig(out, dpi=140); plt.close(fig)
    return out

def main():
    pairs = load()
    outs = [fig1(pairs), fig2(pairs), fig3(pairs)]
    print(json.dumps({"figures": outs}))
    return 0

if __name__ == "__main__":
    import sys; sys.exit(main())
