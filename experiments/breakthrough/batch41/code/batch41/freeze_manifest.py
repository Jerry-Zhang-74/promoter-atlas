#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""batch41: freeze the diagnostic_manifest BEFORE any model forward or joining.

Locks rows (7537 bio + 85 controls), exact 270bp insert sequences (raw[15:285]),
sequence/source/label/code/model/weights SHAs, the 270-only input contract, label-free
groups, the metric plan and the decision rule. Written before predictions exist.
"""
from __future__ import annotations

import json
import os
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import b41_common as C  # noqa: E402


def sequence_sha(seq):
    return C.sha256_text(seq.upper())


def overlapping_components(bio):
    """Label-free 270bp overlapping-window components per chromosome (step90 merge).
    Merge shared exact/RC inserts only where actually recovered; NOT independent CRE families."""
    def rc(x):
        return x.translate(str.maketrans("ACGTacgt", "TGCAtgca"))[::-1]
    byc = defaultdict(list)
    for i, r in enumerate(bio):
        byc[r["chr"]].append((int(r["start"]), int(r["end"]), i))
    comp_of = {}
    comps = []
    for c in sorted(byc):
        iv = sorted(byc[c])
        cur = None
        members = []
        for s, e, i in iv:
            if cur is not None and s <= cur:
                cur = max(cur, e)
                members.append(i)
            else:
                if members:
                    comps.append((c, members))
                cur = e
                members = [i]
        if members:
            comps.append((c, members))
    for ci, (c, members) in enumerate(comps):
        for i in members:
            comp_of[i] = ci
    seen = {}
    merges = 0
    for i, r in enumerate(bio):
        k = r["insert_270bp"].upper()
        kr = rc(k)
        if k in seen or kr in seen:
            merges += 1
        else:
            seen[k] = i
    sizes = defaultdict(int)
    for i in range(len(bio)):
        sizes[comp_of[i]] += 1
    chr_count = {}
    for ci, (c, members) in enumerate(comps):
        chr_count[c] = chr_count.get(c, 0) + 1
    return comp_of, comps, merges, chr_count, sizes


def main():
    t0 = time.time()
    rows = C.read_joined()
    bio = [r for r in rows if r["control"] == "0"]
    ctrl = [r for r in rows if r["control"] == "1"]
    assert len(bio) == 7537 and len(ctrl) == 85
    comp_of, comps, merges, chr_count, sizes = overlapping_components(bio)
    n_chr17 = chr_count.get("chr17", 0)
    n_non17 = len(comps) - n_chr17

    frozen = []
    bi = 0
    for i, r in enumerate(rows):
        seq = r["insert_270bp"].upper()
        is_ctrl = int(r["control"]) == 1
        comp_id = "" if is_ctrl else str(comp_of[bi])
        frozen.append({"label": r["label"], "control": int(r["control"]),
                       "chr": r["chr"], "start": r["start"], "end": r["end"],
                       "strand": r["strand"], "sequence": seq,
                       "sequence_sha256": sequence_sha(seq),
                       "alpha_primary": r["statistic_alpha"],
                       "z_secondary": r["z"], "mad_secondary": r["mad"],
                       "p_empirical_secondary": r["p_empirical"],
                       "p_mad_secondary": r["p_mad"], "p_z_secondary": r["p_z"],
                       "component_id": comp_id})
        if not is_ctrl:
            bi += 1
    hdr = ["label", "control", "chr", "start", "end", "strand", "sequence",
           "sequence_sha256", "alpha_primary", "z_secondary", "mad_secondary",
           "p_empirical_secondary", "p_mad_secondary", "p_z_secondary", "component_id"]
    C.atomic_write_tsv(C.DATA / "frozen_input_table.tsv", hdr,
                       [[f[h] for h in hdr] for f in frozen])
    alphas = [float(r["statistic_alpha"]) for r in bio]
    n_unique_seq = len({f["sequence_sha256"] for f in frozen})
    sizes_hist = dict(sorted(Counter(sizes.values()).items()))

    manifest = {
        "batch": C.BATCH, "version": C.VERSION,
        "record_type": "diagnostic_manifest_FROZEN_BEFORE_FORWARD",
        "frozen_at": C.now_local(), "pid": os.getpid(),
        "frozen_before_any_forward": True, "frozen_before_model_output_joining": True,
        "rows": {"biological": len(bio), "controls": len(ctrl), "total": len(frozen)},
        "input_contract": {
            "input": "exact 270bp genomic insert = raw[15:285] of the author synthetic oligo (300bp=FWD15+270+REV15)",
            "per_design_only": True, "no_RC_ensemble": True, "no_window": True,
            "no_extra_adapter": True, "no_reference_DNA_fill": True,
            "PARM_padding": "official central zero-padding to L_max=600 (bare FASTA)",
            "H1_input": "same 270bp insert, full-sequence 4/5-mer spectrum kernel (forward only)",
        },
        "labels": {
            "primary": "MPRAnalyze getAlpha activity (condition V1; uq depth factors; 3 DNA + 3 RNA aggregate)",
            "author_ascertainment": "upstream DNA-sum>=50 filter",
            "secondary_not_primary_truth": ["z", "mad", "p_empirical", "p_mad", "p_z"],
            "kept_labels": "all (zero/negative/NA/nonsignificant kept); no activity-threshold or p-value selection",
            "alpha_positive_all": all(a > 0 for a in alphas),
            "alpha_finite_all": all(a == a for a in alphas),
            "alpha_min": min(alphas), "alpha_max": max(alphas),
            "note": "apparently all-positive alpha; VERIFIED here, not enforced",
        },
        "sources": {
            "design_sha256": C.DESIGN_SHA, "design_bytes": C.DESIGN_BYTES,
            "join_source": str(C.JOINED.relative_to(C.ROOT)),
            "s5_raw_json": str(C.S5_RAW.relative_to(C.ROOT)),
            "batch22_sequence_table_sha256": C.sha256_file(C.SEQ_TABLE),
            "batch22_fold_assignment_sha256": C.sha256_file(C.FOLD_TABLE),
        },
        "model_assets": {
            "PARM_weights_sha256": {f"HEK293_fold{k}.parm":
                C.sha256_file(C.PARM_EXT / ("src/HEK293/HEK293_fold%d.parm" % k)) for k in range(5)},
            "PARM_L_max": C.PARM_L_MAX, "PARM_filter_size": C.PARM_FILTER,
            "PARM_type_loss": C.PARM_LOSS, "PARM_aggregation": "equal mean of 5 HEK293 folds",
            "H1_gram_sha256": C.sha256_file(C.GRAM_H1),
            "H1_weights_sha256": {f"attempt_fold{k}_H1/weights.npz":
                C.sha256_file(C.B24_MODELS / ("attempt_fold%d_H1/weights.npz" % k)) for k in range(5)},
            "H1_alpha_ridge": 1.0, "H1_aggregation": "equal mean of 5 folds",
            "GC_baseline": "GC fraction of the 270bp insert (higher GC -> higher activity, no fitted sign)",
            "length_baseline": "constant 270 -> rho UNDEFINED (not scored)",
        },
        "groups": {
            "definition": ("label-free overlapping-270bp-window components (step90 merge per chromosome) "
                           "+ merge of shared exact/RC inserts where actually recovered"),
            "n_components": len(comps), "chr17_components": n_chr17,
            "non_chr17_components": n_non17,
            "shared_exact_or_rc_merges_among_distinct_labels": merges,
            "not_independent_cre_families": True,
            "size_histogram": sizes_hist,
        },
        "dedup": {"unique_sequences": n_unique_seq, "max_unique_scored_per_model": n_unique_seq,
                  "note": "at most 7622 unique sequences scored once per model; dedup by sequence SHA"},
        "strata": {"chr17_bio": sum(1 for r in bio if r["chr"] == "chr17"),
                   "non_chr17_bio": sum(1 for r in bio if r["chr"] != "chr17")},
        "design_only_excluded": {"coordinate_biological_no_s5_truth": 1407,
                                 "scrambled_controls_no_s5_truth": 15,
                                 "role": "retained as counts only; excluded from performance"},
        "metric_plan": {
            "primary": "pooled 7537 finite bio Spearman(alpha, pred) for PARM, H1, GC; paired rho differences",
            "no_cross_unit": "no MAE/R2/log of PARM across output units; no calibration",
            "gc_null": "constant or undefined GC -> null with reason (not 0)",
            "top_decile": "k=ceil(0.1*n); deterministic ID tie-break; precision/enrichment vs same-alpha ranks; report true/pred ties",
            "strata_report": ["chr17 (expected 5868)", "non-chr17 (expected 1669)", "pooled"],
            "per_component": "rho for components with >=10 finite/nonconstant rows; macro-average + n_eligible + size histogram",
            "bootstrap": "paired 2000 component-resampling (with multiplicity) for rho/delta-rho/top10 if >=30 eligible components; CONDITIONAL uncertainty (residual regional dependence)",
            "loco": "leave-one-chromosome-out pooled rho sensitivity (NO FITS); no formal chromosome CI",
            "controls": "n=85 descriptive score-vs-alpha + ranked distribution only; not mixed into primary rho",
            "model_spread": "5-fold spread = disagreement, NOT a CI",
            "disclose": ["one MAPT regional concentration", "prior PARM/H1 training exposure UNKNOWN",
                         "retrospective transfer, not prospective or strict new family"],
        },
        "decision_rule": {
            "frozen": True,
            "supported_if_all": ["PARM pooled rho >= 0.30",
                                 "PARM minus H1 rho >= 0.05 with conditional paired-difference CI lo > 0",
                                 "PARM top10 precision >= H1 top10 precision",
                                 "PARM non-chr17 rho > 0 with no audit failure"],
            "else": "LIMITED (if H1 unsupported/CI undefined -> LIMITED)",
            "not_a_promotion": "operational resource decision only; does not replace old H1 MAE10% gate / local D28 FAIL",
            "no_rescue_by": ["strongest chromosome", "subset", "GC flip", "automatic finetuning"],
        },
        "readout_assets": {
            "parm": "official PARM HEK293 5 folds (restrictive license; local academic use; no redistribution)",
            "h1": "batch24 fixed spectrum-kernel ridge readouts (inference only; existing weights)",
            "gc": "no asset",
        },
        "frozen_input_table": str((C.DATA / "frozen_input_table.tsv").relative_to(C.ROOT)),
        "elapsed_s": round(time.time() - t0, 3),
    }
    C.atomic_write_json(C.REPORT / "diagnostic_manifest.json", manifest)
    print(json.dumps({"frozen": True, "rows": manifest["rows"], "components": len(comps),
                      "chr17_comp": n_chr17, "non17_comp": n_non17, "unique_seq": n_unique_seq,
                      "alpha_pos_all": manifest["labels"]["alpha_positive_all"]},
                     ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
