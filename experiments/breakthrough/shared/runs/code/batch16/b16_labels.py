#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Batch16 label assembly: E/S/B targets + masks + promoter-level aux labels.

Deterministic reconstruction from original expression (full precision raw S),
canonical E/B definitions, and observed epigenetic/accessibility assays.

Writes (data/breakthrough_20261002/batch16/):
  labels.npz           gene-level E/S/B + masks + omics aux matrix (promoter-level)
  label_manifest.json  counts, masks, SHAs, provenance
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from b16_common import (ROOT, DATA_OUT, EXPR, CANON, TRACKA_NPZ, OMICS_TSV,
                        N_FEATURE_UNIVERSE, AUX_FIELDS, DETECTABLE_TPM, B_MIN,
                        MIN_DET_TISSUES, reconstruct_targets, load_gene_universe,
                        load_split, sha256_file, atomic_write_json)

N_EXPECTED_S_ELIGIBLE = 18769


def main():
    t0 = time.time()
    DATA_OUT.mkdir(parents=True, exist_ok=True)

    gene_gid, record_id, r400, score_e, starts = load_gene_universe()
    gene_split, new_comp_dense = load_split()
    targets = reconstruct_targets(EXPR)
    b_raw_map, s_raw_map = targets["b_raw"], targets["s_raw"]

    # ---- E target: Score_E (gene-level, from trackA, rank-scaled [0,1]) ----
    E = score_e[starts[:-1]].astype(np.float64)
    assert E.shape == (N_FEATURE_UNIVERSE,)
    E_mask = np.ones(N_FEATURE_UNIVERSE, dtype=bool)
    assert np.isfinite(E).all()

    # ---- B target: cross-tissue breadth B_raw ----
    B = np.array([b_raw_map.get(g, np.nan) for g in gene_gid], dtype=np.float64)
    B_mask = np.isfinite(B)
    n_b_missing = int((~B_mask).sum())

    # ---- S target: raw S (negative MAD) with S_defined mask ----
    S = np.array([s_raw_map.get(g, np.nan) for g in gene_gid], dtype=np.float64)
    S_mask = np.isfinite(S)
    n_s_eligible = int(S_mask.sum())
    assert n_s_eligible == N_EXPECTED_S_ELIGIBLE, "S eligible count %d != %d" % (n_s_eligible, N_EXPECTED_S_ELIGIBLE)

    # ---- promoter-level aux labels (7 observed epigenetic/accessibility fields) ----
    om = pd.read_csv(OMICS_TSV, sep="\t", dtype={"record_id": str})
    om = om.set_index("record_id")
    aux_fields = [f for f in AUX_FIELDS if f in om.columns]
    assert len(aux_fields) == len(AUX_FIELDS), "aux field mismatch"
    # map trackA record_id -> omics rows (all 436291 present, verified)
    om_lookup = om.loc[record_id, aux_fields].to_numpy(dtype=np.float64)
    assert om_lookup.shape == (len(record_id), len(aux_fields))
    # aux missingness: values are observed (0.0 = observed absence); no NaN in source
    aux_nan = int(np.isnan(om_lookup).sum())
    assert aux_nan == 0, "unexpected NaN in aux labels"

    # ---- persist ----
    np.savez_compressed(
        DATA_OUT / "labels.npz",
        gene_gid=gene_gid,
        record_id=record_id,
        r400=r400,
        starts=starts,
        gene_split=gene_split,
        new_comp_dense=new_comp_dense,
        E=E,
        S=S,
        B=B,
        E_mask=E_mask,
        S_mask=S_mask,
        B_mask=B_mask,
        aux=om_lookup,
        aux_fields=np.array(aux_fields),
    )

    manifest = {
        "candidate_version": "v6.15.0-multitask.rc1",
        "batch": "batch16",
        "written_at_local": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "elapsed_s": round(time.time() - t0, 2),
        "targets": {
            "E": {"definition": "Score_E rank-scaled [0,1] gene-level expression strength",
                  "mask": "all feature-universe genes",
                  "n_defined": int(E_mask.sum())},
            "S": {"definition": "S_raw = -MAD(log2(1+detectable TPM)) over detectable tissues",
                  "mask": "S_defined (B_raw>=0.30 AND detectable>=3) AND feature-covered",
                  "n_defined": int(S_mask.sum()),
                  "expected": N_EXPECTED_S_ELIGIBLE,
                  "DETECTABLE_TPM": DETECTABLE_TPM, "B_MIN": B_MIN, "MIN_DET_TISSUES": MIN_DET_TISSUES},
            "B": {"definition": "B_raw = detectable_tissue_count / tissue_count",
                  "mask": "B_raw defined (non-NaN)",
                  "n_defined": int(B_mask.sum()),
                  "n_missing": n_b_missing},
        },
        "aux": {
            "fields": aux_fields,
            "n_fields": len(aux_fields),
            "source": str(OMICS_TSV),
            "level": "promoter (trackA record_id)",
            "excluded": ["cage_tss", "procap_tss"],
            "excluded_reason": "transcription-initiation (TSS-activity) proxies of the E target",
            "missingness": "0.0 = observed absence; no NaN in source",
            "n_nan": aux_nan,
            "n_promoters_covered": int(len(record_id)),
        },
        "source_hashes": {
            "gene_expression_table.tsv": sha256_file(EXPR),
            "gene_expression_label_table.tsv": sha256_file(CANON),
            "trackA_gene_promoters.npz": sha256_file(TRACKA_NPZ),
            "promoter_omics_targets_v1.tsv": sha256_file(OMICS_TSV),
        },
    }
    atomic_write_json(DATA_OUT / "label_manifest.json", manifest)

    print(json.dumps({
        "n_genes": int(N_FEATURE_UNIVERSE),
        "E_defined": int(E_mask.sum()),
        "S_defined": int(S_mask.sum()),
        "B_defined": int(B_mask.sum()),
        "B_missing": n_b_missing,
        "aux_fields": aux_fields,
        "aux_nan": aux_nan,
        "labels_npz": str(DATA_OUT / "labels.npz"),
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
