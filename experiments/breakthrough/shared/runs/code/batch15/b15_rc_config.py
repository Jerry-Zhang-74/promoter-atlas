# -*- coding: utf-8 -*-
"""BATCH15 rc-retrieval-audit: freeze source SHAs + mapping + config + tested
retrieval algorithm BEFORE the production cross-orientation search. Read-only
over prior data; writes only config.json (and a small forward-reference ledger).
"""
import sys, json, csv
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import b15_rc_common as C
from b15_rc_common import (sha256_file, atomic_write_json, now_local, worker_identity,
                           MEMMAP, DATA13, DATA05, DATA03, DATA15, CODE15)

def main():
    DATA15.mkdir(parents=True, exist_ok=True)

    sources = {
        "promoter_sym400_uint8_memmap": str(MEMMAP),
        "trackA_gene_promoters_npz": str(DATA03 / "trackA_gene_promoters.npz"),
        "batch13_eligible_genes_tsv": str(DATA13 / "eligible_genes.tsv"),
        "batch13_fold_assignment_tsv": str(DATA13 / "fold_assignment.tsv"),
        "batch05_components_tsv": str(DATA05 / "components.tsv"),
        "batch05_near_identity_pairs_tsv": str(DATA05 / "near_identity_pairs.tsv"),
        "batch05_split_arrays_npz": str(DATA05 / "split_arrays.npz"),
    }
    code_files = {
        "b15_rc_common_py": str(CODE15 / "b15_rc_common.py"),
        "b15_rc_synthetic_py": str(CODE15 / "b15_rc_synthetic.py"),
        "b15_rc_audit_py": str(CODE15 / "b15_rc_audit.py"),
        "b15_rc_config_py": str(CODE15 / "b15_rc_config.py"),
    }
    sources_sha = {k: {"path": v, "sha256": sha256_file(v)} for k, v in sources.items()}
    code_sha = {k: {"path": v, "sha256": sha256_file(v)} for k, v in code_files.items()}

    # forward-reference (descriptive; same source used for the 413 / 0 numbers)
    el = {}
    with open(DATA13 / "eligible_genes.tsv", encoding="utf-8", newline="") as f:
        r = csv.reader(f, delimiter="\t"); next(r)
        for row in r:
            el[row[0]] = int(row[2])
    n_fwd_rows = 0; n_rc_rows = 0
    fwd_cross_gene = 0; fwd_cross_comp = 0; seen = set()
    with open(DATA05 / "near_identity_pairs.tsv", encoding="utf-8", newline="") as f:
        r = csv.reader(f, delimiter="\t"); next(r)
        for row in r:
            ga, gb, ori = row[2], row[3], row[6]
            if ori == "rc":
                n_rc_rows += 1
            else:
                n_fwd_rows += 1
            if ga in el and gb in el and ga != gb:
                k = tuple(sorted((ga, gb)))
                if k in seen:
                    continue
                seen.add(k)
                fwd_cross_gene += 1
                if el[ga] != el[gb]:
                    fwd_cross_comp += 1

    cfg = {
        "report_version": "rc-retrieval-audit-20261003-r1",
        "generated_at": now_local(),
        "worker_identity": worker_identity(),
        "question": ("does an explicit cross-orientation (FORWARD-index vs REVERSE-COMPLEMENT-query) "
                     "retrieval find eligible near-RC promoter pairs connecting different batch13 components?"),
        "params": {"k": C.K, "n_hash": C.N_HASH, "bands": C.N_BANDS, "rows": C.ROWS,
                   "seed": C.SEED, "min_bands": C.MIN_BANDS, "band_cap": C.BAND_CAP,
                   "pair_cap": C.PAIR_CAP, "max_shift": C.MAX_SHIFT,
                   "identity_min": C.IDENTITY_MIN, "coverage_min": C.COVERAGE_MIN,
                   "n_human_rows": C.N_HUMAN, "seq_len": C.L400},
        "retrieval_algorithm_frozen": {
            "description": ("compute forward sketch F=compute_sketch(seq,a,b) and reverse-complement "
                            "sketch R=compute_sketch(rc(seq),a,b) with the SAME hash seed/functions; "
                            "cross-orientation candidate retrieval = band_matches(F_sketch, R_sketch) "
                            "(FORWARD index vs REVERSE-COMPLEMENT query); a candidate pair (i,j) survives "
                            "iff >=2 distinct bands have EQUAL actual 4-value minhash tuples (not XOR-only); "
                            "both orientations are separate records and unordered promoter pairs are "
                            "deduplicated only AFTER cross-orientation candidate collection."),
            "function": "band_matches(F_sketch, R_sketch, band_cap=2000, pair_cap=5000000, min_bands=2)",
            "verification": ("exact deterministic gapless RC alignment over shifts [-40,40]; "
                             "identity=matches/overlap_len (N never a match); coverage=overlap_len/400; "
                             "confirmed iff identity>=0.90 AND coverage>=0.90 (overlap>=360)."),
            "excluded": ["training", "post-hoc metric optimization", "reading S_raw_full for retrieval",
                         "reverse-complementing every sequence and comparing RC to RC (does not address "
                         "opposite-orientation pairs and is NOT what this audit does)"],
        },
        "mapping": {
            "authoritative_gene_mapping": "trackA_gene_promoters.npz: gene_id -> r400 memmap rows (NOT gene_id[gene_index] positional)",
            "eligible_gene_source": "batch13/eligible_genes.tsv (18769 genes; component_id from batch05 split_arrays comp_dense)",
            "fold_source": "batch13/fold_assignment.tsv",
            "promoter_order": "eligible gene file order, then ascending r400 rows within gene",
        },
        "dedup_policy": ("promoter pairs deduplicated as unordered (row_i, row_j); gene pairs deduplicated "
                         "as unordered (gene_a, gene_b); multiple promoter pairs per gene pair retained as "
                         "evidence; cross-gene only (gene_a != gene_b)"),
        "forward_reference": {
            "near_identity_pairs_total_rows": n_fwd_rows + n_rc_rows,
            "n_forward_orientation_rows": n_fwd_rows,
            "n_rc_orientation_rows": n_rc_rows,
            "n_eligible_cross_gene_edges": fwd_cross_gene,
            "n_eligible_cross_component_edges": fwd_cross_comp,
            "note": ("batch05 saved channel contains ONLY forward-orientation candidate pairs (0 rc rows); "
                     "this is the exact gap the cross-orientation audit is checking, not a causal comparison."),
        },
        "sources_sha256": sources_sha,
        "code_sha256": code_sha,
    }
    atomic_write_json(DATA15 / "config.json", cfg)
    print(json.dumps({"config_written": str(DATA15 / "config.json"),
                      "forward_reference": cfg["forward_reference"],
                      "sources_sha256_ok": True}, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
