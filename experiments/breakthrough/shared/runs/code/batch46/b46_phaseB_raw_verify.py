# -*- coding: utf-8 -*-
"""batch46 Phase B: write per-fragment fixed summaries and INDEPENDENTLY verify 20 raw
sequence/replicate records against them (own code, own seed 20261004). No new normalization."""
import sys, json, math, random, hashlib
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import b46_common as C
import b46_tables as T

NORM = {"K562_r1": "expression.normalized.log2rpm.K562.replicate1",
        "K562_r2": "expression.normalized.log2rpm.K562.replicate2",
        "HepG2_r1": "expression.normalized.log2rpm.HepG2.replicate1",
        "HepG2_r2": "expression.normalized.log2rpm.HepG2.replicate2"}


def write_per_fragment(frags):
    hdr = ["info", "library", "lib", "category", "q_len", "GC", "n_barcodes",
           "pred_K562", "pred_HepG2",
           "K562_r1_mean", "K562_r2_mean", "K562_meas",
           "HepG2_r1_mean", "HepG2_r2_mean", "HepG2_meas",
           "K562_n_nonfinite", "HepG2_n_nonfinite"]
    with open(C.OUT / "per_fragment.tsv", "w", encoding="utf-8") as fh:
        fh.write("\t".join(hdr) + "\n")
        for seq, f in sorted(frags.items()):
            fh.write("\t".join("" if f.get(h) is None else str(f.get(h)) for h in hdr) + "\n")


def main():
    rows = C.parse()
    frags = T.build_fragments()
    write_per_fragment(frags)

    rng = random.Random(20261004)
    idx = sorted(rng.sample(range(len(rows)), 20))
    # force-include the known -Inf dropout record for honesty
    drop = [i for i, r in enumerate(rows) if r[C.CI["info"]] ==
            "K562_fitness_232bp_replicate01_generation0_individual192_median"
            and r[C.CI["barcode"]] == "TTGGATGCGACGA"]
    if drop and drop[0] not in idx:
        idx[0] = drop[0]
    checks = []
    all_ok = True
    for i in idx:
        r = rows[i]
        seq = r[C.CI["query.seq"]]
        f = frags[seq]
        offset_ok = seq in r[C.CI["full.oligo"]] and r[C.CI["full.oligo"]].index(seq) == 18
        rec = {"row_index": i, "barcode": r[C.CI["barcode"]], "info": r[C.CI["info"]],
               "query_length": len(seq),
               "seq_is_exact_substring_offset18": bool(offset_ok),
               "fragment_summary": {k: f[k + "_mean"] for k in ["K562_r1", "K562_r2", "HepG2_r1", "HepG2_r2"]},
               "raw_normalized_row": {k: r[C.CI[v]] for k, v in NORM.items()},
               "fragment_meas_K562": f["K562_meas"], "fragment_meas_HepG2": f["HepG2_meas"],
               "fragment_GLOBAL_mean_K562": None, "fragment_GLOBAL_mean_HepG2": None}
        # recompute the FRAGMENT means independently from raw rows (fresh pass)
        bros = [x for x in rows if x[C.CI["query.seq"]] == seq]
        for cell, (c1, c2) in {"K562": (NORM["K562_r1"], NORM["K562_r2"]),
                               "HepG2": (NORM["HepG2_r1"], NORM["HepG2_r2"])}.items():
            vals1 = [C.to_float(x[C.CI[c1]]) for x in bros]
            vals2 = [C.to_float(x[C.CI[c2]]) for x in bros]
            m = C.mean_finite([C.mean_finite(vals1), C.mean_finite(vals2)])
            rec["fragment_GLOBAL_mean_" + cell] = m
            if (m is None) != (f[cell + "_meas"] is None) or (m is not None and
                                                              abs(m - f[cell + "_meas"]) > 1e-12):
                all_ok = False
        if not offset_ok:
            all_ok = False
        checks.append(rec)
    C.write_json(C.OUT / "raw20_verification.json", {
        "batch": C.BATCH, "version": C.VERSION, "phase": "B_raw_verify",
        "seed": 20261004, "n_records": len(checks),
        "n_barcodes_per_fragment": 4,
        "verified_all_fragment_means_match": all_ok,
        "records": checks})
    print("raw20 verification all_ok", all_ok, "n", len(checks))


if __name__ == "__main__":
    main()
