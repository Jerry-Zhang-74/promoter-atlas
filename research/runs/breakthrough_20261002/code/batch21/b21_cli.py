#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Batch21 minimal supported raw-FASTA entrypoint.

Usage:
  b21_cli.py predict-fasta --fasta in.fasta [--out out.tsv] [--json out.json]
  b21_cli.py constructs

predict-fasta: strict raw-sequence prediction on the K562 164 bp legnet branch.
  * source-catalog rows are marked in_source and get an honest fold-aware OOF;
  * rows outside the prior CV cohort get NO fabricated OOF;
  * genuinely new rows get the 15-model mean labelled out_of_source;
  * inputs violating the declared quality rules are refused row-by-row (exit 0)
    while a hard file/argument error exits 1 with a one-line message (no traceback).
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

from b21_common import (B21InputError, read_fasta, REPORT_OUT, DATA_OUT, atomic_write_json,
                        now_local, VERSION)
import b21_input


def cmd_predict(args):
    recs = read_fasta(args.fasta)
    rows, refused = b21_input.predict_records(recs, compute_nn=(not args.no_nn))
    out_cols = ["name", "sequence_sha256", "length", "n_fraction", "in_source_catalog",
                "in_prior_cv_cohort", "prediction_type", "prediction", "oof_available",
                "source_similarity", "source_similarity_tier", "domain", "robustness",
                "min_hamming_to_catalog", "nearest_catalog_oligo", "prior_cv_fold",
                "prior_component_id", "source_observed_activity", "prediction_std_over_models",
                "note"]
    out_tsv = Path(args.out) if args.out else (DATA_OUT / "raw_fasta_predictions.tsv")
    with open(out_tsv, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=out_cols, delimiter="\t", extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)
    summary = {"generated_at_local": now_local(), "version": VERSION, "input": str(args.fasta),
               "n_records": len(recs), "n_predicted": len(rows), "n_refused": len(refused),
               "rows": rows, "refused": refused, "out_tsv": str(out_tsv),
               "note": "K562 164 bp MPRA branch only; NOT HEK293T. Robustness UNCONFIRMED. "
                       "In-source rows report fold-aware OOF; new rows report the 15-model mean."}
    if args.json:
        atomic_write_json(Path(args.json), summary)
    print("predicted %d / %d records (%d refused)" % (len(rows), len(recs), len(refused)))
    for r in rows:
        print("  %-24s %-28s pred=%s %s" % (r["name"], r["prediction_type"],
              ("%.4f" % r["prediction"]) if r.get("prediction") is not None else "NA",
              r.get("source_similarity_tier", "")))
    for r in refused:
        print("  REFUSED %-20s %s" % (r["name"], r["reason"]))
    return 0


def cmd_constructs(args):
    p = REPORT_OUT / "construct_audit.tsv"
    if not p.exists():
        print("construct audit not built yet: run b21_source.py", file=sys.stderr)
        return 1
    with open(p, encoding="utf-8") as fh:
        for line in fh:
            print(line.rstrip())
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(prog="b21_cli")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p1 = sub.add_parser("predict-fasta")
    p1.add_argument("--fasta", required=True)
    p1.add_argument("--out", default=None)
    p1.add_argument("--json", default=None)
    p1.add_argument("--no-nn", action="store_true")
    p1.set_defaults(func=cmd_predict)
    p2 = sub.add_parser("constructs")
    p2.set_defaults(func=cmd_constructs)
    args = ap.parse_args(argv)
    try:
        return args.func(args)
    except (B21InputError, OSError, FileNotFoundError) as e:
        print("error: %s" % e, file=sys.stderr)   # F2 repair: no traceback
        return 1


if __name__ == "__main__":
    sys.exit(main())
