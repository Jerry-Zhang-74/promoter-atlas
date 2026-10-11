#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Promoter Atlas batch47 CLI -- v6.34.0-scoped-input-routing.rc1

A NEW version of the existing batch20 command routing (NOT another predictor, NOT a GUI).

Supported tasks (task must be chosen EXPLICITLY; the CLI never guesses):
  fragment       raw FASTA -> frozen PARM HEK293 model score (reuses the read-only b35 engine)
  gene-observed  gene symbol / Ensembl ID -> observed CATALOG labels (no model, no GPU)

Explicitly UNSUPPORTED in this entry (no silent switch): mpra-fasta, gene, gene-bag-fasta,
constructs, and any neural K562/old-164 / new-gene E/S/B inference. Ask the parent for a separate
scoped batch before using those.

All work here is ZERO-fit. The fragment path reuses the batch35 PARM engine READ-ONLY.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("PYTHONUTF8", "1")
os.environ.setdefault("PYTHONIOENCODING", "utf-8")
sys.path.insert(0, str(Path(__file__).resolve().parent))

from b47_common import BATCH, B47InputError, VERSION, write_json, write_tsv  # noqa: E402

UNSUPPORTED = ("mpra-fasta", "gene", "gene-bag-fasta", "constructs")
EXIT_INPUT = 2
EXIT_ASSET = 3
EXIT_BACKEND = 4


def _now():
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def _fail(msg, out_json=None, code=EXIT_INPUT):
    payload = {"status": "ERROR", "batch": BATCH, "version": VERSION, "generated_at": _now(),
               "error": msg}
    if out_json:
        try:
            write_json(out_json, payload)
        except Exception:
            pass
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return code


def cmd_fragment(args):
    import b47_fragment as FR
    try:
        res = FR.fragment_route(args.fasta, args.out_json, args.out_tsv,
                                asset_root=args.asset_root, target_cell=args.target_cell,
                                frozen_manifest=args.frozen_asset_manifest)
    except B47InputError as e:
        msg = str(e)
        code = EXIT_ASSET if msg.startswith(("ASSET_", "ENGINE_NOT_FOUND")) else (
            EXIT_BACKEND if msg.startswith("BACKEND_FORWARD_FAILED") else EXIT_INPUT)
        return _fail(msg, args.out_json, code)
    print(json.dumps({"status": "OK", "command": "fragment", "n_records": res["routing_summary"]["n_records"],
                      "n_scored": res["routing_summary"]["n_scored"],
                      "out_json": os.path.abspath(args.out_json), "out_tsv": os.path.abspath(args.out_tsv)},
                     ensure_ascii=False))
    return 0


def cmd_gene_observed(args):
    import b47_gene_observed as GO
    queries = []
    if args.query:
        for tok in args.query:
            queries.extend([q for q in tok.split(",")])
    if args.queryfile:
        p = Path(args.queryfile)
        if not p.exists():
            return _fail("QUERYFILE_NOT_FOUND: %s" % p, args.out_json, EXIT_INPUT)
        text = p.read_text(encoding="utf-8")
        if text.lstrip().startswith(">"):
            return _fail("GENE_INPUT_LOOKS_LIKE_FASTA: queryfile is a FASTA; gene-observed expects gene "
                         "symbols/IDs. Use the fragment command for FASTA/DNA.", args.out_json, EXIT_INPUT)
        for line in text.splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            queries.extend([q for q in line.split(",")])
    if not queries:
        return _fail("NO_QUERIES: provide --query or --queryfile", args.out_json, EXIT_INPUT)
    try:
        contract = GO.load_contract()
    except B47InputError as e:
        return _fail(str(e), args.out_json, EXIT_INPUT)
    records = []
    for i, q in enumerate(queries):
        status, reason, base, sym, cands = GO.resolve(contract, q)
        records.append(GO.observed_record(contract, i, q, status, reason, base, sym, cands))
    n_ok = sum(1 for r in records if r["status"] == "OK")
    n_absent = sum(1 for r in records if r["status"] == "ABSENT")
    n_amb = sum(1 for r in records if r["status"] == "AMBIGUOUS")
    n_err = sum(1 for r in records if r["status"] == "INVALID_QUERY")
    result = {
        "status": "OK", "batch": BATCH, "version": VERSION, "command": "gene-observed",
        "task": "gene_observed_catalog_lookup", "generated_at": _now(),
        "note": ("Catalog-derived labels from the traceable label table + symbol map. These are NOT an "
                 "individual promoter measurement, NOT a GFP / 293FT activity, and NOT a model prediction. "
                 "S_raw is CROSS-TISSUE stability, not time; missing S is null, never 0 or a prediction."),
        "sources": GO.source_sha(),
        "source_field_definitions": GO.FIELD_DEFINITIONS,
        "masks": {"S_raw_defined_mask": "S_defined_main==1 (is_in_main_subset==1 AND detectable_tissue_count>=3)",
                  "is_in_main_subset": "B_raw >= 0.30"},
        "n_queries": len(queries), "n_ok": n_ok, "n_absent": n_absent,
        "n_ambiguous": n_amb, "n_invalid_query": n_err,
        "ordered": True,
        "records": records,
        "warnings": [{"level": "warn", "text": "Catalog-derived observed labels are NOT model predictions and "
                      "NOT 293FT/GFP activity; no neural model, no GPU, no learned weights were used."}],
    }
    write_json(args.out_json, result)
    cols = ["index", "query", "status", "reason", "canonical_gene_id", "symbol", "observedScore_E",
            "S_raw", "S_raw_defined", "B_raw", "observedScore_S", "observedScore_B", "is_in_main_subset", "note"]
    write_tsv(args.out_tsv, records, cols)
    print(json.dumps({"status": "OK", "command": "gene-observed", "n_queries": len(queries), "n_ok": n_ok,
                      "n_absent": n_absent, "n_ambiguous": n_amb, "n_invalid_query": n_err,
                      "out_json": os.path.abspath(args.out_json), "out_tsv": os.path.abspath(args.out_tsv)},
                     ensure_ascii=False))
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(prog="b47_cli.py",
                                 description="Promoter Atlas batch47 scoped input routing %s" % VERSION)
    sub = ap.add_subparsers(dest="command", required=True)

    p1 = sub.add_parser("fragment", help="raw FASTA -> frozen PARM HEK293 model score (reuses read-only b35 engine)")
    p1.add_argument("--fasta", required=True, help="input FASTA (explicit fragment task)")
    p1.add_argument("--out-json", required=True)
    p1.add_argument("--out-tsv", required=True)
    p1.add_argument("--asset-root", default=None, help="read-only PARM external asset root (default b31/_external)")
    p1.add_argument("--frozen-asset-manifest", default=None, help="frozen asset SHA manifest (default b35)")
    p1.add_argument("--target-cell", default="293FT", help="caller target-cell METADATA only")
    p1.set_defaults(func=cmd_fragment)

    p2 = sub.add_parser("gene-observed", help="gene symbol / Ensembl ID -> observed CATALOG labels (no model/GPU)")
    p2.add_argument("--query", nargs="+", default=None, help="one or more gene symbols / Ensembl IDs")
    p2.add_argument("--queryfile", default=None, help="file of queries (one per line or comma separated)")
    p2.add_argument("--out-json", required=True)
    p2.add_argument("--out-tsv", required=True)
    p2.set_defaults(func=cmd_gene_observed)

    args = ap.parse_args(argv)
    if args.command in UNSUPPORTED:
        return _fail("UNSUPPORTED_TASK:%s: not implemented in this narrow entry; no silent switch. "
                     "This batch supports only: fragment, gene-observed." % args.command)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
