#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Independent bounded cross-check of the 133 bp measured sequences vs UCSC hg19.

For a deterministic sample of the joined rows, fetch the hg19 reference window
[oligo_start0, end) and confirm the published sequence equals the hg19 FORWARD
strand for '+' rows and the REVERSE-COMPLEMENT of the hg19 forward strand for '-'
rows. This is a bounded API check (<=48 windows), NOT a genome download, and it is
recorded as diagnostics only (it never changes the published sequences used).
"""
from __future__ import annotations

import csv
import json
import urllib.request

from b22_common import DATA_OUT, REPORT_OUT, atomic_write_json, now_local, rc

N_PER_STRAND = 12


def ucsc(genome, chrom, start, end, timeout=40):
    u = ("https://api.genome.ucsc.edu/getData/sequence?genome=%s;chrom=%s;start=%d;end=%d"
         % (genome, chrom, start, end))
    with urllib.request.urlopen(u, timeout=timeout) as r:
        return json.load(r)["dna"].upper()


def run():
    started = now_local()
    rows = list(csv.DictReader(open(DATA_OUT / "sequence_table.tsv", encoding="utf-8"),
                               delimiter="\t"))
    plus = [r for r in rows if r["strand"] == "+"][:N_PER_STRAND]
    minus = [r for r in rows if r["strand"] == "-"][:N_PER_STRAND]
    checks = []
    ok = 0
    for r in plus + minus:
        s0 = int(r["start1"]) - 1
        e0 = int(r["end"])
        try:
            d = ucsc("hg19", r["chr"], s0, e0)
            fwd = (d == r["sequence"].upper())
            rev = (rc(d) == r["sequence"].upper())
            match = fwd if r["strand"] == "+" else rev
        except Exception as e:  # noqa: BLE001
            fwd = rev = match = None
            d = "ERROR:%r" % (e,)
        ok += int(bool(match))
        checks.append({"prom_id": r["prom_id"], "strand": r["strand"], "chr": r["chr"],
                       "start0": s0, "end": e0, "len_ref": len(d) if d.startswith("ACGT") else 0,
                       "hg19_forward_match": fwd, "hg19_revcomp_match": rev,
                       "expected_match": bool(match)})
    out = {"generated_at_local": now_local(), "started_local": started,
           "genome": "hg19 (UCSC)", "n_checked": len(checks), "n_match": ok,
           "n_mismatch": len(checks) - ok, "checks": checks,
           "note": "diagnostic cross-check only; published sequences used verbatim"}
    atomic_write_json(REPORT_OUT / "source_sequence_verification.json", out)
    print("[b22_verify_source] checked=%d match=%d mismatch=%d"
          % (len(checks), ok, len(checks) - ok))
    return out


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
    run()
