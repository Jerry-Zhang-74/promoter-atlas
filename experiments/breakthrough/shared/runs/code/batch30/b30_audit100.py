# -*- coding: utf-8 -*-
"""batch30 MANDATORY 100-row source audit (ZERO fits).  Own independent parser.

Exactly 100 raw rows of the frozen SuRE gz, chosen DETERMINISTICALLY from the cached 9013
rows by sha256(key) ranking WITHIN each strand (50 '+' and 50 '-'), i.e. coordinate/strand
selection independent of response.  The original gz is reread once by this module's own
parser; counts P/r1/r2 and the target y are recomputed and compared to the cached rows.
Saved immutable BEFORE any future fit, with source/data/code SHA and actual clock.
Library offsets N_D/N_R1/N_R2 are reused from the batch29 independent full-table recount
(passA.json sha), per protocol (no repeat costly hash/recount).
"""
from __future__ import annotations
import gzip
import json
import math
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import b30_common as C

TOL = 1e-9


def main():
    t0 = time.time()
    rows = []
    with open(C.REP29 / "split_manifest.tsv", encoding="utf-8") as f:
        next(f)
        for line in f:
            rows.append(line.rstrip("\n").split("\t"))   # key,chrom,start,end,strand,group,split
    cached = {}
    with open(C.DATA29 / "rows.tsv", encoding="utf-8") as f:
        hdr = f.readline().rstrip("\n").split("\t")
        for line in f:
            d = dict(zip(hdr, line.rstrip("\n").split("\t")))
            cached[d["key"]] = d
    passA = C.read_json(C.REP29 / "passA.json")
    N_D = passA["library_totals_fulltable"]["N_D_1plusP"]
    N_R1 = passA["library_totals_fulltable"]["N_R1"]
    N_R2 = passA["library_totals_fulltable"]["N_R2"]

    plus = [r for r in rows if r[4] == "+"]
    minus = [r for r in rows if r[4] == "-"]
    plus.sort(key=lambda r: C.sha256_text(r[0]))
    minus.sort(key=lambda r: C.sha256_text(r[0]))
    sel_keys = [r[0] for r in plus[:50]] + [r[0] for r in minus[:50]]
    target = set(sel_keys)
    needed_chroms = set(r[1] for r in rows if r[0] in target)
    prefixes = tuple((c + "\t").encode() for c in sorted(needed_chroms))

    found = {}
    n_read = 0
    with gzip.open(C.SOURCE_GZ, "rb") as f:
        f.readline()                                   # header
        for raw in f:
            n_read += 1
            if not raw.startswith(prefixes):
                continue
            p = raw.rstrip(b"\n").split(b"\t")
            if len(p) != 13:
                continue
            key = "%s:%s-%s:%s" % (p[0].decode(), p[1].decode(), p[2].decode(), p[3].decode())
            if key in target:
                found[key] = p

    def num(b):
        return float(b.decode())

    recs = []
    present = 0
    maxdp = maxdr1 = maxdr2 = maxdy = 0.0
    for k in sel_keys:
        c = cached[k]
        p = found.get(k)
        if p is None:
            recs.append({"key": k, "found": False})
            continue
        present += 1
        r1 = sum(num(p[i]) for i in (5, 6, 7, 8))
        r2 = num(p[9]) + num(p[10])
        P = num(p[11]) + num(p[12])
        D = 1.0 + P
        a1 = (r1 / N_R1) / (D / N_D)
        a2 = (r2 / N_R2) / (D / N_D)
        y = 0.5 * (math.log2(1.0 + a1) + math.log2(1.0 + a2))
        dp = abs(P - float(c["P"])); dr1 = abs(r1 - float(c["r1"])); dr2 = abs(r2 - float(c["r2"]))
        dy = abs(y - float(c["y"]))
        maxdp = max(maxdp, dp); maxdr1 = max(maxdr1, dr1); maxdr2 = max(maxdr2, dr2); maxdy = max(maxdy, dy)
        recs.append({"key": k, "found": True, "strand": k[-1], "P": P, "r1": r1, "r2": r2, "y": y,
                     "cached_P": float(c["P"]), "cached_r1": float(c["r1"]), "cached_r2": float(c["r2"]),
                     "cached_y": float(c["y"]), "dP": dp, "dr1": dr1, "dr2": dr2, "dy": dy})

    all_ok = (present == 100) and (maxdp <= TOL) and (maxdr1 <= TOL) and (maxdr2 <= TOL) and (maxdy <= TOL)
    out = {"batch": "batch30", "kind": "mandatory_100_row_source_audit",
           "n_selected": 100, "n_plus": 50, "n_minus": 50,
           "selection_rule": "50(+) and 50(-) with smallest sha256('chrom:start-end:strand') among cached 9013; response-independent",
           "n_source_rows_read": n_read, "n_present": present, "n_missing": 100 - present,
           "tolerance": TOL, "max_abs_dP": maxdp, "max_abs_dr1": maxdr1, "max_abs_dr2": maxdr2, "max_abs_dy": maxdy,
           "all_100_present_and_exact": bool(all_ok),
           "library_offsets_reused": {"N_D": N_D, "N_R1": N_R1, "N_R2": N_R2,
                                      "passA_sha256": C.sha256_file(C.REP29 / "passA.json"),
                                      "note": "reused from batch29 independent full-table recount"},
           "source_gz": str(C.SOURCE_GZ), "source_sha256": C.SOURCE_SHA,
           "data_rows_tsv_sha256": C.sha256_file(C.DATA29 / "rows.tsv"),
           "split_manifest_sha256": C.sha256_file(C.REP29 / "split_manifest.tsv"),
           "code_sha256": C.sha256_file(Path(__file__)),
           "written_at": C.now_local(), "elapsed_s": round(time.time() - t0, 1),
           "rows": recs}
    C.atomic_write_json(C.REP30 / "audit100.json", out)
    print(json.dumps({k: v for k, v in out.items() if k != "rows"}, indent=1))


if __name__ == "__main__":
    main()
