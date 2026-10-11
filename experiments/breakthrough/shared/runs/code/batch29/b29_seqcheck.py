# -*- coding: utf-8 -*-
"""BATCH29 independent sequence/strand QC: second source (UCSC hg19 REST) vs Ensembl GRCh37 for
>=20 intervals on BOTH strands, N-base counts, length match, exact/RC duplicate pairs, RC-once check."""
import json, time, urllib.request, random
from pathlib import Path

ROOT = Path("E:/promoter_atlas"); DB = ROOT / "data/breakthrough_20261002/batch29"
REP = ROOT / "reports/breakthrough_20261002/batch29"
sel = json.load(open(DB / "selected_rows.json", encoding="utf-8"))["rows"]

def load_fa(p):
    d = {}; k = None
    for line in open(p, encoding="utf-8"):
        line = line.strip()
        if not line: continue
        if line.startswith(">"):
            p2 = line[1:].split("_"); k = "%s:%s:%s" % (p2[0], p2[1], p2[2])
        else: d[k] = line
    return d
fwd = load_fa(DB / "sure29_forward.fa"); assay = load_fa(DB / "sure29_assay_oriented.fa")
RC = str.maketrans("ACGT", "TGCA")

def ucsc(c, st, en):
    url = "https://api.genome.ucsc.edu/getData/sequence?genome=hg19;chrom=%s;start=%d;end=%d" % (c, st, en)
    r = urllib.request.urlopen(url, timeout=30).read()
    return json.loads(r.decode())["dna"].upper()

plus = [r for r in sel if r["strand"] == "+"]
minus = [r for r in sel if r["strand"] == "-"]
random.seed(20261003)
samp = random.sample(plus, 25) + random.sample(minus, 25)
res = {"n_plus_sampled": 25, "n_minus_sampled": 25, "matches_ucsc": 0, "mismatches_ucsc": 0,
       "rc_once_ok_plus": 0, "rc_once_ok_minus": 0, "details": []}
for r in samp:
    k = r["key"]; c, st, en = r["chrom"], r["start"], r["end"]
    try:
        u = ucsc(c, st, en)
    except Exception as e:
        res["details"].append({"key": k, "err": repr(e)[:80]}); continue
    e_ = fwd[k]
    match = (u == e_)
    res["matches_ucsc" if match else "mismatches_ucsc"] += 1
    if r["strand"] == "+":
        if assay[k] == e_: res["rc_once_ok_plus"] += 1
    else:
        if assay[k] == e_.translate(RC)[::-1]: res["rc_once_ok_minus"] += 1
    if len(res["details"]) < 6:
        res["details"].append({"key": k, "ensembl_len": len(e_), "ucsc_len": len(u), "match": match,
                               "strand": r["strand"]})

# whole-cohort QC
n_N = sum(1 for k, s in assay.items() if any(ch not in "ACGT" for ch in s))
lens = [len(s) for s in assay.values()]
seen = {}; rcseen = {}; n_exact_dup = 0; n_rc_dup = 0
for k, s in assay.items():
    if s in seen: n_exact_dup += 1
    else: seen[s] = k
    rr = s.translate(RC)[::-1]
    if rr in rcseen: n_rc_dup += 1
    else: rcseen[rr] = k
res.update({"n_usable_assay": len(assay), "n_containing_N": n_N,
            "len_min": min(lens), "len_max": max(lens), "len_mean": sum(lens) / len(lens),
            "n_exact_dup_pairs": n_exact_dup, "n_rc_dup_pairs": n_rc_dup,
            "written_at": time.strftime("%Y-%m-%dT%H:%M:%S%z")})
json.dump(res, open(REP / "seqcheck.json", "w", encoding="utf-8"), indent=2)
print(json.dumps(res, indent=2))
