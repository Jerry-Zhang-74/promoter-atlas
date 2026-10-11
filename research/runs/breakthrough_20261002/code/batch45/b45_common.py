# -*- coding: utf-8 -*-
"""batch45 v6.32.0-synthetic-design-utility.rc1 -- shared parsing/freeze helpers.
ZERO model fit/GPU/new prediction.  Read-only on batch44 source; writes only under
reports/breakthrough_20261002/batch45 and data/breakthrough_20261002/batch45.
"""
import gzip, hashlib, math, re, collections, json
from pathlib import Path

ROOT = Path(r"E:/promoter_atlas")
SRC = ROOT / "data/breakthrough_20261002/batch44/syn.txt.gz"
PARM_XML = ROOT / "data/breakthrough_20261002/batch44/parm_fulltext.xml"
GEO_LIST = ROOT / "data/breakthrough_20261002/batch44/gse301246_suppl.html"
OUT = ROOT / "reports/breakthrough_20261002/batch45"
FIG = OUT / "figures"
DATAOUT = ROOT / "data/breakthrough_20261002/batch45"

COLS = ["full.oligo","info","query.seq","barcode","barcode.index","library",
        "input.pDNA.count",
        "expression.RNA.count.HepG2.replicate1","expression.RNA.count.HepG2.replicate2",
        "expression.RNA.count.K562.replicate1","expression.RNA.count.K562.replicate2",
        "expression.normalized.log2rpm.K562.replicate1","expression.normalized.log2rpm.K562.replicate2",
        "expression.normalized.log2rpm.HepG2.replicate1","expression.normalized.log2rpm.HepG2.replicate2",
        "expression.normalized.log2rpm.K562.averaged","expression.normalized.log2rpm.HepG2.averaged",
        "HepG2.query.PARM.prediction","K562.query.PARM.prediction"]

def sha256_file(p, chunk=1<<20):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b: break
            h.update(b)
    return h.hexdigest()

def parse():
    rows = []
    with gzip.open(SRC, "rt", encoding="utf-8", errors="replace") as f:
        hdr = f.readline().rstrip("\n").split("\t")
        assert hdr == COLS, ("header mismatch", hdr)
        for i, line in enumerate(f):
            r = line.rstrip("\n").split("\t")
            rows.append(r)
    return rows

def category(info):
    if info.startswith("control"):
        return "control"
    if "MUTATED" in info: return "MUTATED"
    if "top_individual" in info: return "top_individual"
    if "_best" in info: return "best"
    return "design"

def libtag(lib):
    return "long" if "232bp" in lib else "short"

def to_float(s):
    try:
        return float(s)
    except Exception:
        return None

def mean_finite(vals):
    fin = [v for v in vals if v is not None and math.isfinite(v)]
    if not fin: return None
    return sum(fin)/len(fin)

def gc_fraction(seq):
    if not seq: return None
    gc = sum(1 for c in seq.upper() if c in "GC")
    return gc/len(seq)

def spearman(x, y):
    """tie-aware Spearman; returns (rho, n) or (None,n) if undefined."""
    pairs = [(a,b) for a,b in zip(x,y) if a is not None and b is not None
             and math.isfinite(a) and math.isfinite(b)]
    n = len(pairs)
    if n < 3: return (None, n)
    xs = [p[0] for p in pairs]; ys = [p[1] for p in pairs]
    def rank(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0]*len(v); i = 0
        while i < len(v):
            j = i
            while j+1 < len(v) and v[order[j+1]] == v[order[i]]: j += 1
            avg = (i+j)/2.0 + 1.0
            for k in range(i, j+1): r[order[k]] = avg
            i = j+1
        return r
    rx = rank(xs); ry = rank(ys)
    mx = sum(rx)/n; my = sum(ry)/n
    num = sum((a-mx)*(b-my) for a,b in zip(rx,ry))
    dx = math.sqrt(sum((a-mx)**2 for a in rx)); dy = math.sqrt(sum((b-my)**2 for b in ry))
    if dx == 0 or dy == 0: return (None, n)
    return (num/(dx*dy), n)

def topk_overlap(pred, meas, ids, k):
    """k = ceil(0.1 n). lexical-ID tie rule (ascending) on both rankings."""
    idx = list(range(len(ids)))
    order_p = sorted(idx, key=lambda i: (-pred[i], ids[i]))
    order_m = sorted(idx, key=lambda i: (-meas[i], ids[i]))
    top_p = set(order_p[:k]); top_m = set(order_m[:k])
    inter = len(top_p & top_m)
    precision = inter/k
    base = k/len(ids)
    enrichment = precision/base if base > 0 else None
    return precision, enrichment, inter, k
