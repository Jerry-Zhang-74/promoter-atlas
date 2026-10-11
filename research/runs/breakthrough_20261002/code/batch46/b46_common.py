# -*- coding: utf-8 -*-
"""batch46 v6.33.0-within-generation-selection.rc1 -- shared parsing / freeze / metrics.

ZERO fit / solve / regression / calibration / new-model-score / neural-forward / GPU / web / download.
Read-only on the cached b44 source (syn.txt.gz) and all prior batch artifacts.
Writes ONLY under:
  reports/breakthrough_20261002/batch46
  data/breakthrough_20261002/batch46
  runs/breakthrough_20261002/code/batch46
The ONLY permitted writes outside the batch46 dirs are the atomic append to
  reports/breakthrough_20261002/version_registry.json  and  reports/breakthrough_20261002/CHANGELOG.txt
(Protocol C), performed by b46_phaseC_corrections.py, preserving every existing entry + production pointer.
"""
import gzip, hashlib, math, re, json, collections
from pathlib import Path

ROOT = Path(r"E:/promoter_atlas")
SRC = ROOT / "data/breakthrough_20261002/batch44/syn.txt.gz"
B45 = ROOT / "reports/breakthrough_20261002/batch45"
OUT = ROOT / "reports/breakthrough_20261002/batch46"
FIG = OUT / "figures"
DATAOUT = ROOT / "data/breakthrough_20261002/batch46"
CODE = ROOT / "runs/breakthrough_20261002/code/batch46"

BATCH = "batch46"
VERSION = "v6.33.0-within-generation-selection.rc1"
SRC_SHA256_COMPRESSED = "c7ce4b506a133afa5aee04c478c821b16dc64fdbfd612a8722779d06766244d6"
SRC_SHA256_DECOMPRESSED = "e7fc776ccb982130d3eb225cc6f47ae22baa3752b43d60bdf41a8db265e8c223"

COLS = ["full.oligo","info","query.seq","barcode","barcode.index","library",
        "input.pDNA.count",
        "expression.RNA.count.HepG2.replicate1","expression.RNA.count.HepG2.replicate2",
        "expression.RNA.count.K562.replicate1","expression.RNA.count.K562.replicate2",
        "expression.normalized.log2rpm.K562.replicate1","expression.normalized.log2rpm.K562.replicate2",
        "expression.normalized.log2rpm.HepG2.replicate1","expression.normalized.log2rpm.HepG2.replicate2",
        "expression.normalized.log2rpm.K562.averaged","expression.normalized.log2rpm.HepG2.averaged",
        "HepG2.query.PARM.prediction","K562.query.PARM.prediction"]

CI = {c: i for i, c in enumerate(COLS)}

def sha256_file(p, chunk=1 << 20):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()

def parse():
    """Return list of row-lists (positional, matching COLS)."""
    rows = []
    with gzip.open(SRC, "rt", encoding="utf-8", errors="replace", newline="") as f:
        hdr = f.readline().rstrip("\n").split("\t")
        assert hdr == COLS, ("header mismatch", hdr)
        for line in f:
            rows.append(line.rstrip("\n").split("\t"))
    return rows

def to_float(s):
    try:
        return float(s)
    except Exception:
        return None

def mean_finite(vals):
    fin = [v for v in vals if v is not None and math.isfinite(v)]
    if not fin:
        return None
    return sum(fin) / len(fin)

def gc_fraction(seq):
    if not seq:
        return None
    return sum(1 for c in seq.upper() if c in "GC") / len(seq)

# ---------------------------------------------------------------- info grammar
_INFO_RE = re.compile(
    r"^K562_fitness_(?P<lib232>232bp_)?replicate(?P<rep>\d+)_generation(?P<gen>\d+)"
    r"_individual(?P<ind>\d+)_(?P<sel>max|median|min)$")

SELS = ("max", "median", "min")

def parse_info(info):
    m = _INFO_RE.match(info)
    if not m:
        return None
    return {"lib": "long" if m.group("lib232") else "short",
            "rep": m.group("rep"),
            "gen": int(m.group("gen")),
            "ind": int(m.group("ind")),
            "sel": m.group("sel")}

def libtag(library):
    return "long" if "232bp" in library else "short"

def category(info):
    if info.startswith("control"):
        return "control"
    if "MUTATED" in info:
        return "MUTATED"
    if "top_individual" in info:
        return "top_individual"
    if info.endswith("_best"):
        return "best"
    return "design"

# ---------------------------------------------------------------- statistics
def spearman(x, y):
    """tie-aware Spearman; returns (rho, n) with rho None if undefined."""
    pairs = [(a, b) for a, b in zip(x, y)
             if a is not None and b is not None and math.isfinite(a) and math.isfinite(b)]
    n = len(pairs)
    if n < 3:
        return (None, n)
    xs = [p[0] for p in pairs]
    ys = [p[1] for p in pairs]

    def rank(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        i = 0
        while i < len(v):
            j = i
            while j + 1 < len(v) and v[order[j + 1]] == v[order[i]]:
                j += 1
            avg = (i + j) / 2.0 + 1.0
            for k in range(i, j + 1):
                r[order[k]] = avg
            i = j + 1
        return r

    rx, ry = rank(xs), rank(ys)
    mx, my = sum(rx) / n, sum(ry) / n
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    dx = math.sqrt(sum((a - mx) ** 2 for a in rx))
    dy = math.sqrt(sum((b - my) ** 2 for b in ry))
    if dx == 0 or dy == 0:
        return (None, n)
    return (num / (dx * dy), n)

def median(v):
    v = sorted(v)
    n = len(v)
    if n == 0:
        return None
    if n % 2:
        return v[n // 2]
    return (v[n // 2 - 1] + v[n // 2]) / 2.0

def summarize(vals):
    """sign/median/min/max summary of a list of finite floats (None -> null)."""
    fin = [v for v in vals if v is not None and math.isfinite(v)]
    return {"n": len(fin),
            "median": median(fin) if fin else None,
            "min": min(fin) if fin else None,
            "max": max(fin) if fin else None,
            "n_positive": sum(1 for v in fin if v > 0),
            "n_negative": sum(1 for v in fin if v < 0),
            "n_zero": sum(1 for v in fin if v == 0),
            "n_nonfinite_or_missing": len(vals) - len(fin)}

# ---------------------------------------------------------------- strict JSON
def _sanitize(o):
    """Recursively replace non-finite floats with {value:null, reason:'...'} so that
    json.dumps(..., allow_nan=False) never emits a literal NaN/Infinity."""
    if isinstance(o, float):
        if math.isnan(o):
            return {"value": None, "reason": "NAN"}
        if math.isinf(o):
            return {"value": None, "reason": "POSITIVE_INFINITY" if o > 0 else "NEGATIVE_INFINITY"}
        return o
    if isinstance(o, dict):
        return {k: _sanitize(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_sanitize(v) for v in o]
    return o

def write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    txt = json.dumps(_sanitize(obj), ensure_ascii=False, indent=2, allow_nan=False)
    path.write_text(txt, encoding="utf-8")
    return path
