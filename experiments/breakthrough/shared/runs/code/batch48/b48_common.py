# -*- coding: utf-8 -*-
"""batch48 v6.34.1-scoped-input-routing.rc2 -- shared paths / constants / honest utilities.

MINIMAL scoped copy of batch47 b47_common.py (the ONLY changes are the batch/version identity and
the batch48 output roots). All routing semantics are unchanged. The read-only batch35 PARM engine and
its frozen external weights are reused EXACTLY as in batch47 (engine/weights UNCHANGED).

Scope of batch48 = two correctness repairs over b47:
  (R1) artifact FINALIZATION ordering (b48_finalize.py) -- immutable files written BEFORE the manifest;
  (R2) gene-observed FIELD_DEFINITIONS.observedScore_S truthfulness (b48_gene_observed.py);
plus an accurate Chinese USAGE and a corrective acceptance note. No new predictor, no GUI, no fit/GPU.

This module imports NOTHING heavy (no numpy/torch/Bio), so gene-observed runs with CUDA hidden.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]

CODE = ROOT / "runs" / "breakthrough_20261002" / "code" / "batch48"
B48_REPORT = ROOT / "reports" / "breakthrough_20261002" / "batch48"
B48_DATA = ROOT / "data" / "breakthrough_20261002" / "batch48"
B48_RUNS = ROOT / "runs" / "breakthrough_20261002" / "batch48"

# read-only reused b35 engine + its frozen asset manifest / external asset root (UNCHANGED from b47)
B35_ENGINE = ROOT / "code" / "batch35" / "parm_fasta_predict.py"
B35_MANIFEST = ROOT / "reports" / "breakthrough_20261002" / "batch35" / "frozen_assets.json"
DEFAULT_ASSET_ROOT = ROOT / "data" / "breakthrough_20261002" / "batch31" / "_external"

# read-only b20-authoritative catalog sources (no network)
LABEL_TABLE = ROOT / "labels" / "build" / "gene_expression_label_table.tsv"
ANALYSIS_SET = ROOT / "labels" / "build" / "promoter_analysis_set.tsv"

REGISTRY = ROOT / "reports" / "breakthrough_20261002" / "version_registry.json"
CHANGELOG = ROOT / "reports" / "breakthrough_20261002" / "CHANGELOG.txt"

BATCH = "batch48"
VERSION = "v6.34.1-scoped-input-routing.rc2"

# frozen PARM software input limit (official central zero-padding to L_max=600)
L_MAX = 600
CANONICAL = set("ACGT")
OFFICIAL_AMBIGUOUS = set("N")

SCORE_LABEL = "PARM_HEK293_model_score"
MODEL_SOURCE_CELL = "HEK293"
DEFAULT_TARGET_CELL = "293FT"
BIOLOGICAL_STATUS = "UNVALIDATED_FOR_293FT"

# gene-observed label contract (exact source columns; see source_field_definitions at runtime)
REQUIRED_LABEL_COLUMNS = ["gene_id", "Score_E", "S_raw", "B_raw", "S_defined_main", "is_in_main_subset"]


class B48InputError(ValueError):
    """Typed user-input error."""


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def sha256_text(t: str) -> str:
    return hashlib.sha256(t.encode("utf-8")).hexdigest()


def sanitize(o):
    """Recursively replace non-finite floats with {"value": null, "reason": "..."}.
    Guarantees json.dumps(..., allow_nan=False) never emits literal NaN/Infinity."""
    if isinstance(o, float):
        if math.isnan(o):
            return {"value": None, "reason": "NAN"}
        if math.isinf(o):
            return {"value": None, "reason": "POSITIVE_INFINITY" if o > 0 else "NEGATIVE_INFINITY"}
        return o
    if isinstance(o, dict):
        return {k: sanitize(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [sanitize(v) for v in o]
    return o


def write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    txt = json.dumps(sanitize(obj), ensure_ascii=False, indent=2, allow_nan=False)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(txt + "\n", encoding="utf-8")
    os.replace(tmp, path)
    return path


def write_tsv(path, rows, cols):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as fh:
        fh.write("\t".join(cols) + "\n")
        for r in rows:
            fh.write("\t".join("" if r.get(c) is None else str(r.get(c)) for c in cols) + "\n")
    return path


def read_fasta(path):
    """Minimal strict FASTA reader -> [(name, sequence)]. Raises on no records.
    A header with no sequence yields an empty-string sequence (routed as EMPTY, never scored)."""
    recs = []
    name = None
    buf = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if name is not None:
                    recs.append((name, "".join(buf)))
                name = line[1:].strip() or ("seq%d" % (len(recs) + 1))
                buf = []
            else:
                buf.append(line)
    if name is not None:
        recs.append((name, "".join(buf)))
    if not recs:
        raise B48InputError("no FASTA records found in %s" % path)
    return recs


def looks_like_dna(tok: str) -> bool:
    """Heuristic guard: a token that is clearly a DNA sequence, not a gene symbol/ID."""
    t = (tok or "").strip().upper()
    if len(t) < 12:
        return False
    if t.startswith(">"):
        return True
    return bool(t) and all(c in "ACGTN" for c in t)
