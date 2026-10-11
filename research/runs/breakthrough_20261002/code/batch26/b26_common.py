# -*- coding: utf-8 -*-
"""batch26 shared helpers: paths, hashing, atomic IO, deterministic seeds.

Pure stdlib + numpy. No model fits here. UTF-8 explicit.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
import random
from pathlib import Path

ROOT = Path(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "..")))
RUNS_BT = ROOT / "runs" / "breakthrough_20261002"
REPORTS_BT = ROOT / "reports" / "breakthrough_20261002"

CODE26 = RUNS_BT / "code" / "batch26"
RUN26 = RUNS_BT / "batch26"
DATA26 = ROOT / "data" / "breakthrough_20261002" / "batch26"
MODEL26 = ROOT / "models" / "breakthrough_20261002" / "batch26"
REPORT26 = REPORTS_BT / "batch26"
FIG26 = ROOT / "figures" / "batch26"

# read-only inputs (batch25 freeze)
B25_DATA = ROOT / "data" / "breakthrough_20261002" / "batch25"
VARIANTS_TSV = B25_DATA / "str_variant_variants.tsv"
VARIANTS_SHA = "72a5298f0015dea50255ee169a5eee0f918a5ef6b9bc4775dec518384cae4c0e"
B25_SRC = RUNS_BT / "batch25" / "source"
ARRAY_PROBES = B25_SRC / "str_mpra_design_array_probes.tsv"
TSS_STR_PAIRS = B25_SRC / "str_mpra_design_tss_str_pairs.tsv"
B25_FREEZE = REPORTS_BT / "batch25" / "external_eval_freeze.json"

# frozen batch02 architecture definition (READ-ONLY import target)
BATCH02_CODE = RUNS_BT / "code" / "batch02"

SPLIT_SEED = 20261026
SPLIT_MSG = "batch26_seed20261026"
SHUFFLE_SEED = 20261028

FIVE = "ACTGGCCGCTTGACG"
ASISI = "CACTGCGGCTCCTGCGATCGC"
COPY_VAR = {"m5": -5, "ref": 0, "p3": 3, "p5": 5}


def now_local():
    return dt.datetime.now().astimezone().replace(microsecond=0).isoformat()


def sha256_file(p, chunk=1 << 20):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def sha256_bytes(b):
    return hashlib.sha256(b).hexdigest()


def atomic_write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8", newline="\n") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, sort_keys=False)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def atomic_write_text(path, text):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8", newline="\n") as f:
        f.write(text)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def append_jsonl(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())


def read_json(path, default=None):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return default


def set_thread_limits(n=4):
    for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
              "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        os.environ[v] = str(n)


def set_seed(seed):
    random.seed(seed)
    import numpy as np
    np.random.seed(seed % (2 ** 32))
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def stable_hash_u01(s):
    """Deterministic [0,1) from a string (sha256)."""
    h = hashlib.sha256(s.encode("utf-8")).hexdigest()
    return int(h[:16], 16) / float(1 << 64)


def groupid_of(locus_ids, seq_shas):
    """Stable group id = sha256(min sequence sha + sorted locus ids)."""
    payload = min(seq_shas) + "|" + ",".join(sorted(locus_ids))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:32]
