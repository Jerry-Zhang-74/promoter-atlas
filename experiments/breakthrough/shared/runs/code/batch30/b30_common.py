# -*- coding: utf-8 -*-
"""batch30 shared helpers (minimal, self-contained).
Provenance: logic derived from runs/breakthrough_20261002/code/batch26/b26_common.py
  sha256 f38ba9762e7f357dd255e4ae69e4957138e9e308cbdb3745ed71ec86c7a9e923 (READ-ONLY original).
Pure stdlib + numpy.  No model fits, no gradients here.  UTF-8 explicit.
"""
from __future__ import annotations
import datetime as dt, hashlib, json, os, random
from pathlib import Path

ROOT = Path(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "..")))
RUNS_BT = ROOT / "runs" / "breakthrough_20261002"
REPORTS_BT = ROOT / "reports" / "breakthrough_20261002"
CODE30 = RUNS_BT / "code" / "batch30"
RUN30 = RUNS_BT / "batch30"
DATA30 = ROOT / "data" / "breakthrough_20261002" / "batch30"
REP30 = REPORTS_BT / "batch30"
DATA29 = ROOT / "data" / "breakthrough_20261002" / "batch29"
REP29 = REPORTS_BT / "batch29"
SOURCE_GZ = ROOT / "data" / "breakthrough_20261002" / "batch27" / "source" / "GSE78709_SuRE-counts-K562_B45_B55_LP170105.txt.gz"
SOURCE_SHA = "8868fa641c2c89bef0c8aa18b6b553030cc7199ea4be4f12c2cd1f3ecd458b85"


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


def sha256_text(s):
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def atomic_write_json(path, obj):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8", newline="\n") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, sort_keys=False)
        f.flush(); os.fsync(f.fileno())
    os.replace(tmp, path)


def atomic_write_text(path, text):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8", newline="\n") as f:
        f.write(text); f.flush(); os.fsync(f.fileno())
    os.replace(tmp, path)


def append_jsonl(path, obj):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n"); f.flush(); os.fsync(f.fileno())


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
