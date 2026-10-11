# -*- coding: utf-8 -*-
"""batch42 v6.29.0-hekft-supervised-pilot.rc1 shared paths + fixed contracts.

Scope: assay-specific SUPERVISED learning on the public HEK293FT OligoMPRA
270bp alpha target.  CPU only, <=4 BLAS/OMP threads, ZERO neural/GPU/torch.
"""
from __future__ import annotations

import csv
import datetime
import hashlib
import json
import os
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[4]

BATCH = "batch42"
VERSION = "v6.29.0-hekft-supervised-pilot.rc1"

B41_DATA = ROOT / "data" / "breakthrough_20261002" / "batch41"
B41_REPORT = ROOT / "reports" / "breakthrough_20261002" / "batch41"
B42_DATA = ROOT / "data" / "breakthrough_20261002" / "batch42"
B42_RUN = ROOT / "runs" / "breakthrough_20261002" / "batch42"
B42_CODE = ROOT / "runs" / "breakthrough_20261002" / "code" / "batch42"
B42_REPORT = ROOT / "reports" / "breakthrough_20261002" / "batch42"
B42_FIG = B42_REPORT / "figures"

FROZEN_TABLE = B41_DATA / "frozen_input_table.tsv"
DESIGN_FA = ROOT / "data" / "breakthrough_20261002" / "batch40" / "source" / "GSM9580234_design_rmIllegalChars.fa.gz"
H1_CACHE = B41_DATA / "cache" / "h1" / "preds.json"
PARM_CACHE = B41_DATA / "cache" / "parm" / "preds.json"
EXEC_GATE = ROOT / "runs" / "breakthrough_20261002" / "code" / "batch27" / "exec_gate.py"

FROZEN_TABLE_SHA = "984093604759d7f7e0a968765d278177941c9658c37f356149afb6ee78a9fbf1"
DESIGN_SHA = "a61f3d7624bdc1cd6066cb0cfb015d482e2994885cf321da9e0bdb7e753aa82b"
H1_GRAM_SHA = "45b5c161ad7c63f7831e2d466ba82e92b71325500520659b159afc02138cd5ad"
B22_SEQ_TABLE = ROOT / "data" / "breakthrough_20261002" / "batch22" / "sequence_table.tsv"
B24_KERNELS_DIR = ROOT / "runs" / "breakthrough_20261002" / "code" / "batch24"

SPLIT_DOMAIN = "b42-group-v1"
TRAIN_MOD = 80
R0_ALPHA = 100.0
R1_ALPHA = 1.0
NULL_SEED = 20261043
N_BOOT = 2000
BOOT_SEED = 20261044
# Feasibility gate per PARENT PRE-FIT GROUP-COUNT CORRECTION 2026-10-04 02:42
# (protocol line 6): after source-parent merging, minimum feasibility is
# train>=10 groups / test>=5 groups with row gates 1000/300 unchanged.
MIN_TRAIN_ROWS = 1000
MIN_TRAIN_GROUPS = 10
MIN_TEST_ROWS = 300
MIN_TEST_GROUPS = 5
# Formal conditional bootstrap still requires >=30 test groups; 5-29 => DESCRIPTIVE
# pilot, no CI/significance, no operational PASS (verdict LIMITED).
MIN_TEST_GROUPS_FORMAL = 30
MAX_FITS = 3


def set_thread_limits(n=4):
    for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
              "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "BLIS_NUM_THREADS"):
        os.environ[k] = str(n)


def sha256_file(p):
    with open(p, "rb") as fh:
        return hashlib.file_digest(fh, "sha256").hexdigest()


def sha256_text(t):
    return hashlib.sha256(t.encode("utf-8")).hexdigest()


def sha256_bytes(b):
    return hashlib.sha256(b).hexdigest()


def now_local():
    return datetime.datetime.now().astimezone().strftime("%Y-%m-%dT%H:%M:%S%z")


def atomic_write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, ensure_ascii=False, indent=1)
        fh.write("\n")
    os.replace(tmp, path)


def append_jsonl(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(obj, ensure_ascii=False) + "\n")
        fh.flush()
        os.fsync(fh.fileno())


def load_frozen_rows(with_labels=False):
    rows = []
    with open(FROZEN_TABLE, encoding="utf-8") as fh:
        for r in csv.DictReader(fh, delimiter="\t"):
            ctl = int(r["control"])
            d = {
                "label": r["label"],
                "control": ctl,
                "chr": r["chr"],
                "start": int(r["start"]) if r["start"].strip() else None,
                "end": int(r["end"]) if r["end"].strip() else None,
                "strand": r["strand"],
                "sequence": r["sequence"],
                "sequence_sha256": r["sequence_sha256"],
                "component_id": r["component_id"],
            }
            if with_labels:
                d["alpha_primary"] = float(r["alpha_primary"])
            rows.append(d)
    return rows


H1_ALPHABET = "ACGT"
H1_BASE = {c: i for i, c in enumerate(H1_ALPHABET)}


def h1_kmer_counts(seq, k):
    n = len(seq)
    if n < k:
        return np.zeros(4 ** k, dtype=np.float64)
    mask = (4 ** k) - 1
    codes = np.empty(n - k + 1, dtype=np.int64)
    code = 0
    for i, ch in enumerate(seq):
        d = H1_BASE[ch]
        code = ((code * 4) + d) & mask
        if i >= k - 1:
            codes[i - k + 1] = code
    return np.bincount(codes, minlength=4 ** k).astype(np.float64)


def h1_l2(v):
    n = np.linalg.norm(v)
    return np.zeros_like(v) if n <= 0 else v / n


def h1_features(seqs):
    n = len(seqs)
    X = np.empty((n, 1280), dtype=np.float64)
    s = np.sqrt(0.5)
    for i, seq in enumerate(seqs):
        X[i, :256] = s * h1_l2(h1_kmer_counts(seq, 4))
        X[i, 256:] = s * h1_l2(h1_kmer_counts(seq, 5))
    return X


def h1_gram(seqs):
    X = h1_features(seqs)
    return X @ X.T


def rc(seq):
    comp = {"A": "T", "C": "G", "G": "C", "T": "A"}
    return "".join(comp[c] for c in reversed(seq))


def _bincode(s):
    v = 0
    for ch in s:
        v = v * 4 + H1_BASE[ch]
    return v


def composition_features(seq):
    out = np.zeros(342, dtype=np.float64)
    L = len(seq)
    off = 0
    for k, dim in ((1, 4), (2, 16), (3, 64), (4, 256)):
        c = np.bincount([_bincode(seq[j:j + k]) for j in range(L - k + 1)], minlength=dim).astype(np.float64)
        out[off:off + dim] = c / max(1, (L - k + 1))
        off += dim
    gc = (seq.count("G") + seq.count("C")) / max(1, L)
    out[340] = gc
    out[341] = np.log(L)
    return out


def spearman(a, b):
    a = np.asarray(a, float); b = np.asarray(b, float)
    if len(a) < 3 or np.std(a) == 0 or np.std(b) == 0:
        return float("nan")
    ra = _rank(a) - np.mean(_rank(a)); rb = _rank(b) - np.mean(_rank(b))
    den = np.sqrt((ra @ ra) * (rb @ rb))
    return float((ra @ rb) / den) if den > 0 else float("nan")


def _rank(x):
    order = np.argsort(x, kind="mergesort")
    r = np.empty(len(x), dtype=np.float64)
    r[order] = np.arange(1, len(x) + 1, dtype=np.float64)
    xs = x[order]
    i = 0
    while i < len(xs):
        j = i
        while j + 1 < len(xs) and xs[j + 1] == xs[i]:
            j += 1
        if j > i:
            r[order[i:j + 1]] = (i + 1 + j + 1) / 2.0
        i = j + 1
    return r


def mae(pred, obs):
    return float(np.mean(np.abs(np.asarray(pred, float) - np.asarray(obs, float))))


def r2(pred, obs):
    pred = np.asarray(pred, float); obs = np.asarray(obs, float)
    ss = np.sum((obs - obs.mean()) ** 2)
    return float(1 - np.sum((obs - pred) ** 2) / ss) if ss > 0 else float("nan")


def top_decile_precision(pred, obs, tie_keys):
    pred = np.asarray(pred, float); obs = np.asarray(obs, float)
    n = len(pred)
    k = int(np.ceil(0.1 * n))
    order = np.lexsort((np.asarray(tie_keys, dtype=str), -pred))
    pred_top = set(order[:k].tolist())
    oorder = np.lexsort((np.asarray(tie_keys, dtype=str), -obs))
    true_top = set(oorder[:k].tolist())
    inter = len(pred_top & true_top)
    base = k / n
    return {"k": k, "n": n, "precision": inter / k if k else float("nan"),
            "enrichment": (inter / k) / base if base > 0 else float("nan"),
            "base_rate": base}
