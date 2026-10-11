# -*- coding: utf-8 -*-
"""Standalone batch52 predictor for the saved neural checkpoints.

Loads a saved Nabs/Njoint checkpoint (strict config+datahash check), encodes a raw DNA
sequence (A/C/G/T; N/other -> ambiguous, never pooled), and returns HEK293 Log2RPM_HEK293
activity in the author's units. Sequences >600 bp are rejected (>=601 unsupported); the
model is NOT windowed. CPU or CUDA forward (no gradients, no training).

Usage:
  python predict_v2.py --checkpoint <ckpt> --seq ACGT...
  python predict_v2.py --checkpoint <ckpt> --seq-file seqs.txt   (one sequence per line)
  python predict_v2.py --checkpoint <ckpt> --example             (run a real cohort example)
"""
from __future__ import annotations
import argparse, json, os, sys

os.environ.setdefault("OMP_NUM_THREADS", "4")
HERE = os.path.dirname(os.path.abspath(__file__))
HELPER = os.path.abspath(os.path.join(HERE, "..", "..", "runs", "breakthrough_20261002", "code", "batch53_model_core"))
sys.path.insert(0, HELPER)

import numpy as np  # noqa: E402
import torch  # noqa: E402

from checkpoint import load_checkpoint  # noqa: E402
from model_core import build_model  # noqa: E402
from normalizer import TargetNormalizer  # noqa: E402
from spec import MAX_LEN, TARGET_COLUMN, SOURCE_SHA256, config_sha256  # noqa: E402

_BASE = {"A": 0, "C": 1, "G": 2, "T": 3}
COH = os.path.abspath(os.path.join(HERE, "..", "..", "data", "breakthrough_20261002", "batch52",
                                   "corrected", "cohort_v2"))


def encode(seq: str) -> np.ndarray:
    s = seq.strip().upper()
    if len(s) == 0:
        raise ValueError("empty sequence")
    if len(s) > MAX_LEN:
        raise ValueError("sequence length %d > %d unsupported (no windowing)" % (len(s), MAX_LEN))
    arr = np.full((1, MAX_LEN), 4, dtype=np.int64)
    arr[0, :len(s)] = [_BASE.get(c, 4) for c in s]
    return arr


def load_model(ckpt, device="cpu", expect_datahash=None):
    obj = load_checkpoint(ckpt, expect_config_sha=config_sha256(), expect_datahash=expect_datahash)
    if obj["config"]["target_column"] != TARGET_COLUMN:
        raise RuntimeError("checkpoint target != %s" % TARGET_COLUMN)
    m = build_model(); m.load_state_dict(obj["model"]); m.eval(); m.to(device)
    norm = TargetNormalizer(**{k: v for k, v in obj["normalizer"].items() if k in ("mean", "std")})
    return m, norm, obj


def predict(ckpt, seq, device="cpu", expect_datahash=None):
    m, norm, obj = load_model(ckpt, device, expect_datahash)
    x = torch.from_numpy(encode(seq)).to(device)
    with torch.no_grad():
        z = m(x)
        y = norm.inverse(z).cpu().numpy()
    return float(y[0]), obj


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--seq", default=None)
    ap.add_argument("--seq-file", default=None)
    ap.add_argument("--example", action="store_true")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--cohort-dir", default=COH)
    a = ap.parse_args()
    seqs = []
    if a.example:
        with open(os.path.join(a.cohort_dir, "sequences.fa"), encoding="utf-8") as f:
            name = None
            for ln in f:
                ln = ln.strip()
                if ln.startswith(">"):
                    if name and seqs:
                        break
                    name = ln[1:]
                elif ln:
                    seqs.append(ln); break
    if a.seq:
        seqs.append(a.seq)
    if a.seq_file:
        seqs += [ln.strip() for ln in open(a.seq_file, encoding="utf-8") if ln.strip()]
    if not seqs:
        raise SystemExit("provide --seq, --seq-file or --example")
    for s in seqs:
        y, obj = predict(a.checkpoint, s, a.device)
        print(json.dumps({"length": len(s), "predicted_%s" % TARGET_COLUMN: y,
                          "checkpoint": os.path.basename(a.checkpoint),
                          "target_column": TARGET_COLUMN, "unit": "author log2 RPM (HEK293)",
                          "note": "HEK293 reference-proxy exploratory; not 293FT/GFP validation"},
                         ensure_ascii=False))


if __name__ == "__main__":
    main()
