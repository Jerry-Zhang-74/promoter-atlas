#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Raw-input CLI for batch22 v6.17.0-hektransfer.rc1.

Scientifically supported because frozen weights and the sequence-feature /
representation definitions are data-independent. It exposes:

  repr     : raw sequences -> frozen RecipeA pooled representation (128-d, cuda:0)
  features : raw sequences -> R1 composition features (mono/di/tri + GC + log1p len)
  predict  : raw sequences -> a FOLD-SPECIFIC saved readout prediction.

`predict` applies a pre-existing per-fold readout (train stats saved during Stage 3);
it deliberately does NOT train an all-data delivery estimator (reserved for parent
acceptance per the batch22 protocol). Predictions are in the trained HEK293 units
(bulk log2 RNA/input) and are out-of-domain for other cell lines/assays.

Input : FASTA or a TSV/CSV/txt with a 'sequence' column (or one sequence per line).
Usage :
  python -X utf8 b22_cli.py repr     --in seqs.fa --out repr.tsv
  python -X utf8 b22_cli.py features --in seqs.fa --out comp.tsv
  python -X utf8 b22_cli.py predict  --in seqs.fa --recipe R2 --fold 0 --out preds.tsv
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from b22_common import MODEL_OUT, now_local, set_thread_limits  # noqa: E402
from b22_fits import feature_composition, fit_ridge  # noqa: E402

ALLOWED = set("ACGT")


def read_sequences(path):
    p = Path(path)
    txt = p.read_text(encoding="utf-8-sig")
    if txt.lstrip().startswith(">"):
        seqs, cur = [], []
        for line in txt.splitlines():
            if line.startswith(">"):
                if cur:
                    seqs.append("".join(cur)); cur = []
            else:
                cur.append(line.strip())
        if cur:
            seqs.append("".join(cur))
        return [s.upper() for s in seqs]
    first = txt.splitlines()[0]
    if "\t" in first or "," in first:
        delim = "\t" if "\t" in first else ","
        rows = list(csv.DictReader(txt.splitlines(), delimiter=delim))
        cols = [c for c in (rows[0].keys() if rows else []) if c.lower() == "sequence"]
        if not cols:
            raise SystemExit("no 'sequence' column found")
        return [str(r[cols[0]]).strip().upper() for r in rows]
    return [l.strip().upper() for l in txt.splitlines() if l.strip()]


def validate(seqs):
    bad = [(i, len(s), "".join(sorted(set(s) - ALLOWED))) for i, s in enumerate(seqs)
           if set(s) - ALLOWED]
    return bad


def encode_any(model, seqs, batch=32):
    """Length-agnostic encode: group equal-length inputs (no padding bias)."""
    from b22_frozen import encode_pooled
    H = None
    for L in sorted({len(s) for s in seqs}):
        idx = [i for i, s in enumerate(seqs) if len(s) == L]
        part = encode_pooled(model, [seqs[i] for i in idx], batch=min(batch, len(idx)))
        if H is None:
            H = np.zeros((len(seqs), part.shape[1]), dtype=part.dtype)
        H[idx] = part
    return H


def cmd_repr(args):
    from b22_frozen import assert_cuda, load_frozen
    seqs = read_sequences(args.infile)
    bad = validate(seqs)
    if bad:
        raise SystemExit("non-ACGT input rows (row,length,extra): %r" % bad[:5])
    assert_cuda()
    model, meta = load_frozen()
    H = encode_any(model, seqs)
    with open(args.out, "w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh, delimiter="\t")
        w.writerow(["row", "length"] + ["h%d" % i for i in range(H.shape[1])])
        for i, s in enumerate(seqs):
            w.writerow([i, len(s)] + [float(v) for v in H[i]])
    print(json.dumps({"cmd": "repr", "n": len(seqs), "dim": int(H.shape[1]),
                      "device": "cuda:0", "model_sha256": meta["model_sha256"],
                      "at": now_local()}))


def cmd_features(args):
    seqs = read_sequences(args.infile)
    X = feature_composition(seqs)
    with open(args.out, "w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh, delimiter="\t")
        w.writerow(["row", "length"] + ["f%d" % i for i in range(X.shape[1] - 1)])
        for i, s in enumerate(seqs):
            w.writerow([i, len(s)] + [float(v) for v in X[i]])
    print(json.dumps({"cmd": "features", "n": len(seqs), "dim": int(X.shape[1]),
                      "definition": "mono4+di16+tri64+GC+log1plen",
                      "units": "unstandardized", "at": now_local()}))


def cmd_predict(args):
    seqs = read_sequences(args.infile)
    bad = validate(seqs)
    if bad:
        raise SystemExit("non-ACGT input rows (row,length,extra): %r" % bad[:5])
    adir = MODEL_OUT / ("attempt_fold%d_%s" % (args.fold, args.recipe))
    ts = adir / "train_stats.npz"
    if not ts.exists():
        raise SystemExit("no saved readout for %s" % adir)
    s = np.load(ts)
    X1 = feature_composition(seqs)
    if args.recipe == "R2":
        from b22_frozen import assert_cuda, load_frozen
        assert_cuda()
        model, meta = load_frozen()
        H = encode_any(model, seqs)
        X = np.concatenate([X1, H.astype(np.float64)], axis=1)
    else:
        X = X1
    mu, sd = s["train_mean"], s["train_scale"]
    Z = (X - mu) / sd
    pred = Z @ s["coef"] + float(s["intercept"])
    with open(args.out, "w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh, delimiter="\t")
        w.writerow(["row", "length", "prediction", "units"])
        for i, sq in enumerate(seqs):
            w.writerow([i, len(sq), float(pred[i]), "log2(HEK293 RNA/input)"])
    print(json.dumps({"cmd": "predict", "recipe": args.recipe, "fold": args.fold,
                      "n": len(seqs), "readout_dir": str(adir),
                      "note": "fold-specific readout; NOT an all-data delivery estimator",
                      "at": now_local()}))


def main():
    set_thread_limits(4)
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name, fn in (("repr", cmd_repr), ("features", cmd_features), ("predict", cmd_predict)):
        sp = sub.add_parser(name)
        sp.add_argument("--in", dest="infile", required=True)
        sp.add_argument("--out", required=True)
        if name == "predict":
            sp.add_argument("--recipe", choices=["R1", "R2"], required=True)
            sp.add_argument("--fold", type=int, required=True)
        sp.set_defaults(func=fn)
    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()

