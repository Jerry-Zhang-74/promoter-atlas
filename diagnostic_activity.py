#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""按长度分组预测片段活性，恢复原始行序。"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

sys.dont_write_bytecode = True
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from predict import find_weights, load_model  # noqa: E402
from spec_contract import encode_sequence  # noqa: E402

FRAG_MIN_LEN = 100
FRAG_MAX_LEN = 600


def read_fragments(path):
    with open(path, encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f, delimiter="\t"))
    out = []
    for i, r in enumerate(rows, 2):
        seq = (r.get("sequence") or "").upper()
        if not FRAG_MIN_LEN <= len(seq) <= FRAG_MAX_LEN:
            raise ValueError("line %d: sequence length %d outside [%d, %d]"
                             % (i, len(seq), FRAG_MIN_LEN, FRAG_MAX_LEN))
        if set(seq) - set("ACGTN"):
            raise ValueError("line %d: sequence contains unsupported characters" % i)
        out.append({"seq_id": r.get("seq_id") or "frag_%d" % (i - 1), "sequence": seq})
    if not out:
        raise ValueError("input TSV is empty")
    return out


def predict_activity(input_path, weights=None, out="activity.tsv", device="cpu"):
    rows = read_fragments(input_path)
    weights_path = find_weights(weights)
    model, obj = load_model(weights_path, device)
    norm = obj.get("activity_normalizer")
    if not norm:
        raise ValueError("checkpoint has no activity_normalizer")
    mean, std = float(norm["mean"]), float(norm["std"])
    torch.set_num_threads(1)
    z = np.empty(len(rows), dtype=np.float32)
    by_length = {}
    for i, row in enumerate(rows):
        by_length.setdefault(len(row["sequence"]), []).append(i)
    with torch.inference_mode():
        for indices in by_length.values():
            for start in range(0, len(indices), 4):
                ii = indices[start:start + 4]
                codes = torch.tensor(np.stack([encode_sequence(rows[i]["sequence"]) for i in ii]),
                                     dtype=torch.long, device=device)
                z[ii] = model.forward_fragment(codes).float().cpu().numpy()
    out_path = Path(out)
    if not out_path.is_absolute():
        out_path = Path.cwd() / out_path
    with out_path.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(["seq_id", "length", "activity_z", "activity_log2rpm_pred", "target_scale"])
        for r, zi in zip(rows, z):
            w.writerow([r["seq_id"], len(r["sequence"]), float(zi), float(zi * std + mean),
                        norm.get("target", "")])
    print("wrote %d rows to %s" % (len(rows), out_path))
    return out_path


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", required=True)
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--out", default="activity.tsv")
    ap.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    args = ap.parse_args(argv)
    predict_activity(args.input, args.checkpoint, args.out, args.device)


if __name__ == "__main__":
    main()
