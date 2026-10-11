
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Reproducible minimal example for Promoter Atlas v6.15.1-release.rc2.

Run from ANY directory after extracting the release zip:
    python code/run_example.py            # or: python run_example.py
It loads code/example_input.npz, predicts E/S/B with the shipped weights, compares
to code/example_expected_output.json within the declared tolerance, and exercises
four invalid inputs (including the F17-3 EMPTY batch N=0) to show the schema
errors are typed. No training assets are needed.
"""
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from promoter_atlas_predict import (PromoterAtlasGenePredictor, PromoterAtlasInputError)  # noqa: E402

TOL = 1e-5


def main():
    inp = np.load(HERE / "example_input.npz", allow_pickle=True)
    exp = json.load(open(HERE / "example_expected_output.json", encoding="utf-8"))
    pred = PromoterAtlasGenePredictor()
    res = pred.predict_batch(inp["seqs"], inp["features"])
    genes = inp["gene_ids"].astype(str)
    worst = 0.0
    print("candidate_version:", exp["candidate_version"])
    for i, g in enumerate(genes):
        line = ["  %s" % g]
        for t in ("E", "S", "B"):
            got = float(res[t][i]); want = float(exp["predictions"][g][t])
            d = abs(got - want); worst = max(worst, d)
            line.append("%s got=%.6f exp=%.6f |d|=%.1e" % (t, got, want, d))
        print(" | ".join(line))
    ok = worst <= TOL
    print("max |diff| = %.3e (tol %.1e) -> %s" % (worst, TOL, "REPLAY_OK" if ok else "REPLAY_FAIL"))

    # --- schema validity errors (must raise the TYPED error) ---
    cases = [
        ("bad shape", np.zeros((4, 399), dtype=np.uint8), inp["features"][0]),
        ("bad alphabet", np.full((4, 400), 9, dtype=np.uint8), inp["features"][0]),
        ("bad features", inp["seqs"][0], np.zeros(1000, dtype=np.float64)),
        # F17-3: empty batch must raise PromoterAtlasInputError, not a raw numpy error
        ("empty batch N=0", np.zeros((0, 4, 400), dtype=np.uint8), np.zeros((0, 1069), dtype=np.float64)),
    ]
    errs = 0
    untyped = 0
    for name, seqs, feats in cases:
        try:
            pred.predict_gene(seqs, feats)
            print("VALIDATION FAIL (no error):", name)
        except PromoterAtlasInputError as e:
            errs += 1
            print("validation raised (%s): %s" % (name, str(e)[:70]))
        except Exception as e:  # untyped -> F17-3 regression
            untyped += 1
            print("UNTYPED ERROR (%s): %s" % (name, type(e).__name__))
    print("validity_errors_ok:", errs == len(cases) and untyped == 0)
    return 0 if (ok and errs == len(cases) and untyped == 0) else 1


if __name__ == "__main__":
    sys.exit(main())

