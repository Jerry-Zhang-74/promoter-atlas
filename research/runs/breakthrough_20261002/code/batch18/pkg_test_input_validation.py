#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""F17-3 regression: the input-schema validator must raise the TYPED
PromoterAtlasInputError for every declared violation, in particular an EMPTY
batch (N=0).  Before the rc2 fix, an empty batch fell through to
``numpy``'s ``min()/max()`` on a zero-size array, raising a BARE ValueError that
was NOT a PromoterAtlasInputError -- so ``except PromoterAtlasInputError`` did
not catch it.  This test pins the fixed behaviour and is meaningful because it
fails on rc1.

Run:  python code/test_input_validation.py   (from anywhere in the package)
"""
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from promoter_atlas_predict import (  # noqa: E402
    PromoterAtlasGenePredictor, PromoterAtlasInputError, validate_seqs, validate_features)

GOOD_SEQ = np.zeros((4, 400), dtype=np.uint8)
GOOD_FEAT = np.zeros(1069, dtype=np.float64)


def _expect_typed(name, seqs, feats):
    try:
        PromoterAtlasGenePredictor().predict_gene(seqs, feats)
    except PromoterAtlasInputError:
        return True
    except Exception as e:  # untyped -> regression
        print("  FAIL %-22s -> untyped %s: %s" % (name, type(e).__name__, e))
        return False
    print("  FAIL %-22s -> no error raised" % name)
    return False


def main():
    cases = [
        ("empty batch N=0", np.zeros((0, 4, 400), dtype=np.uint8), np.zeros((0, 1069))),
        ("bad seqs shape", np.zeros((4, 399), dtype=np.uint8), GOOD_FEAT),
        ("bad seqs dtype", np.zeros((4, 400), dtype=np.float64), GOOD_FEAT),
        ("bad alphabet", np.full((4, 400), 9, dtype=np.uint8), GOOD_FEAT),
        ("bad features length", GOOD_SEQ, np.zeros(1000)),
        ("non-finite features", GOOD_SEQ, np.full(1069, np.nan)),
        ("length mismatch", np.zeros((2, 4, 400), dtype=np.uint8), np.zeros((1, 1069))),
    ]
    results = {name: _expect_typed(name, s, f) for name, s, f in cases}

    # direct empty-features check (bypasses the (4,400)->(1,4,400) reshape in predict_gene)
    def direct(fn):
        try:
            fn(np.zeros((0, 1069)))
        except PromoterAtlasInputError:
            return True
        except Exception as e:
            print("  FAIL direct empty features -> untyped %s: %s" % (type(e).__name__, e))
            return False
        print("  FAIL direct empty features -> no error raised")
        return False

    results["direct empty features N=0"] = direct(validate_features)
    results["direct empty seqs N=0"] = direct(lambda a: validate_seqs(np.zeros((0, 4, 400), dtype=np.uint8)))
    # positive control: a valid batch must still work
    try:
        out = PromoterAtlasGenePredictor().predict_batch(GOOD_SEQ[None, ...], GOOD_FEAT[None, :])
        ok_valid = all(np.isfinite(out[t]).all() for t in ("E", "S", "B"))
    except Exception as e:
        print("  FAIL valid batch:", type(e).__name__, e)
        ok_valid = False
    for name, ok in results.items():
        print(("  ok  " if ok else "  FAIL") + " " + name)
    print("valid batch predict:", "ok" if ok_valid else "FAIL")
    allok = all(results.values()) and ok_valid
    print("F17_3_REGRESSION:", "PASS" if allok else "FAIL")
    return 0 if allok else 1


if __name__ == "__main__":
    sys.exit(main())

