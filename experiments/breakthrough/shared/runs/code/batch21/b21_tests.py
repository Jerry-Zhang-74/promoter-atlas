#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Batch21 tests for the repaired raw-input / reporting contract (no fabrication,
in-source marking, honest fold-aware OOF, declared N rule, clear errors)."""
from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from b21_common import B21InputError, read_fasta, sha256_text, MPRA_SUPERVISED  # noqa: E402
import b21_input  # noqa: E402
import b21_train  # noqa: E402
import b21_mpra  # noqa: E402

PASS, FAIL = [], []


def check(name, cond, detail=""):
    (PASS if cond else FAIL).append(name)
    print(("PASS " if cond else "FAIL ") + name + ((" | " + detail) if detail else ""))


def test_validate():
    ok = "ACGT" * 41
    r = b21_input.validate_sequence(ok, "ok164")
    check("validate.valid164", r["length"] == 164 and r["n_fraction"] == 0.0)
    for name, seq in [("empty", ""), ("wronglen", "ACGT" * 10),
                      ("iupac", "ACGT" * 40 + "RY"),
                      ("badbase", "ACGT" * 40 + "Z"),
                      ("nheavy", "N" * 12 + "ACGT" * 38)]:
        try:
            b21_input.validate_sequence(seq, name)
            check("validate.refuse_" + name, False, "did not raise")
        except B21InputError as e:
            check("validate.refuse_" + name, True, str(e)[:60])


def test_fasta():
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "e.fa"; p.write_text("", encoding="utf-8")
        try:
            read_fasta(p); check("fasta.empty_refused", False)
        except B21InputError:
            check("fasta.empty_refused", True)
        p2 = Path(d) / "nohdr.fa"; p2.write_text("ACGTACGT\n", encoding="utf-8")
        try:
            read_fasta(p2); check("fasta.noheader_refused", False)
        except B21InputError:
            check("fasta.noheader_refused", True)


def test_in_source_and_oof():
    df = pd.read_csv(MPRA_SUPERVISED, sep="\t")
    seqs = df["sequence"].astype(str).str.upper().tolist()
    cohort = b21_mpra._cohort_oof()
    known = next(s for s in seqs if s in cohort)          # in prior CV cohort
    outside = next(s for s in seqs if s not in cohort)    # catalog but outside cohort
    rows, refused = b21_input.predict_records([("known", known), ("outside", outside)],
                                              compute_nn=False)
    rk = next(r for r in rows if r["name"] == "known")
    ro = next(r for r in rows if r["name"] == "outside")
    check("in_source.known_oof", rk["in_prior_cv_cohort"] and rk["prediction_type"] == "fold_aware_oof"
          and rk["oof_available"] and rk["prediction"] is not None)
    check("in_source.outside_no_oof", (not ro["in_prior_cv_cohort"]) and ro["prediction"] is None
          and ro["prediction_type"] == "in_source_no_oof")
    check("never_fabricate_oof_outside_cohort", ro.get("prediction") is None)
    # a genuinely novel sequence -> ensemble15
    rng = np.random.default_rng(0)
    novel = "".join(rng.choice(list("ACGT"), 164))
    while novel in cohort:
        novel = "".join(rng.choice(list("ACGT"), 164))
    rn, _ = b21_input.predict_records([("novel", novel)], compute_nn=False)
    check("novel.ensemble15", rn[0]["prediction_type"] == "ensemble15_new_sequence"
          and rn[0]["prediction"] is not None and rn[0]["source_similarity"] == "out_of_source_catalog")


def test_ranking_loss():
    import torch
    z = torch.tensor([0.0, 1.0, 2.0])
    p = torch.tensor([0.0, 0.5, 2.0])
    l = b21_train.ranking_loss(p, z)
    check("rankloss.nonzero_with_pairs", float(l) > 0.0, "loss=%.4f" % float(l))
    z0 = torch.zeros(4); p0 = torch.zeros(4)
    check("rankloss.zero_no_pairs", float(b21_train.ranking_loss(p0, z0)) == 0.0)


def test_cli_errors():
    exe = [sys.executable, "-X", "utf8", str(HERE / "b21_cli.py")]
    r = subprocess.run(exe + ["predict-fasta", "--fasta", "does_not_exist.fa"],
                       capture_output=True, text=True, env={**os.environ, "PYTHONUTF8": "1"})
    check("cli.missing_file_exit1_no_traceback",
          r.returncode == 1 and "Traceback" not in (r.stderr or ""),
          "rc=%d stderr=%s" % (r.returncode, (r.stderr or "").strip()[:60]))


if __name__ == "__main__":
    test_validate(); test_fasta(); test_in_source_and_oof(); test_ranking_loss(); test_cli_errors()
    print("\n%d PASS, %d FAIL" % (len(PASS), len(FAIL)))
    sys.exit(1 if FAIL else 0)
