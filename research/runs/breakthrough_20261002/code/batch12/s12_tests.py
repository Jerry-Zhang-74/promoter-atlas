#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Batch12 v6.13.0-siface.rc1 worker tests (unittest).

Meaningful regression + behavioral tests for the frozen gene-level raw-S
interface. Runs read-only against the frozen artifacts; no scientific fit is
performed. Fail-closed integrity tests use a temporary mutated copy (never the
real frozen artifacts), so frozen inputs are never mutated.
"""
from __future__ import annotations

import copy
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

CODE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(CODE_DIR))

import numpy as np  # noqa: E402

from s12_common import (  # noqa: E402
    B10_MODELS, CANDIDATE_VERSION, DATA_OUT, FROZEN_ARTIFACTS, N_ELIGIBLE,
    N_EXCLUDED, N_FEATURE_UNIVERSE, S_THEORETICAL_MAX, atomic_write_json,
    now_local, sha256_file, verify_artifacts)
from s12_predict import FrozenPredictor, strip_version  # noqa: E402

PY = sys.executable
PRED = str(CODE_DIR / "s12_predict.py")

# Canonical known genes.
ACTB = "ENSG00000075624"
GAPDH = "ENSG00000111640"
S_UNDEFINED = "ENSG00000000003"  # need an actual undefined example; replaced below


def _catalog_row(gid):
    p = FrozenPredictor(verify=False)
    return p.catalog.get(gid)


def _pick_undefined():
    p = FrozenPredictor(verify=False)
    for g, r in p.catalog.items():
        if not r["S_defined_main"] and g in p.universe_set:
            return g
    return None


def _pick_excluded():
    p = FrozenPredictor(verify=False)
    sdef = {g for g, r in p.catalog.items() if r["S_defined_main"]}
    for g in sorted(sdef):
        if g not in p.universe_set:
            return g
    return None


class TestFrozenInference(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.p = FrozenPredictor(verify=False)

    def test_known_valid_prediction(self):
        rec = self.p.predict([ACTB])[0]
        self.assertEqual(rec["status"], "predicted")
        self.assertEqual(rec["normalized_id"], ACTB)
        self.assertAlmostEqual(rec["observed_raw_S"], -1.079803244, places=9)
        self.assertIsInstance(rec["predicted_raw_S"], float)
        self.assertFalse(np.isnan(rec["predicted_raw_S"]))
        self.assertEqual(rec["split"], "train")
        self.assertTrue(rec["in_sample"])

    def test_gapdh_prediction(self):
        rec = self.p.predict([GAPDH])[0]
        self.assertEqual(rec["status"], "predicted")
        self.assertAlmostEqual(rec["observed_raw_S"], -0.2915078281, places=9)

    def test_s_undefined_rejected(self):
        g = _pick_undefined()
        self.assertIsNotNone(g)
        rec = self.p.predict([g])[0]
        self.assertEqual(rec["status"], "rejected")
        self.assertEqual(rec["reason_code"], "S_undefined")
        self.assertIn("B_raw", rec["reason_detail"])

    def test_transcript_id_rejected(self):
        rec = self.p.predict(["ENST00000331789"])[0]
        self.assertEqual(rec["status"], "rejected")
        self.assertEqual(rec["reason_code"], "transcript_id")

    def test_non_human_rejected(self):
        rec = self.p.predict(["ENSMUSG00000000001"])[0]
        self.assertEqual(rec["status"], "rejected")
        self.assertEqual(rec["reason_code"], "non_human")

    def test_unknown_human_rejected(self):
        rec = self.p.predict(["ENSG99999999999"])[0]
        self.assertEqual(rec["status"], "rejected")
        self.assertEqual(rec["reason_code"], "unknown")

    def test_empty_input_rejected(self):
        rec = self.p.predict([""])[0]
        self.assertEqual(rec["status"], "rejected")
        self.assertEqual(rec["reason_code"], "empty_input")

    def test_excluded_feature_missing(self):
        g = _pick_excluded()
        self.assertIsNotNone(g)
        rec = self.p.predict([g])[0]
        self.assertEqual(rec["status"], "rejected")
        self.assertEqual(rec["reason_code"], "S_defined_feature_missing")

    def test_determinism(self):
        a = self.p.predict([ACTB, GAPDH])
        b = self.p.predict([ACTB, GAPDH])
        self.assertEqual(a, b)
        self.assertEqual(a[0]["predicted_raw_S"], b[0]["predicted_raw_S"])

    def test_duplicate_input_preserved_order(self):
        a = self.p.predict([ACTB, GAPDH, ACTB])
        self.assertEqual([r["row_id"] for r in a], [1, 2, 3])
        self.assertEqual(a[0]["predicted_raw_S"], a[2]["predicted_raw_S"])
        self.assertEqual(a[0]["normalized_id"], a[2]["normalized_id"])

    def test_version_normalization(self):
        norm, stripped = strip_version("ENSG00000111640.14")
        self.assertEqual(norm, GAPDH)
        self.assertTrue(stripped)
        norm2, stripped2 = strip_version(GAPDH)
        self.assertEqual(norm2, GAPDH)
        self.assertFalse(stripped2)

    def test_catalog_counts(self):
        rows = self.p.build_catalog()
        self.assertEqual(len(rows), N_ELIGIBLE)
        excl = self.p.build_exclusion()
        self.assertEqual(len(excl), N_EXCLUDED)
        ids = [r["gene_id"] for r in rows]
        self.assertEqual(ids, sorted(ids))
        self.assertEqual(len(set(ids)), len(ids))

    def test_observed_predicted_correlation(self):
        rows = self.p.build_catalog()
        obs = np.array([r["observed_raw_S"] for r in rows])
        pred = np.array([r["predicted_raw_S"] for r in rows])
        r = float(np.corrcoef(obs, pred)[0, 1])
        self.assertGreater(r, 0.4)  # should be ~ Spearman .47 region
        self.assertTrue(np.all(np.isfinite(pred)))

    def test_out_of_range_flag_consistent(self):
        rows = self.p.build_catalog()
        for r in rows:
            self.assertEqual(r["outside_theoretical_range"],
                             bool(r["predicted_raw_S"] > S_THEORETICAL_MAX))
        n_flag = sum(1 for r in rows if r["outside_theoretical_range"])
        self.assertGreaterEqual(n_flag, 0)

    def test_provenance_present(self):
        rec = self.p.predict([ACTB])[0]
        self.assertIn("provenance", rec)
        self.assertEqual(rec["provenance"]["candidate_version"], CANDIDATE_VERSION)
        self.assertEqual(rec["provenance"]["n_features"], 1069)


class TestIntegrityFailClosed(unittest.TestCase):
    def test_real_artifacts_pass(self):
        ok, problems = verify_artifacts(FROZEN_ARTIFACTS)
        self.assertTrue(ok, problems)

    def test_missing_artifact_fails(self):
        reg = copy.deepcopy(FROZEN_ARTIFACTS)
        reg[0]["path"] = str(Path(tempfile.gettempdir()) / "nope_missing.joblib")
        ok, problems = verify_artifacts(reg)
        self.assertFalse(ok)
        self.assertTrue(any(p["error"] == "missing_file" for p in problems))

    def test_tampered_artifact_fails(self):
        tmpdir = Path(tempfile.mkdtemp(prefix="s12_tamper_"))
        fake = tmpdir / "tampered.joblib"
        fake.write_bytes(b"not a real model")
        reg = copy.deepcopy(FROZEN_ARTIFACTS)
        reg[0]["path"] = str(fake)
        ok, problems = verify_artifacts(reg)
        self.assertFalse(ok)
        self.assertTrue(any(p["error"] == "sha256_mismatch" for p in problems))
        shutil.rmtree(tmpdir, ignore_errors=True)

    def test_predictor_fails_closed_on_tamper(self):
        tmpdir = Path(tempfile.mkdtemp(prefix="s12_failclosed_"))
        fake = tmpdir / "models.joblib"
        fake.write_bytes(b"junk")
        reg = copy.deepcopy(FROZEN_ARTIFACTS)
        reg[0]["path"] = str(fake)
        import s12_predict
        old = s12_predict.FROZEN_ARTIFACTS
        s12_predict.FROZEN_ARTIFACTS = reg
        try:
            with self.assertRaises(RuntimeError):
                FrozenPredictor(verify=True)
        finally:
            s12_predict.FROZEN_ARTIFACTS = old
            shutil.rmtree(tmpdir, ignore_errors=True)


class TestCLI(unittest.TestCase):
    def _run(self, *args):
        return subprocess.run([PY, "-X", "utf8", PRED, *args],
                              capture_output=True, text=True, encoding="utf-8",
                              cwd=str(CODE_DIR))

    def test_version(self):
        r = self._run("version")
        self.assertEqual(r.returncode, 0)
        self.assertEqual(r.stdout.strip(), CANDIDATE_VERSION)

    def test_predict_json_valid(self):
        r = self._run("predict", "--genes", ACTB, "--format", "json")
        self.assertEqual(r.returncode, 0, r.stderr)
        data = json.loads(r.stdout)
        self.assertEqual(data[0]["status"], "predicted")
        self.assertEqual(data[0]["normalized_id"], ACTB)

    def test_trace_tsv(self):
        r = self._run("trace", "--genes", ACTB, "--format", "tsv")
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn("ENSG00000075624", r.stdout)
        self.assertIn("found", r.stdout)

    def test_no_nan_json(self):
        r = self._run("predict", "--genes", ACTB, GAPDH, "--format", "json")
        self.assertNotIn("NaN", r.stdout)
        self.assertNotIn("Infinity", r.stdout)


class TestReadOnlyNoMutation(unittest.TestCase):
    def test_frozen_artifacts_not_mutated(self):
        before = {a["key"]: sha256_file(a["path"]) for a in FROZEN_ARTIFACTS
                  if a.get("sha256")}
        p = FrozenPredictor(verify=True)
        p.predict([ACTB])
        p.build_catalog()
        after = {a["key"]: sha256_file(a["path"]) for a in FROZEN_ARTIFACTS
                 if a.get("sha256")}
        self.assertEqual(before, after)


if __name__ == "__main__":
    unittest.main(verbosity=2)

