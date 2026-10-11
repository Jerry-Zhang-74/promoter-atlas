#!/usr/bin/env python3
"""Batch07 inference wrapper v6.8.0-covergate.rc1 (scope-aware, CPU).

Wraps the frozen gene-level HGB predictor with a feature-coverage score/flag.
This is a diagnostic applicability tool, NOT a confidence/calibration guarantee.

Scope rules:
  - valid gene-level 45-feature context -> prediction + coverage score + flag.
  - flag = inside_assessed_coverage if distance <= train 90th percentile,
           else outside_assessed_coverage (NOT "confirmed unreliable", NOT "zero activity").
  - single promoter / synthetic construct / HEK293T activity -> unsupported
    regardless of coverage.

A gate miss is not proof of biological out-of-domain.
"""
import sys
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from b7_common import (  # noqa: E402
    MODEL_DIR, MODEL_DIR06, sha256_file, mahalanobis_sq, CANDIDATE,
    HGB_SHA, FEATURES45_SHA, SPLIT_ARRAYS_SHA, GEOMETRY_NPZ, GEOMETRY_META,
)
from b7_data import load_labels_split, load_features  # noqa: E402

GATE_NOTE = (
    "coverage gate NOT validated for error reduction on this dataset "
    "(primary 90% gate MAE reduction below the pre-registered 5% threshold); "
    "coverage is diagnostic only, never an automatic claim of reliability"
)


class CoverGatePredictor:
    def __init__(self):
        import joblib
        self.hgb = joblib.load(MODEL_DIR06 / "hgb_baseline.joblib")
        self.geom = np.load(GEOMETRY_NPZ)
        self.geom_meta = json.loads(GEOMETRY_META.read_text(encoding="utf-8"))
        self.geom_sha = sha256_file(GEOMETRY_NPZ)
        assert self.geom_sha == self.geom_meta["geometry_sha256"]
        self.mean = self.geom["mean"]
        self.scale = self.geom["scale"]
        self.location = self.geom["location"]
        self.precision = self.geom["precision"]
        self.threshold = float(self.geom["threshold"])
        self.gene_gid, self.gene_score_e, self.gene_split, self.comp_dense, self.n_prom, self.n_genes = load_labels_split()
        self.X45, self.comp44, self.log1p_count, self.gc_gene = load_features()
        self.gid_to_idx = {g: i for i, g in enumerate(self.gene_gid)}

    def _feature_contract(self):
        return {
            "n_features": 45,
            "feature_order": "comp44 (per-promoter mono+dinuc+GC+length for sym400 and up2000 => 44, gene=mean over promoters) + log1p(n_promoters) (1)",
            "aggregation": "gene-mean over promoters; no label-derived features",
            "features45_sha256": FEATURES45_SHA,
        }

    def _score(self, x45):
        x45 = np.asarray(x45, dtype=np.float64).reshape(1, 45)
        xs = (x45 - self.mean) / self.scale
        d = float(mahalanobis_sq(xs, self.location, self.precision)[0])
        inside = d <= self.threshold
        flag = "inside_assessed_coverage" if inside else "outside_assessed_coverage"
        return d, flag, inside

    def predict_features(self, x45):
        x45 = np.asarray(x45, dtype=np.float64)
        if x45.shape != (45,):
            return {"status": "refused_invalid_input",
                    "reason": "expected 45-dim feature vector, got shape " + str(tuple(x45.shape)),
                    "candidate_version": CANDIDATE}
        if not np.isfinite(x45).all():
            return {"status": "refused_nonfinite_features", "candidate_version": CANDIDATE}
        d, flag, inside = self._score(x45)
        pred = float(self.hgb.predict(x45.reshape(1, 45))[0])
        return {
            "status": "ok",
            "candidate_version": CANDIDATE,
            "prediction_Score_E": pred,
            "coverage_score_mahalanobis_sq": d,
            "coverage_flag": flag,
            "inside_assessed_coverage": inside,
            "model_sha256": HGB_SHA,
            "geometry_sha256": self.geom_sha,
            "threshold_90_train": self.threshold,
            "feature_contract": self._feature_contract(),
            "limitations": [
                "gene-level Score_E ranking only; NOT promoter/construct/HEK293T activity",
                "coverage score = feature centrality, not calibrated probability and not comprehensive density/OOD",
                "gate miss is not proof of biological out-of-domain",
                GATE_NOTE,
            ],
            "abstention_supported": "optional research abstention consistent with the fixed rule is supported",
        }

    def predict_gene_id(self, gene_id, abstain_if_outside=False):
        gene_id = str(gene_id)
        if gene_id not in self.gid_to_idx:
            return {
                "status": "missing_context",
                "gene_id": gene_id,
                "candidate_version": CANDIDATE,
                "reason": "gene not present in the assessed dataset; full valid gene context is required",
            }
        i = self.gid_to_idx[gene_id]
        x45 = self.X45[i]
        out = self.predict_features(x45)
        out["gene_id"] = gene_id
        out["dataset_split"] = str(self.gene_split[i])
        out["n_promoters"] = int(self.n_prom[i])
        if abstain_if_outside and not out["inside_assessed_coverage"]:
            out["abstained"] = True
            out["abstention_reason"] = "outside_assessed_coverage (optional research abstention)"
        return out

    def unsupported_scope(self, scope):
        scopes = {
            "single_promoter": "arbitrary single-promoter expression is unsupported regardless of coverage",
            "synthetic_construct": "synthetic construct activity is unsupported regardless of coverage",
            "cell_line_activity": "HEK293T / arbitrary cell-line activity is unsupported regardless of coverage",
        }
        if scope not in scopes:
            return {"status": "unknown_scope", "scope": scope, "candidate_version": CANDIDATE}
        return {
            "status": "unsupported_scope",
            "scope": scope,
            "reason": scopes[scope],
            "candidate_version": CANDIDATE,
        }


def demo():
    p = CoverGatePredictor()
    test_idx = np.flatnonzero(p.gene_split == "test")
    xs_all = (p.X45.astype(np.float64) - p.mean) / p.scale
    dist = mahalanobis_sq(xs_all, p.location, p.precision)

    order = np.argsort(dist[test_idx], kind="mergesort")
    in_genes = [str(p.gene_gid[test_idx[i]]) for i in order[:3]]
    out_genes = [str(p.gene_gid[test_idx[i]]) for i in order[-3:]]

    print("=== in-coverage examples (chosen by coverage only) ===")
    for g in in_genes:
        r = p.predict_gene_id(g)
        print("  %s: flag=%s d=%.2f pred=%.4f" % (g, r["coverage_flag"], r["coverage_score_mahalanobis_sq"], r["prediction_Score_E"]))
    print("=== out-of-coverage examples (chosen by coverage only) ===")
    for g in out_genes:
        r = p.predict_gene_id(g)
        print("  %s: flag=%s d=%.2f pred=%.4f" % (g, r["coverage_flag"], r["coverage_score_mahalanobis_sq"], r["prediction_Score_E"]))
    print("=== refusal examples ===")
    print(json.dumps(p.unsupported_scope("single_promoter"), ensure_ascii=False))
    print(json.dumps(p.unsupported_scope("cell_line_activity"), ensure_ascii=False))
    print(json.dumps(p.predict_gene_id("ENSG00000000000"), ensure_ascii=False))
    print("DEMO DONE")


if __name__ == "__main__":
    demo()
