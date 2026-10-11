#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Batch16 unified inference entrypoint (strict declared input scope).

Predicts HUMAN GENE-LEVEL natural-expression E/S/B from a gene-bag input:
  4 diverse promoter sequences (sym400, uint8 {A,C,G,T,N}=0..4) + the frozen
  1069-feature composition/position vector for the gene.

This is NOT a fragment-activity predictor and NOT a HEK293T/construct-activity
predictor. Arbitrary promoter fragments cannot inherit a whole-gene S/B score.
Outputs distinguish catalog-observed labels (not available here) from model
predictions and return provenance + unsupported flags.

Bundle selection is validation-based (never test): the package step writes
models/breakthrough_20261002/batch16/deployment_bundle.json pointing at the
selected checkpoint(s) and scalers.
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from b16_common import (DATA_OUT, MODEL_OUT, CANDIDATE_VERSION, TASKS, AUX_FIELDS,
                        N_SLOTS, L400)
from b16_model import SingleTaskModel, MultiTaskModel, onehot


class PromoterAtlasGenePredictor:
    def __init__(self, bundle_path=None):
        bundle_path = Path(bundle_path or (MODEL_OUT / "deployment_bundle.json"))
        self.bundle = json.load(open(bundle_path, encoding="utf-8"))
        self.scalers = np.load(DATA_OUT / "scalers.npz", allow_pickle=True)
        self._load_models()

    def _load_models(self):
        self.models = {}
        arch = self.bundle.get("architecture")
        for task in TASKS:
            p = self.bundle["models"][task]
            if arch == "independent":
                m = SingleTaskModel(task)
                m.load_state_dict(torch.load(p, map_location="cpu")["model"])
                m.eval()
                self.models[task] = m
        if arch == "shared" or arch == "shared_aux":
            p = self.bundle["models"]["shared"]
            m = MultiTaskModel(n_aux=len(AUX_FIELDS) if arch == "shared_aux" else 0)
            m.load_state_dict(torch.load(p, map_location="cpu")["model"])
            m.eval()
            self.models["shared"] = m

    def _prep(self, seqs, features):
        """seqs: (4, 400) uint8; features: (1069,) float64 -> standardized."""
        seqs = np.asarray(seqs, dtype=np.uint8)
        assert seqs.shape == (N_SLOTS, L400), "expected (4,400) uint8 sequences"
        features = np.asarray(features, dtype=np.float64).reshape(1069)
        x_feat = (features - self.scalers["X_mean"]) / self.scalers["X_std"]
        x_seq = onehot(torch.from_numpy(seqs).unsqueeze(0).long(), L400)  # (1,4,400)
        return x_seq, torch.tensor(x_feat, dtype=torch.float32).unsqueeze(0)

    def predict_gene(self, seqs, features):
        """Return E/S/B raw-unit predictions + provenance. seqs=(4,400) uint8."""
        x_seq, x_feat = self._prep(seqs, features)
        out = {}
        with torch.no_grad():
            if "shared" in self.models:
                pred = self.models["shared"](x_seq, x_feat)
                for t in TASKS:
                    mu = float(self.scalers["%s_mean" % t])
                    sd = float(self.scalers["%s_std" % t])
                    out[t] = float(pred[t].item() * sd + mu)
            else:
                for t in TASKS:
                    mu = float(self.scalers["%s_mean" % t])
                    sd = float(self.scalers["%s_std" % t])
                    out[t] = float(self.models[t](x_seq, x_feat).item() * sd + mu)
        return {
            "candidate_version": CANDIDATE_VERSION,
            "architecture": self.bundle.get("architecture"),
            "predictions": {
                "E": {"value": out["E"], "unit": "Score_E rank-scaled [0,1] expression strength"},
                "S": {"value": out["S"], "unit": "S_raw negative cross-tissue MAD (less negative = more stable breadth)"},
                "B": {"value": out["B"], "unit": "B_raw cross-tissue detectable fraction [0,1]"},
            },
            "scope": "human gene-level natural-expression E/S/B from a 4-promoter gene-bag",
            "unsupported": {
                "fragment_activity": "arbitrary promoter fragments cannot inherit a whole-gene S/B score",
                "hek293t_construct": "no HEK293T / synthetic construct activity validation",
                "multispecies": "human only",
            },
            "is_model_prediction": True,
            "is_catalog_observed": False,
        }


if __name__ == "__main__":
    print(json.dumps({"entrypoint": "PromoterAtlasGenePredictor",
                      "candidate_version": CANDIDATE_VERSION,
                      "input_scope": "4-promoter gene-bag (4x400 uint8) + 1069 feature vector"},
                     ensure_ascii=False, indent=2))
