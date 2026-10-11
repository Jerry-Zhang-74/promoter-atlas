#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Promoter Atlas v6.15.1-release.rc2 - portable gene-level E/S/B predictor.

SUPPORTED INPUT (strict, declared scope):
  * gene bag: exactly 4 promoter slot sequences, each length 400, dtype uint8,
    alphabet {A,C,G,T,N} -> encoded {0,1,2,3,4}. Any other value is invalid.
  * features: one frozen 1069-dim composition/position vector for the gene
    (44/45-region composition + log1p promoter count + 1024 kmer-bin features).
    Order/scaling are FIXED by the training pipeline; supply RAW (unstandardized)
    values - standardization uses the shipped train-fitted transforms.

OUTPUT: raw-unit human gene-level natural-expression predictions E, S, B.
  E = Score_E rank-scaled expression strength [0,1]
  S = S_raw = -MAD(log2(1+detectable TPM)) over detectable tissues (less negative = stable)
  B = B_raw = detectable_tissue_count / tissue_count (cross-tissue breadth) [0,1]

NOT SUPPORTED / NOT CLAIMED: arbitrary promoter-fragment activity, HEK293T or
synthetic-construct activity, multi-species, or temporal-stability interpretation
of S. This is a MODEL prediction; catalog OBSERVED labels are a separate artifact
(see labels/README.md) and must not be conflated with predictions.

CHANGE vs rc1 (packaging-only): F17-3 fix - an EMPTY batch (N=0) now raises the
typed PromoterAtlasInputError instead of a raw numpy ValueError. No weight,
transform, architecture or numerical tolerance change.

Self-contained: no absolute paths, no training assets required to run inference.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

PKG_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = PKG_ROOT / "models"
TRANSFORMS = PKG_ROOT / "transforms" / "scalers.npz"
N_SLOTS = 4
L400 = 400
N_FEATURES = 1069
TASKS = ("E", "S", "B")
TASK_META = {
    "E": {"unit": "Score_E rank-scaled [0,1] expression strength", "mask": "all feature-universe genes"},
    "S": {"unit": "S_raw = -MAD(log2(1+detectable TPM)); less negative = more stable",
          "mask": "S_defined: B_raw>=0.30 AND detectable_tissues>=3"},
    "B": {"unit": "B_raw = detectable_tissue_count / tissue_count [0,1]", "mask": "B_raw defined"},
}
CANDIDATE_VERSION = "v6.15.1-release.rc2"


class PromoterAtlasInputError(ValueError):
    """Raised when an input violates the declared schema (order/scale/length/alphabet)."""


def validate_seqs(seqs):
    a = np.asarray(seqs)
    if a.ndim == 2:
        a = a[None, ...]
    if a.ndim != 3 or a.shape[1] != N_SLOTS or a.shape[2] != L400:
        raise PromoterAtlasInputError(
            "seqs must have shape (N,%d,%d) or (%d,%d); got %r" % (N_SLOTS, L400, N_SLOTS, L400, tuple(a.shape)))
    if not np.issubdtype(a.dtype, np.integer):
        raise PromoterAtlasInputError("seqs dtype must be integer/uint8 (alphabet {0..4}); got %s" % a.dtype)
    # F17-3: an EMPTY batch must raise the typed error (a.min()/a.max() on an
    # empty array otherwise raises an untyped numpy ValueError).
    if a.shape[0] == 0:
        raise PromoterAtlasInputError(
            "seqs batch is empty (N=0); supply at least one gene bag of shape (%d,%d)" % (N_SLOTS, L400))
    if a.min() < 0 or a.max() > 4:
        raise PromoterAtlasInputError("seqs values must be in 0..4 (A,C,G,T,N); got min=%d max=%d" % (int(a.min()), int(a.max())))
    return a.astype(np.uint8)


def validate_features(features):
    f = np.asarray(features, dtype=np.float64)
    if f.ndim == 1:
        f = f[None, :]
    if f.ndim != 2 or f.shape[1] != N_FEATURES:
        raise PromoterAtlasInputError("features must have shape (N,%d) or (%d,); got %r" % (N_FEATURES, N_FEATURES, tuple(f.shape)))
    # F17-3: an EMPTY batch (N=0) must raise the typed error too.
    if f.shape[0] == 0:
        raise PromoterAtlasInputError(
            "features batch is empty (N=0); supply at least one %d-dim feature vector" % N_FEATURES)
    if not np.all(np.isfinite(f)):
        raise PromoterAtlasInputError("features contain non-finite values; supply finite RAW (unstandardized) features")
    return f


def _torch():
    import torch
    return torch


def onehot(x, length=L400):
    torch = _torch()
    b, l = x.shape
    oh = torch.zeros(b, 4, l, dtype=torch.float32, device=x.device)
    ok = x < 4
    bidx = torch.arange(b, device=x.device).unsqueeze(1).expand(b, l)
    pos = torch.arange(l, device=x.device).unsqueeze(0).expand(b, -1)
    oh[bidx[ok], x[ok], pos[ok]] = 1.0
    return oh


def _build_model_classes():
    torch = _torch()
    import torch.nn as nn
    import torch.nn.functional as F

    class Trunk(nn.Module):
        def __init__(self, out_dim=128):
            super().__init__()
            self.conv1 = nn.Conv1d(4, 64, kernel_size=7, padding=3)
            self.bn1 = nn.BatchNorm1d(64)
            self.conv2 = nn.Conv1d(64, 64, kernel_size=5, padding=2)
            self.bn2 = nn.BatchNorm1d(64)
            self.conv3 = nn.Conv1d(64, out_dim, kernel_size=5, padding=2)
            self.bn3 = nn.BatchNorm1d(out_dim)

        def forward(self, x):
            h = F.relu(self.bn1(self.conv1(x)))
            h = F.relu(self.bn2(self.conv2(h)))
            h = F.relu(self.bn3(self.conv3(h)))
            return h

    class FeatureBranch(nn.Module):
        def __init__(self, n_feat=1069, hidden=128):
            super().__init__()
            self.fc1 = nn.Linear(n_feat, hidden)
            self.fc2 = nn.Linear(hidden, hidden)

        def forward(self, x):
            return F.relu(self.fc2(F.relu(self.fc1(x))))

    class SharedRep(nn.Module):
        def __init__(self, n_feat=1069, emb=128, feat_hidden=128):
            super().__init__()
            self.trunk = Trunk(emb)
            self.feat = FeatureBranch(n_feat, feat_hidden)
            self.emb = emb
            self.feat_hidden = feat_hidden

        def rep(self, x_seq, x_feat):
            z = self.trunk(x_seq).mean(dim=2)
            B = x_seq.shape[0] // 4
            z = z.reshape(B, 4, self.emb)
            seq_emb = torch.cat([z.mean(dim=1), z.amax(dim=1)], dim=1)
            feat_emb = self.feat(x_feat)
            return torch.cat([seq_emb, feat_emb], dim=1)

    class SingleTaskModel(nn.Module):
        def __init__(self, n_feat=1069, emb=128, feat_hidden=128):
            super().__init__()
            self.shared = SharedRep(n_feat, emb, feat_hidden)
            self.head = nn.Linear(2 * emb + feat_hidden, 1)

        def forward(self, x_seq, x_feat):
            return self.head(self.shared.rep(x_seq, x_feat)).squeeze(-1)

    return SingleTaskModel


class PromoterAtlasGenePredictor:
    """Loads the shipped independent-arm (A, reference seed) E/S/B weights + transforms."""

    def __init__(self, package_root=None):
        root = Path(package_root) if package_root else PKG_ROOT
        self.models_dir = root / "models"
        self.scalers = np.load(root / "transforms" / "scalers.npz", allow_pickle=True)
        self.torch = _torch()
        SingleTaskModel = _build_model_classes()
        self.models = {}
        for t in TASKS:
            m = SingleTaskModel()
            sd = self.torch.load(self.models_dir / ("model_%s.pt" % t), map_location="cpu")
            m.load_state_dict(sd["model"])
            m.eval()
            self.models[t] = m
        self._meta = {"architecture": "independent", "reference_seed": 20261016,
                      "candidate_version": CANDIDATE_VERSION}

    def _prep(self, seqs, features):
        a = validate_seqs(seqs)
        f = validate_features(features)
        if len(a) != len(f):
            raise PromoterAtlasInputError("seqs batch (%d) and features batch (%d) length mismatch" % (len(a), len(f)))
        torch = self.torch
        x_std = (f - self.scalers["X_mean"]) / self.scalers["X_std"]
        x_seq = onehot(torch.from_numpy(a.reshape(-1, L400).copy()).long(), L400)
        return x_seq, torch.tensor(x_std, dtype=torch.float32), len(a)

    def predict_batch(self, seqs, features):
        x_seq, x_feat, n = self._prep(seqs, features)
        out = {t: np.zeros(n, dtype=np.float64) for t in TASKS}
        with self.torch.no_grad():
            for i in range(n):
                s = x_seq[i * N_SLOTS:(i + 1) * N_SLOTS]
                f = x_feat[i:i + 1]
                for t in TASKS:
                    std = float(self.models[t](s, f).item())
                    out[t][i] = std * float(self.scalers["%s_std" % t]) + float(self.scalers["%s_mean" % t])
        return out

    def predict_gene(self, seqs, features):
        """Single gene -> full JSON-serializable result with provenance and scope flags."""
        pred = self.predict_batch(np.asarray(seqs)[None, ...], np.asarray(features)[None, :])
        return {
            "candidate_version": CANDIDATE_VERSION,
            "architecture": self._meta["architecture"],
            "reference_seed": self._meta["reference_seed"],
            "predictions": {t: {"value": float(pred[t][0]), "unit": TASK_META[t]["unit"],
                                "defined_for": TASK_META[t]["mask"]} for t in TASKS},
            "scope": {"organism": "human", "level": "gene (4-promoter bag)",
                      "label_family": "natural gene expression E/S/B"},
            "unsupported": {
                "fragment_activity": "arbitrary promoter fragments cannot inherit a whole-gene S/B score",
                "hek293t_construct": "no HEK293T / synthetic-construct activity validation",
                "multispecies": "human only",
                "temporal_stability": "raw S is a negative cross-tissue MAD, NOT a temporal-stability measure",
            },
            "is_model_prediction": True,
            "is_catalog_observed": False,
            "labels_reference": "catalog OBSERVED labels live in labels/README.md (not produced by this model)",
        }


def main(argv=None):
    ap = argparse.ArgumentParser(description="Promoter Atlas v6.15.1-release.rc2 gene-level E/S/B predictor")
    ap.add_argument("--input", required=True, help="npz with arrays seqs (N,4,400 uint8) and features (N,1069 float)")
    ap.add_argument("--out", default=None, help="optional JSON output path")
    args = ap.parse_args(argv)
    z = np.load(args.input, allow_pickle=True)
    predictor = PromoterAtlasGenePredictor()
    res = predictor.predict_batch(z["seqs"], z["features"])
    genes = z["gene_ids"].astype(str) if "gene_ids" in z.files else ["gene_%d" % i for i in range(len(res["E"]))]
    out = {"candidate_version": CANDIDATE_VERSION, "n": len(genes),
           "predictions": [{"gene_id": g, **{t: float(res[t][i]) for t in TASKS}} for i, g in enumerate(genes)]}
    text = json.dumps(out, ensure_ascii=False, indent=2)
    print(text)
    if args.out:
        Path(args.out).write_text(text + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())

