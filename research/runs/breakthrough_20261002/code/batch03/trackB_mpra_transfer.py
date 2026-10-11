#!/usr/bin/env python3
"""Batch03 Track B: frozen MPRA (batch02 legnet) representation transfer to 14 constructs."""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.linear_model import Ridge

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from b3_common import atomic_write_json  # noqa: E402

BATCH02_CODE = ROOT / "runs" / "breakthrough_20261002" / "code" / "batch02"
sys.path.insert(0, str(BATCH02_CODE))
from trackB_grouped_cv import LegNetInspired, DilatedResBlock, BASE, encode_sequences  # noqa: E402

REPORT = ROOT / "reports" / "breakthrough_20261002" / "batch03"
CELL = ROOT / "reports" / "cellline_v2_20261001"
MPRA_SUP = ROOT / "features" / "mpra" / "mpra_supervised.tsv"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
L = 164
STRIDE = 82
ALPHAS = [0.1, 1.0, 10.0, 100.0, 1000.0]
PCA_GRID = [0, 8, 16]
CKPTS = [
    "ckpt_legnet_fold0_seed20261008.pt",
    "ckpt_legnet_fold0_seed20261009.pt",
    "ckpt_legnet_fold0_seed20261010.pt",
]
CKPT_DIR = ROOT / "models" / "breakthrough_20261002" / "batch02"


class LegNetExtractor(nn.Module):
    def __init__(self, ch=64):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv1d(4, ch, 7, padding=3), nn.BatchNorm1d(ch), nn.ReLU(inplace=True))
        self.blocks = nn.ModuleList([DilatedResBlock(ch, d) for d in [1, 2, 4, 8]])
        self.head = nn.Sequential(
            nn.Linear(ch * 2, 64), nn.ReLU(inplace=True), nn.Dropout(0.1),
            nn.Linear(64, 1),
        )

    def embed(self, x_idx):
        x = F.one_hot(x_idx.long(), num_classes=5)[..., :4].float().permute(0, 2, 1).contiguous()
        x = self.stem(x)
        for blk in self.blocks:
            x = blk(x)
        return torch.cat([x.mean(dim=2), x.max(dim=2).values], dim=1)


def load_ensemble(device):
    models = []
    for name in CKPTS:
        st = torch.load(CKPT_DIR / name, map_location="cpu", weights_only=False)
        m = LegNetExtractor()
        m.load_state_dict(st["model"])
        m.eval().to(device)
        models.append(m)
    return models


def windows(seq, length=L, stride=STRIDE):
    seq = seq.upper()
    n = len(seq)
    out = []
    if n < length:
        pad = length - n
        left = pad // 2
        s = "N" * left + seq + "N" * (pad - left)
        out.append(s)
    else:
        for i in range(0, n - length + 1, stride):
            out.append(seq[i:i + length])
        if (n - length) % stride != 0:
            out.append(seq[n - length:n])
    return out


@torch.inference_mode()
def embed_construct(models, seq, device):
    ws = windows(seq)
    x = encode_sequences(ws)
    x_idx = torch.from_numpy(x.astype(np.int64)).to(device)
    emb = []
    for m in models:
        emb.append(m.embed(x_idx).cpu().numpy())
    e = np.mean(np.stack(emb, axis=0), axis=0)
    return e.mean(axis=0)


def gc_frac(seq):
    s = seq.upper()
    if len(s) == 0:
        return 0.0
    return float((s.count("G") + s.count("C")) / len(s))


def overlap_audit(constructs, mpra_seqs):
    k = 20
    mpra_set = set()
    for s in mpra_seqs:
        su = s.upper()
        for i in range(0, len(su) - k + 1):
            mpra_set.add(su[i:i + k])
    exact_set = set(s.upper() for s in mpra_seqs)
    rows = []
    for _, c in constructs.iterrows():
        su = c["sequence"].upper()
        exact = su in exact_set
        hits = 0
        tot = max(1, len(su) - k + 1)
        for i in range(0, tot):
            if su[i:i + k] in mpra_set:
                hits += 1
        rows.append({"construct_id": c["construct_id"], "family": c["family"],
                     "exact_mpra_match": bool(exact), "kmer20_hit_fraction": hits / tot})
    return rows


def _fit_ridge(Xtr, ytr, Xte, alpha, pca_n):
    mu = Xtr.mean(axis=0)
    sd = Xtr.std(axis=0)
    sd = np.where(sd < 1e-12, 1.0, sd)
    Xtr_s = (Xtr - mu) / sd
    Xte_s = (Xte - mu) / sd
    if pca_n and pca_n > 0:
        from sklearn.decomposition import PCA
        pca = PCA(n_components=min(pca_n, Xtr_s.shape[0], Xtr_s.shape[1]), random_state=0)
        Xtr_s = pca.fit_transform(Xtr_s)
        Xte_s = pca.transform(Xte_s)
    m = Ridge(alpha=alpha, fit_intercept=True)
    m.fit(Xtr_s, ytr)
    return m.predict(Xte_s), m, mu, sd


def inner_select(X, y, groups, alpha_grid, pca_grid):
    uniq = np.unique(groups)
    best = None
    for alpha in alpha_grid:
        for pca_n in pca_grid:
            errs = []
            for g in uniq:
                tr = np.flatnonzero(groups != g)
                te = np.flatnonzero(groups == g)
                pred, _, _, _ = _fit_ridge(X[tr], y[tr], X[te], alpha, pca_n)
                errs.append(np.abs(pred - y[te]))
            mae = float(np.mean(np.concatenate(errs)))
            if best is None or mae < best["mae"]:
                best = {"alpha": alpha, "pca_n": pca_n, "mae": mae}
    return best


def run_protocol(X, y, ids, families, outer_mode):
    if outer_mode == "construct_LOO":
        outer_groups = np.asarray(ids)
    else:
        outer_groups = np.asarray(families)
    errs = []
    for og in np.unique(outer_groups):
        tr = np.flatnonzero(outer_groups != og)
        te = np.flatnonzero(outer_groups == og)
        sel = inner_select(X[tr], y[tr], outer_groups[tr], ALPHAS, PCA_GRID)
        pred, m, mu, sd = _fit_ridge(X[tr], y[tr], X[te], sel["alpha"], sel["pca_n"])
        for i in te:
            j = int(np.where(te == i)[0][0])
            errs.append({"construct_id": ids[i], "family": families[i], "y": float(y[i]),
                         "pred": float(pred[j]), "abs_err": float(abs(pred[j] - y[i])),
                         "alpha": sel["alpha"], "pca_n": sel["pca_n"]})
    e = np.array([r["abs_err"] for r in errs])
    yv = np.array([r["y"] for r in errs])
    pv = np.array([r["pred"] for r in errs])
    ss_res = float(np.sum((yv - pv) ** 2))
    ss_tot = float(np.sum((yv - yv.mean()) ** 2))
    return {
        "outer_mode": outer_mode,
        "n": len(errs),
        "MAE_log10": float(e.mean()),
        "median_fold_error": float(np.median(10 ** e)),
        "mean_fold_error": float(np.mean(10 ** e)),
        "R2": float(1 - ss_res / ss_tot) if ss_tot > 0 else float("nan"),
        "per_construct": errs,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default=DEVICE)
    args = ap.parse_args()

    ep = pd.read_csv(CELL / "construct_endpoints_D28.tsv", sep="\t", encoding="utf-8-sig")
    ep = ep[ep.endpoint_eligible == 1].sort_values("construct_id").reset_index(drop=True)
    assert len(ep) == 14 and ep.construct_id.is_unique
    frame = ep
    assert frame.sequence.notna().all()
    for _, r in frame.iterrows():
        assert len(r.sequence) == int(r.sequence_length_bp), r.construct_id

    ids = frame.construct_id.tolist()
    families = frame.family.tolist()
    y = frame.log10_relative_gfp.to_numpy(dtype=np.float64)

    mpra = pd.read_csv(MPRA_SUP, sep="\t")
    audit = overlap_audit(frame, mpra.sequence.tolist())
    atomic_write_json(REPORT / "trackB_overlap_audit.json",
                      {"rows": audit, "n_mpra_oligos": int(len(mpra)),
                       "note": "exact + 20-mer hit fraction of each construct against MPRA training oligos"})

    models = load_ensemble(args.device)
    E = np.zeros((len(ids), 128), dtype=np.float64)
    emb_rows = []
    for i, r in frame.iterrows():
        E[i] = embed_construct(models, r.sequence, args.device)
        d = {"construct_id": r.construct_id, "family": r.family}
        for j in range(128):
            d["e%d" % j] = float(E[i, j])
        emb_rows.append(d)
    pd.DataFrame(emb_rows).to_csv(REPORT / "trackB_embeddings.tsv", sep="\t", index=False)

    length = np.log10(frame.sequence_length_bp.to_numpy(dtype=np.float64))[:, None]
    gc = np.array([gc_frac(s) for s in frame.sequence])[:, None]
    X_simple = np.concatenate([length, gc], axis=1)
    X_mpra = np.concatenate([E, length, gc], axis=1)

    results = {"constructs": ids, "families": families, "target": y.tolist(),
               "methods": {}}
    for mid, X in [("length_gc_ridge", X_simple), ("frozen_mpra_legnet_length_gc_ridge", X_mpra)]:
        results["methods"][mid] = {
            "construct_LOO": run_protocol(X, y, ids, families, "construct_LOO"),
            "family_LOO": run_protocol(X, y, ids, families, "family_LOO"),
        }
    atomic_write_json(REPORT / "trackB_results.json", results)
    print(json.dumps({k: {kk: {kkk: vv for kkk, vv in v.items() if kkk != "per_construct"}
                           for kk, v in results["methods"][k].items()}
                      for k in results["methods"]}, indent=2))
    print("DONE")


if __name__ == "__main__":
    main()
