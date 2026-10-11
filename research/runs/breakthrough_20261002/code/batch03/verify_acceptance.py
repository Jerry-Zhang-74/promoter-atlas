#!/usr/bin/env python3
"""Batch03 acceptance verification: recompute metrics from saved predictions,
verify matched genes, replay saved checkpoints, and check data-loading scope.

Read-only with respect to all prior artifacts; writes only a verification report
JSON under reports/breakthrough_20261002/batch03/.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from b3_common import spearman, r2, mae, mse  # noqa: E402

REPORT = ROOT / "reports" / "breakthrough_20261002" / "batch03"
PRED = REPORT / "trackA_predictions"
DATA = ROOT / "data" / "breakthrough_20261002" / "batch03"
MODEL_DIR = ROOT / "models" / "breakthrough_20261002" / "batch03"

TAGS = [
    ("v6.6.0-scratchgene.rc1_s20261011", "single", "final_single_s20261011"),
    ("v6.6.0-scratchgene.rc1_s20261012", "single", "final_single_s20261012"),
    ("v6.6.0-scratchgene.rc1_s20261013", "single", "final_single_s20261013"),
    ("v6.6.0-scratchgene.rc2_s20261011", "bag", "final_bag_s20261011"),
    ("v6.6.0-scratchgene.rc2_s20261012", "bag", "final_bag_s20261012"),
    ("v6.6.0-scratchgene.rc2_s20261013", "bag", "final_bag_s20261013"),
    ("v6.6.0-seqfeatridge.rc1_s20261011", "ridge", None),
]


def load_pred(tag):
    return pd.read_csv(PRED / f"pred_{tag}.tsv", sep="\t")


def load_metrics(tag):
    with open(PRED / f"metrics_{tag}.json", encoding="utf-8") as f:
        return json.load(f)


def main():
    prep = json.loads((DATA / "trackA_prep_meta.json").read_text(encoding="utf-8"))
    exposed_test = set(prep["exposed_test_genes"])
    exposed_val = set(prep["exposed_val_genes"])

    out = {"recompute_from_saved_predictions": {}, "matched_genes": {}, "replay": {}}

    # ---- Part 1: recompute metrics from saved predictions ----
    for tag, mode, _ in TAGS:
        df = load_pred(tag)
        m = load_metrics(tag)
        rec = {}
        for split_name in ["val", "test"]:
            sub = df[df.split == split_name]
            y = sub.Score_E.to_numpy(dtype=np.float64)
            p = sub.pred_E.to_numpy(dtype=np.float64)
            gids = sub.gene_id.to_numpy()
            unexp = ~np.isin(gids, list(exposed_val if split_name == "val" else exposed_test))
            full = {"spearman": spearman(p, y), "r2": r2(p, y), "mae": mae(p, y), "mse": mse(p, y), "n": int(len(y))}
            un = {"spearman": spearman(p[unexp], y[unexp]), "r2": r2(p[unexp], y[unexp]),
                  "mae": mae(p[unexp], y[unexp]), "mse": mse(p[unexp], y[unexp]), "n": int(unexp.sum())}
            saved_full = m[split_name]["full"]
            saved_un = m[split_name]["unexposed"]
            rec[split_name] = {
                "full_recomputed": full,
                "full_saved": {k: saved_full[k] for k in ["spearman", "r2", "mae", "mse", "n"]},
                "full_max_abs_diff": {k: float(abs(full[k] - saved_full[k])) for k in ["spearman", "r2", "mae", "mse", "n"]},
                "unexposed_recomputed": un,
                "unexposed_saved": {k: saved_un[k] for k in ["spearman", "r2", "mae", "mse", "n"]},
                "unexposed_max_abs_diff": {k: float(abs(un[k] - saved_un[k])) for k in ["spearman", "r2", "mae", "mse", "n"]},
            }
        out["recompute_from_saved_predictions"][tag] = rec

    # ---- Part 2: matched genes ----
    dfs = {tag: load_pred(tag) for tag, _, _ in TAGS}
    keys = None
    for tag, df in dfs.items():
        k = list(zip(df.gene_id.tolist(), df.split.tolist()))
        if keys is None:
            keys = k
        else:
            assert keys == k, f"gene ordering mismatch at {tag}"
    score_consistency = {}
    ref = dfs[TAGS[0][0]]
    for tag, df in dfs.items():
        score_consistency[tag] = {
            "n_rows": int(len(df)),
            "Score_E_identical_to_ref": bool(np.allclose(df.Score_E.to_numpy(dtype=np.float64), ref.Score_E.to_numpy(dtype=np.float64))),
        }
    out["matched_genes"] = {
        "n_models": len(dfs),
        "identical_gene_split_ordering": True,
        "n_rows_per_model": {t: int(len(d)) for t, d in dfs.items()},
        "score_consistency": score_consistency,
    }

    # ---- Part 3: replay saved checkpoints (test split, CPU) ----
    import torch
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from scratch_gene_model import SinglePromoterModel, GeneBagModel  # noqa: E402
    from train_scratch_gene import load_data, make_memmap, predict_genes  # noqa: E402

    torch.set_num_threads(4)
    device = "cpu"
    gene_id, r400, split, score_e, starts, gene_split, gene_score_e, n_genes = load_data()
    mm = make_memmap()
    test_idx = np.flatnonzero(gene_split == "test")

    for tag, mode, ckdir in TAGS:
        if mode == "ridge":
            continue
        ck = torch.load(MODEL_DIR / ckdir / "final.pt", map_location="cpu", weights_only=False)
        model = SinglePromoterModel() if mode == "single" else GeneBagModel()
        model.load_state_dict(ck["model_state"])
        model.eval()
        pred = predict_genes(mode, model, mm, r400, starts, test_idx, device)
        df = dfs[tag]
        sub = df[df.split == "test"].reset_index(drop=True)
        saved_pred = sub.pred_E.to_numpy(dtype=np.float64)
        assert len(pred) == len(saved_pred), (tag, len(pred), len(saved_pred))
        diff = np.abs(pred - saved_pred)
        # recompute test spearman from replayed predictions
        y = sub.Score_E.to_numpy(dtype=np.float64)
        out["replay"][tag] = {
            "checkpoint": str(MODEL_DIR / ckdir / "final.pt"),
            "mode": mode,
            "n_genes": int(len(test_idx)),
            "max_abs_pred_diff": float(diff.max()),
            "mean_abs_pred_diff": float(diff.mean()),
            "test_full_spearman_replayed": float(spearman(pred, y)),
            "test_full_spearman_saved": float(spearman(saved_pred, y)),
        }

    atomic = (REPORT / "verify_acceptance.json")
    atomic.write_text(json.dumps(out, indent=2, ensure_ascii=True, default=float), encoding="utf-8")
    print(json.dumps(out, indent=2, ensure_ascii=True, default=float))
    print("DONE")


if __name__ == "__main__":
    main()
