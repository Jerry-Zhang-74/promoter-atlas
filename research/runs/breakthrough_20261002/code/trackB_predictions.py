#!/usr/bin/env python3
"""Track B stage 2c: deterministic reproduction to persist per-sequence predictions + checkpoints.

Re-trains ridge / smallcnn / legnet with the EXACT same hyperparameters and seeds
as train_trackB.py (reusing its functions), then saves per-sequence val/test
predictions and model checkpoints. Verifies reproduction against the canonical
trackB_results.json (the gate already passed; this run only materializes
per-sequence predictions and checkpoints for traceability).

Deterministic CUDA is enabled; any residual run-to-run delta is recorded.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "runs" / "breakthrough_20261002" / "code"))

import train_trackB as TB  # reuse build_cnn/train_cnn/predict_cnn/kmer_features/encode_sequences/spearman/mae/r2/rankdata

OUT = ROOT / "reports" / "breakthrough_20261002"
CKPT_DIR = ROOT / "models" / "breakthrough_20261002"
GROUPED = TB.GROUPED
SEEDS = TB.SEEDS
ALPHAS = TB.ALPHAS


def main():
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    OUT.mkdir(parents=True, exist_ok=True)
    CKPT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(GROUPED, sep="\t")
    seqs = df["sequence"].astype(str).tolist()
    y = df["activity"].to_numpy(dtype=np.float32)
    split = df["new_split"].to_numpy()
    comp = df["component_id"].to_numpy()
    oligo = df["Oligo_index"].astype(str).to_numpy()
    fam = df["family_id"].fillna("").astype(str).to_numpy()
    x = TB.encode_sequences(seqs)
    tr = split == "train"; va = split == "val"; te = split == "test"
    x_tr, y_tr = x[tr], y[tr]
    x_va, y_va = x[va], y[va]
    x_te, y_te = x[te], y[te]
    comp_tr = comp[tr]
    y_mu = float(y_tr.mean()); y_sd = float(y_tr.std())
    y_tr_z = (y_tr - y_mu) / y_sd

    cols = {"ridge": {}, "smallcnn": {}, "legnet": {}}

    # ---- ridge ----
    Xk = TB.kmer_features(seqs)
    Xk_tr, Xk_va, Xk_te = Xk[tr], Xk[va], Xk[te]
    mu_f = Xk_tr.mean(0); sc_f = Xk_tr.std(0); sc_f[sc_f < 1e-12] = 1.0
    gkf = GroupKFold(5); cv_mse = {a: [] for a in ALPHAS}
    for a in ALPHAS:
        for ti, vi in gkf.split(Xk_tr, y_tr, comp_tr):
            muf = Xk_tr[ti].mean(0); scf = Xk_tr[ti].std(0); scf[scf < 1e-12] = 1.0
            clf = Ridge(alpha=a).fit((Xk_tr[ti] - muf) / scf, y_tr[ti])
            cv_mse[a].append(float(np.mean((clf.predict((Xk_tr[vi] - muf) / scf) - y_tr[vi]) ** 2)))
    best_a = min(cv_mse, key=lambda a: np.mean(cv_mse[a]))
    clf = Ridge(alpha=best_a).fit((Xk_tr - mu_f) / sc_f, y_tr)
    cols["ridge"]["val"] = clf.predict((Xk_va - mu_f) / sc_f)
    cols["ridge"]["test"] = clf.predict((Xk_te - mu_f) / sc_f)
    cols["ridge"]["best_alpha"] = best_a
    print("ridge alpha", best_a, "test sp", round(TB.spearman(cols["ridge"]["test"], y_te), 4), flush=True)

    # ---- CNNs ----
    for name in ["smallcnn", "legnet"]:
        cols[name]["test_seeds"] = []
        cols[name]["val_seeds"] = []
        cols[name]["best_eps"] = []
        for sd in SEEDS:
            model = TB.build_cnn(name).to(TB.DEVICE)
            model, best_ep, best_mse, log = TB.train_cnn(
                model, x_tr, y_tr_z, x_va, y_va, 20, 3e-4, 256, 1e-4, sd, y_mu, y_sd)
            p_te = TB.predict_cnn(model, x_te) * y_sd + y_mu
            p_va = TB.predict_cnn(model, x_va) * y_sd + y_mu
            cols[name]["test_seeds"].append(p_te)
            cols[name]["val_seeds"].append(p_va)
            cols[name]["best_eps"].append(int(best_ep))
            torch.save({"state_dict": model.state_dict(), "seed": int(sd),
                        "best_val_epoch": int(best_ep), "best_val_mse": float(best_mse),
                        "target_scale": {"mean": y_mu, "std": y_sd}},
                       CKPT_DIR / f"trackB_{name}_seed{sd}.pt")
            print(name, "seed", sd, "best_ep", best_ep, "test sp",
                  round(TB.spearman(p_te, y_te), 4), flush=True)
        cols[name]["ensemble_val"] = np.mean(cols[name]["val_seeds"], axis=0)
        cols[name]["ensemble_test"] = np.mean(cols[name]["test_seeds"], axis=0)

    # ---- reproduction check against canonical gate ----
    canon = json.load(open(OUT / "trackB_results.json", encoding="utf-8"))
    repro = {}
    for name in ["smallcnn", "legnet"]:
        sp = TB.spearman(cols[name]["ensemble_test"], y_te)
        ma = TB.mae(cols[name]["ensemble_test"], y_te)
        csp = canon["candidates"][name]["ensemble_test_spearman"]
        repro[name] = {"reproduced_ensemble_test_spearman": sp,
                       "reproduced_ensemble_test_mae": ma,
                       "canonical_ensemble_test_spearman": csp,
                       "delta_spearman": sp - csp}
    repro["ridge"] = {"reproduced_test_spearman": TB.spearman(cols["ridge"]["test"], y_te),
                      "canonical_test_spearman": canon["candidates"]["ridge"]["test_spearman"],
                      "delta_spearman": TB.spearman(cols["ridge"]["test"], y_te) - canon["candidates"]["ridge"]["test_spearman"]}
    (OUT / "trackB_reproduction.json").write_text(json.dumps(repro, indent=2, default=float), encoding="utf-8")
    print(json.dumps(repro, indent=2), flush=True)

    # ---- per-sequence val + test predictions ----
    va_pos = np.nonzero(va)[0]
    te_pos = np.nonzero(te)[0]
    rows = []
    for m, pos in [("val", va_pos), ("test", te_pos)]:
        for k, j in enumerate(pos):
            rec = {"Oligo_index": oligo[j], "split": split[j], "activity": float(y[j]),
                   "component_id": comp[j], "family_id": fam[j],
                   "ridge_pred": float(cols["ridge"][m][k])}
            for name in ["smallcnn", "legnet"]:
                rec[f"{name}_ensemble_pred"] = float(cols[name][f"ensemble_{m}"][k])
                for si, sd in enumerate(SEEDS):
                    rec[f"{name}_seed{sd}_pred"] = float(cols[name][f"{m}_seeds"][si][k])
            rows.append(rec)
    preds = pd.DataFrame(rows)
    preds.to_csv(OUT / "trackB_sequence_predictions.tsv", sep="\t", index=False)
    print("saved", OUT / "trackB_sequence_predictions.tsv", "rows", len(preds), flush=True)


if __name__ == "__main__":
    main()
