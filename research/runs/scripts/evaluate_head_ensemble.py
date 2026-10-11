#!/usr/bin/env python3
"""Head-level ensemble evaluation: pick the best head per target.

Candidates:
  - v2_conv_both (200 bp + arch + motif)
  - v2_dual_nobn (dual-stream, no BN)
  - v2_dual_nobn_zs / v2_long_nobn_zs (z-score-S variants, if available)

Computes test predictions for each model and reports the metrics of the
head-level combination (E/B from the dual model, S from whichever candidate
has the best S head), plus each model's individual metrics.
"""

import importlib.util
import numpy as np
import pandas as pd
import torch
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

INPUT_TABLE = ROOT / "labels/build/promoter_model_input_core.tsv"
SPLIT_TABLE = ROOT / "labels/releases/promoter_split_core.tsv"
MM200_PATH = ROOT / "promoters/memmap/promoter_up200_uint8.memmap"
MM200_INDEX = ROOT / "promoters/memmap/promoter_up200_index.tsv"
MM2000_PATH = ROOT / "promoters/memmap/promoter_up2000_uint8.memmap"
MM2000_INDEX = ROOT / "promoters/memmap/promoter_up2000_index.tsv"
ARCH_FEATURES = ROOT / "features/sequence/promoter_architecture_features_v4_arch2_reduced.tsv"
MOTIF_FEATURES = ROOT / "features/motif/promoter_motif_reduced_v1.tsv"


def rankdata(x):
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(len(x), dtype=np.float64)
    i = 0
    while i < len(x):
        j = i + 1
        while j < len(x) and x[order[j]] == x[order[i]]:
            j += 1
        ranks[order[i:j]] = 0.5 * (i + j - 1)
        i = j
    return ranks


def spearman(a, b):
    ra = rankdata(a); rb = rankdata(b)
    ra = ra - ra.mean(); rb = rb - rb.mean()
    return float((ra * rb).sum() / np.sqrt((ra ** 2).sum() * (rb ** 2).sum()))


def r2(a, b):
    return 1.0 - float(((a - b) ** 2).sum()) / float(((b - b.mean()) ** 2).sum())


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main():
    fusion_mod = load_module("fusion", ROOT / "runs/scripts/train_promoter_model_v2_fusion.py")
    dual_mod = load_module("dual", ROOT / "runs/scripts/train_promoter_model_v2_dual.py")
    long_mod = load_module("long", ROOT / "runs/scripts/train_promoter_model_v2_longcontext.py")

    mm200 = np.memmap(MM200_PATH, dtype="uint8", mode="r", shape=(len(pd.read_csv(MM200_INDEX, sep="\t")), 200))
    mm2000 = np.memmap(MM2000_PATH, dtype="uint8", mode="r", shape=(len(pd.read_csv(MM2000_INDEX, sep="\t")), 2000))

    # ---- 200 bp model (fusion) on test ----
    df = fusion_mod.load_dataframe("both")
    feat_cols = [c for c in df.columns if c not in fusion_mod.BASE_COLS]
    test_df = df[df["split"] == "test"].reset_index(drop=True)
    y_te, m_te = fusion_mod.build_targets(test_df)
    f_te = fusion_mod.standardize_features(test_df[feat_cols].to_numpy(),
                                           torch.load(ROOT / "models/checkpoints/v2_conv_both.pt",
                                                      map_location="cpu", weights_only=False)["feat_stats"])[0]
    rows200 = test_df["row_index"].to_numpy(dtype=np.int64)
    x200 = np.asarray(mm200[rows200], dtype=np.uint8)

    ck = torch.load(ROOT / "models/checkpoints/v2_conv_both.pt", map_location="cpu", weights_only=False)
    m200 = fusion_mod.ConvBaseline(3, feat_dim=len(feat_cols)).to(DEVICE)
    m200.load_state_dict(ck["state_dict"])
    m200.eval()
    preds200 = []
    with torch.no_grad():
        for i in range(0, len(x200), 256):
            xb = torch.from_numpy(x200[i:i + 256]).to(DEVICE)
            fb = torch.from_numpy(f_te[i:i + 256]).to(DEVICE)
            preds200.append(m200(xb, fb).cpu().numpy())
    preds200 = np.concatenate(preds200, axis=0)

    # ---- dual model on test ----
    ddf = dual_mod.load_dataframe("none")
    dtest = ddf[ddf["split"] == "test"].reset_index(drop=True)
    y_d, m_d = dual_mod.build_targets(dtest)
    r2rows = dtest["r200"].to_numpy(dtype=np.int64)
    r20rows = dtest["r2000"].to_numpy(dtype=np.int64)
    x2 = np.asarray(mm200[r2rows], dtype=np.uint8)
    x20 = np.asarray(mm2000[r20rows], dtype=np.uint8)

    ckd = torch.load(ROOT / "models/checkpoints/v2_dual_nobn.pt", map_location="cpu", weights_only=False)
    md = dual_mod.DualStreamNet(3, feat_dim=0, norm="none").to(DEVICE)
    md.load_state_dict(ckd["state_dict"])
    md.eval()
    predsD = []
    with torch.no_grad():
        for i in range(0, len(x2), 256):
            a = torch.from_numpy(x2[i:i + 256]).to(DEVICE)
            b = torch.from_numpy(x20[i:i + 256]).to(DEVICE)
            predsD.append(md(a, b).cpu().numpy())
    predsD = np.concatenate(predsD, axis=0)

    models = {
        "v2_conv_both": (preds200, y_te, m_te),
        "v2_dual_nobn": (predsD, y_d, m_d),
    }

    # recompute s_stats identically to the training scripts (checkpoints don't save it)
    def compute_s_stats(y, m):
        v = y[m[:, 1] > 0, 1]
        return float(v.mean()), float(v.std()) + 1e-6

    # optional z-score-S variants
    for name, path, loader in [
        ("v2_dual_nobn_zs", ROOT / "models/checkpoints/v2_dual_nobn_zs.pt", "dual"),
        ("v2_long_nobn_zs", ROOT / "models/checkpoints/v2_long_nobn_zs.pt", "long"),
    ]:
        if not path.exists():
            continue
        ckz = torch.load(path, map_location="cpu", weights_only=False)
        if loader == "dual":
            mz = dual_mod.DualStreamNet(3, feat_dim=0, norm="none").to(DEVICE)
            mz.load_state_dict(ckz["state_dict"])
            mz.eval()
            pz = []
            with torch.no_grad():
                for i in range(0, len(x2), 256):
                    a = torch.from_numpy(x2[i:i + 256]).to(DEVICE)
                    b = torch.from_numpy(x20[i:i + 256]).to(DEVICE)
                    pz.append(mz(a, b).cpu().numpy())
            pz = np.concatenate(pz, axis=0)
            # unstandardize S head with recomputed train stats
            mu, sd = compute_s_stats(
                dual_mod.build_targets(ddf[ddf["split"] == "train"].reset_index(drop=True))[0],
                dual_mod.build_targets(ddf[ddf["split"] == "train"].reset_index(drop=True))[1])
            pz[:, 1] = pz[:, 1] * sd + mu
            models[name] = (pz, y_d, m_d)
        else:
            ldf = long_mod.load_dataframe("none")
            ltest = ldf[ldf["split"] == "test"].reset_index(drop=True)
            y_l, m_l = long_mod.build_targets(ltest)
            rowsL = ltest["row_index"].to_numpy(dtype=np.int64)
            xL = np.asarray(mm2000[rowsL], dtype=np.uint8)
            mz = long_mod.LongContextNet(3, feat_dim=0, norm="none").to(DEVICE)
            mz.load_state_dict(ckz["state_dict"])
            mz.eval()
            pz = []
            with torch.no_grad():
                for i in range(0, len(xL), 256):
                    a = torch.from_numpy(xL[i:i + 256]).to(DEVICE)
                    pz.append(mz(a).cpu().numpy())
            pz = np.concatenate(pz, axis=0)
            ltrain = ldf[ldf["split"] == "train"].reset_index(drop=True)
            y_lt, m_lt = long_mod.build_targets(ltrain)
            mu, sd = compute_s_stats(y_lt, m_lt)
            pz[:, 1] = pz[:, 1] * sd + mu
            models[name] = (pz, y_l, m_l)

    print("=== individual models (test) ===")
    for name, (p, y, m) in models.items():
        line = [name]
        for j in range(3):
            mask = m[:, j] > 0
            line.append("r2=%.4f sp=%.4f" % (r2(p[mask, j], y[mask, j]), spearman(p[mask, j], y[mask, j])))
        print("  " + " | ".join(line))

    # ---- S0.3: gene-level aggregation + bootstrap CI ----
    base = models["v2_dual_nobn"]
    y_base, m_base = base[1], base[2]
    best_s = {}
    for name, (p, y, m) in models.items():
        mask = m[:, 1] > 0
        best_s[name] = spearman(p[mask, 1], y[mask, 1])
    best_s_name = max(best_s, key=best_s.get)
    pS = models[best_s_name][0]
    comb = base[0].copy()
    comb[:, 1] = pS[:, 1]

    gid = dtest["gene_id_base"].to_numpy()
    gene_level_rows = []
    for j, label in enumerate(["E", "S", "B"]):
        mask = m_base[:, j] > 0
        dfg = pd.DataFrame({"g": gid[mask], "p": comb[mask, j], "y": y_base[mask, j]})
        agg = dfg.groupby("g").agg(p=("p", "mean"), y=("y", "mean")).reset_index()
        n_genes = len(agg)
        r2g = r2(agg["p"].to_numpy(), agg["y"].to_numpy())
        spg = spearman(agg["p"].to_numpy(), agg["y"].to_numpy())
        rng = np.random.default_rng(42)
        b_r2, b_sp = [], []
        for _ in range(1000):
            s = agg.sample(n=n_genes, replace=True, random_state=int(rng.integers(2**31 - 1)))
            b_r2.append(r2(s["p"].to_numpy(), s["y"].to_numpy()))
            b_sp.append(spearman(s["p"].to_numpy(), s["y"].to_numpy()))
        b_r2 = np.array(b_r2); b_sp = np.array(b_sp)
        gene_level_rows.append({
            "axis": label, "n_promoters": int(mask.sum()), "n_genes": n_genes,
            "gene_r2": r2g, "gene_r2_lo95": float(np.percentile(b_r2, 2.5)),
            "gene_r2_hi95": float(np.percentile(b_r2, 97.5)),
            "gene_spearman": spg, "gene_sp_lo95": float(np.percentile(b_sp, 2.5)),
            "gene_sp_hi95": float(np.percentile(b_sp, 97.5)),
        })
        print("  gene-level %s: n_genes=%d r2=%.4f [%.4f, %.4f] sp=%.4f [%.4f, %.4f]"
              % (label, n_genes, r2g, gene_level_rows[-1]["gene_r2_lo95"], gene_level_rows[-1]["gene_r2_hi95"],
                 spg, gene_level_rows[-1]["gene_sp_lo95"], gene_level_rows[-1]["gene_sp_hi95"]))
    pd.DataFrame(gene_level_rows).to_csv(
        ROOT / "models/eval/genelevel_ensemble_bootstrap.tsv", sep="\t", index=False)
    print("wrote models/eval/genelevel_ensemble_bootstrap.tsv")

    print("\n=== head-level ensemble (E/B from dual, S = best S head) ===")
    base = models["v2_dual_nobn"]
    y_base, m_base = base[1], base[2]
    best_s = {}
    for name, (p, y, m) in models.items():
        mask = m[:, 1] > 0
        best_s[name] = (spearman(p[mask, 1], y[mask, 1]), r2(p[mask, 1], y[mask, 1]))
        print("  S head %s: sp=%.4f r2=%.4f" % (name, best_s[name][0], best_s[name][1]))
    best_s_name = max(best_s, key=lambda k: best_s[k][0])
    pS, _, _ = models[best_s_name]
    print("  -> using S head from", best_s_name)
    for j, label in enumerate(["E", "S", "B"]):
        mask = m_base[:, j] > 0
        if j == 1:
            pj = pS[mask, 1]
        else:
            pj = base[0][mask, j]
        yj = y_base[mask, j]
        print("  combined %s: r2=%.4f sp=%.4f" % (label, r2(pj, yj), spearman(pj, yj)))


if __name__ == "__main__":
    main()
