#!/usr/bin/env python3
"""S0.1: split-half label reliability for E/S/B (the model's r2 ceiling).

Splits GTEx samples within each tissue into random halves, computes E/S/B from
each half with the exact label formulas, and measures split-half Spearman
correlation + Spearman-Brown-corrected reliability (= the theoretical r2 ceiling
for a model trained on the full-data labels).

All repeats are accumulated in a SINGLE pass over the gct.gz.

Outputs:
    labels/qc/label_reliability_splithalf.tsv
"""

import gzip
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[2]
GCT = ROOT / "data/raw/GTEx_Analysis_v10_RNASeQCv2.4.2_gene_tpm.gct.gz"
ATT = ROOT / "data/raw/GTEx_Analysis_v10_Annotations_SampleAttributesDS.txt"
GTABLE = ROOT / "labels/build/gene_expression_table.tsv"
OUT = ROOT / "labels/qc/label_reliability_splithalf.tsv"

N_REPEATS = 5
CHUNK = 3000


def main():
    tissues = sorted(pd.read_csv(GTABLE, sep="\t", usecols=["tissue"])["tissue"].unique())
    print("label tissues:", len(tissues), flush=True)

    att = pd.read_csv(ATT, sep="\t", usecols=["SAMPID", "SMTSD"], low_memory=False)
    att = att[att["SMTSD"].isin(tissues)].reset_index(drop=True)
    print("samples with label tissue:", len(att), flush=True)

    with gzip.open(GCT, "rt") as f:
        f.readline()
        f.readline()
        cols = f.readline().strip().split("\t")
    samples = cols[2:]
    print("gct samples:", len(samples), flush=True)

    samp_tissue = att.set_index("SAMPID")["SMTSD"]
    keep = [i for i, s in enumerate(samples) if s in samp_tissue.index]
    sel_samples = [samples[i] for i in keep]
    sel_tissues = [samp_tissue[s] for s in sel_samples]
    n_sel = len(sel_samples)
    tissue_set = sorted(set(sel_tissues))
    n_groups = len(tissue_set) * 2
    print("selected samples:", n_sel, "| tissues:", len(tissue_set), flush=True)

    rng = np.random.default_rng(20260910)
    onehots = []
    for rep in range(N_REPEATS):
        gidx = np.zeros(n_sel, dtype=np.int64)
        for ti, t in enumerate(tissue_set):
            tidx = [i for i, tt in enumerate(sel_tissues) if tt == t]
            perm = rng.permutation(len(tidx))
            half = len(perm) // 2
            a_set = set(perm[:half].tolist())
            for k, i in enumerate(tidx):
                gidx[i] = ti * 2 + (0 if k in a_set else 1)
        oh = np.zeros((n_sel, n_groups), dtype=np.float32)
        oh[np.arange(n_sel), gidx] = 1.0
        onehots.append(oh)
    counts = np.array([oh.sum(axis=0) for oh in onehots])  # (R, n_groups)
    sums = [np.zeros((0, n_groups), dtype=np.float64) for _ in range(N_REPEATS)]
    gene_ids = []

    usecols = [0] + [2 + i for i in keep]
    reader = pd.read_csv(GCT, sep="\t", skiprows=2, chunksize=CHUNK, usecols=usecols,
                         index_col=0, compression="gzip")
    for chunk in reader:
        v = chunk.to_numpy(dtype=np.float32)
        for r in range(N_REPEATS):
            sums[r] = np.vstack([sums[r], v @ onehots[r]])
        gene_ids.extend(chunk.index.tolist())
    n_genes = len(gene_ids)
    print("genes:", n_genes, flush=True)

    tidx = {t: k for k, t in enumerate(tissue_set)}
    results = []

    def esb(m):
        z = np.log2(1.0 + m)
        with np.errstate(invalid="ignore", divide="ignore"):
            e = np.nanmean(z, axis=1)
            det = m >= 1.0
            n_det = det.sum(axis=1)
            n_fin = np.isfinite(m).sum(axis=1)
            b = np.where(n_fin > 0, n_det / n_fin, np.nan)
            zd = np.where(det, z, np.nan)
            med = np.nanmedian(zd, axis=1)
            mad = np.nanmedian(np.abs(zd - med[:, None]), axis=1)
            s = -mad
            s_gated = np.where((n_det >= 3) & (b >= 0.3), s, np.nan)
        return e, s, b, s_gated

    for rep in range(N_REPEATS):
        means = sums[rep] / counts[rep][None, :]
        ma = np.full((n_genes, len(tissue_set)), np.nan)
        mb = np.full((n_genes, len(tissue_set)), np.nan)
        for t in tissue_set:
            ti = tidx[t]
            ma[:, ti] = means[:, ti * 2]
            mb[:, ti] = means[:, ti * 2 + 1]
        ea, sa, ba, sga = esb(ma)
        eb, sb, bb, sgb = esb(mb)
        for label, xa, xb in [("E_raw", ea, eb), ("S_raw_ungated", sa, sb),
                              ("B_raw", ba, bb), ("S_raw_gated", sga, sgb)]:
            m = np.isfinite(xa) & np.isfinite(xb)
            n = int(m.sum())
            if n < 100:
                continue
            r = spearmanr(xa[m], xb[m]).statistic
            r_full = 2.0 * r / (1.0 + r) if r > -1 else float("nan")
            results.append({"repeat": rep, "axis": label, "n_genes": n,
                            "split_half_spearman": float(r),
                            "sb_corrected_reliability": float(r_full)})
            print("rep %d %s: n=%d r_half=%.4f r_full=%.4f" % (rep, label, n, r, r_full), flush=True)

    res = pd.DataFrame(results)
    res.to_csv(OUT, sep="\t", index=False)
    summ = res.groupby("axis").agg(
        r_half_mean=("split_half_spearman", "mean"),
        r_half_sd=("split_half_spearman", "std"),
        r_full_mean=("sb_corrected_reliability", "mean"),
        n_mean=("n_genes", "mean"),
    ).round(4)
    print("\n=== summary (mean over repeats) ===", flush=True)
    print(summ.to_string(), flush=True)
    print("\nwrote", OUT, flush=True)
    print("ceiling r2 per axis = r_full_mean; current model test r2: E .3167 / S .1147 / B .2849", flush=True)


if __name__ == "__main__":
    main()
