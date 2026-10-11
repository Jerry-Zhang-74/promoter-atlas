"""
Build promoter motif pair features v1 for promoter_atlas.

This script builds a small grammar-lite pair feature table from:
- features/motif/promoter_motif_hits_v1_exact_unique.tsv
- features/motif/motif_pair_whitelist_v1.tsv

Outputs
-------
- features/motif/promoter_motif_pair_features_v1.tsv
- features/motif/promoter_motif_pair_feature_summary_v1.tsv

Notes
-----
1. Only pairs whose both motifs exist in the full hits table are kept.
2. Distance is based on best-hit positions, not all-hit minimum spacing.
3. This is a grammar-lite pilot layer, not the final grammar system.
"""

from pathlib import Path
import numpy as np
import pandas as pd


ROOT = Path(".")

HITS_PATH = ROOT / "features/motif/promoter_motif_hits_v1_exact_unique.tsv"
PAIR_WHITELIST = ROOT / "features/motif/motif_pair_whitelist_v1.tsv"

PAIR_OUT = ROOT / "features/motif/promoter_motif_pair_features_v1.tsv"
SUMMARY_OUT = ROOT / "features/motif/promoter_motif_pair_feature_summary_v1.tsv"


PROXIMAL_MIN_REL = -40
PROXIMAL_MAX_REL = -1


def besthit_rel(start):
    if pd.isna(start):
        return np.nan
    return float(start) - 200.0


def is_proximal(rel):
    if pd.isna(rel):
        return False
    return PROXIMAL_MIN_REL <= rel <= PROXIMAL_MAX_REL


def main():
    if not HITS_PATH.exists():
        raise FileNotFoundError(f"Missing hits file: {HITS_PATH}")
    if not PAIR_WHITELIST.exists():
        raise FileNotFoundError(f"Missing pair whitelist: {PAIR_WHITELIST}")

    hits = pd.read_csv(HITS_PATH, sep="\t", low_memory=False)
    pairs = pd.read_csv(PAIR_WHITELIST, sep="\t", low_memory=False)

    req_hits = [
        "record_id", "gene_id_base", "motif_id",
        "count_strict", "best_start", "best_score_norm"
    ]
    req_pairs = ["pair_id", "motif_a", "motif_b"]

    miss_hits = [c for c in req_hits if c not in hits.columns]
    miss_pairs = [c for c in req_pairs if c not in pairs.columns]
    if miss_hits:
        raise KeyError(f"Missing hits columns: {miss_hits}")
    if miss_pairs:
        raise KeyError(f"Missing pair whitelist columns: {miss_pairs}")

    available_motifs = set(hits["motif_id"].dropna().astype(str).unique().tolist())
    pairs = pairs.loc[
        pairs["motif_a"].astype(str).isin(available_motifs)
        & pairs["motif_b"].astype(str).isin(available_motifs)
    ].copy()

    if len(pairs) == 0:
        raise ValueError("No whitelist pairs available in exact-unique hits table")

    base = hits[["record_id", "gene_id_base"]].drop_duplicates().copy()
    base = base.sort_values(["record_id"]).reset_index(drop=True)

    wide = base.copy()
    summary_rows = []

    for row in pairs.itertuples(index=False):
        pair_id = row.pair_id
        motif_a = row.motif_a
        motif_b = row.motif_b

        a = hits.loc[hits["motif_id"] == motif_a, [
            "record_id", "gene_id_base", "count_strict", "best_start", "best_score_norm"
        ]].copy()
        b = hits.loc[hits["motif_id"] == motif_b, [
            "record_id", "gene_id_base", "count_strict", "best_start", "best_score_norm"
        ]].copy()

        a = a.rename(columns={
            "count_strict": "a_count_strict",
            "best_start": "a_best_start",
            "best_score_norm": "a_best_score_norm",
        })
        b = b.rename(columns={
            "count_strict": "b_count_strict",
            "best_start": "b_best_start",
            "best_score_norm": "b_best_score_norm",
        })

        merged = base.merge(a, on=["record_id", "gene_id_base"], how="left")
        merged = merged.merge(b, on=["record_id", "gene_id_base"], how="left")

        strict_pair_present = (
            (merged["a_count_strict"].fillna(0) > 0)
            & (merged["b_count_strict"].fillna(0) > 0)
        ).astype(np.int32)

        a_rel = merged["a_best_start"].map(besthit_rel)
        b_rel = merged["b_best_start"].map(besthit_rel)

        besthit_distance_bp = np.where(
            strict_pair_present.astype(bool),
            np.abs(a_rel - b_rel),
            np.nan,
        )

        besthit_both_proximal_flag = np.where(
            strict_pair_present.astype(bool),
            (a_rel.map(is_proximal) & b_rel.map(is_proximal)).astype(np.int32),
            0,
        )

        mean_best_score = np.where(
            strict_pair_present.astype(bool),
            (merged["a_best_score_norm"].fillna(np.nan) + merged["b_best_score_norm"].fillna(np.nan)) / 2.0,
            np.nan,
        )

        pair_df = pd.DataFrame({
            "record_id": merged["record_id"],
            "gene_id_base": merged["gene_id_base"],
            f"{pair_id}_strict_pair_present": strict_pair_present.astype(np.int32),
            f"{pair_id}_besthit_distance_bp": besthit_distance_bp.astype(np.float32),
            f"{pair_id}_besthit_both_proximal_flag": besthit_both_proximal_flag.astype(np.int32),
            f"{pair_id}_besthit_mean_score_norm": mean_best_score.astype(np.float32),
        })

        wide = wide.merge(pair_df, on=["record_id", "gene_id_base"], how="left")

        summary_rows.append({
            "pair_id": pair_id,
            "motif_a": motif_a,
            "motif_b": motif_b,
            "n_promoters": int(len(merged)),
            "strict_pair_presence_rate": float(strict_pair_present.mean()),
            "both_proximal_rate_among_present": float(
                np.nanmean(
                    np.where(strict_pair_present.astype(bool), besthit_both_proximal_flag, np.nan)
                )
            ),
            "mean_besthit_distance_bp_among_present": float(
                np.nanmean(besthit_distance_bp)
            ),
            "mean_besthit_mean_score_norm_among_present": float(
                np.nanmean(mean_best_score)
            ),
        })

    pair_feature_cols = [c for c in wide.columns if c not in {"record_id", "gene_id_base"}]
    for c in pair_feature_cols:
        if c.endswith("_strict_pair_present") or c.endswith("_besthit_both_proximal_flag"):
            wide[c] = wide[c].fillna(0).astype(np.int32)
        else:
            wide[c] = pd.to_numeric(wide[c], errors="coerce").astype(np.float32)

    wide.to_csv(PAIR_OUT, sep="\t", index=False)
    pd.DataFrame(summary_rows).to_csv(SUMMARY_OUT, sep="\t", index=False)

    print("pair feature build finished")
    print(f"pair_count_used: {len(summary_rows)}")
    print(f"pair_feature_table: {PAIR_OUT}")
    print(f"pair_summary_table: {SUMMARY_OUT}")


if __name__ == "__main__":
    main()
