"""
Build reduced architecture feature table from architecture v2 for promoter_atlas.

Inputs
------
- features/sequence/promoter_architecture_features_v2.tsv

Outputs
-------
- features/sequence/promoter_architecture_features_v4_arch2_reduced.tsv
"""

from pathlib import Path
import pandas as pd


ROOT = Path(".")

SRC = ROOT / "features/sequence/promoter_architecture_features_v2.tsv"
OUT = ROOT / "features/sequence/promoter_architecture_features_v4_arch2_reduced.tsv"

KEEP_COLS = [
    "record_id",
    "gene_id_base",
    "gc_content",
    "cpg_density",
    "cpg_obs_exp",
    "gc_skew",
    "dinuc_entropy",
    "distal_gc_content",
    "proximal_gc_content",
    "gc_content_delta_prox_minus_dist",
    "distal_cpg_density",
    "proximal_cpg_density",
    "cpg_density_delta_prox_minus_dist",
    "distal_cpg_obs_exp",
    "proximal_cpg_obs_exp",
    "distal_dinuc_entropy",
    "proximal_dinuc_entropy",
    "cpg_island_like_flag",
]


def main():
    if not SRC.exists():
        raise FileNotFoundError(f"Missing source architecture table: {SRC}")

    df = pd.read_csv(SRC, sep="\t", low_memory=False)

    missing = [c for c in KEEP_COLS if c not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns in source table: {missing}")

    out_df = df[KEEP_COLS].copy()

    OUT.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(OUT, sep="\t", index=False)

    print("reduced architecture table written")
    print(f"out: {OUT}")
    print(f"rows: {len(out_df)}")
    print(f"columns: {len(out_df.columns)}")


if __name__ == "__main__":
    main()
