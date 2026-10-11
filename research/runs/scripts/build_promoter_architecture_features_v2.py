"""
Build promoter architecture features v2 for promoter_atlas.

Inputs
------
- promoters/memmap/promoter_up200_uint8.memmap
- promoters/memmap/promoter_up200_index.tsv
- labels/build/promoter_model_input_core.tsv

Outputs
-------
- features/sequence/promoter_architecture_features_v2.tsv

Features
--------
overall:
- gc_content
- cpg_density
- cpg_obs_exp
- gc_skew
- dinuc_entropy
- max_polyA_run
- max_polyT_run

zone-aware:
- distal_gc_content
- proximal_gc_content
- gc_content_delta_prox_minus_dist
- distal_cpg_density
- proximal_cpg_density
- cpg_density_delta_prox_minus_dist
- distal_cpg_obs_exp
- proximal_cpg_obs_exp
- distal_dinuc_entropy
- proximal_dinuc_entropy

proxy:
- cpg_island_like_flag
"""

from pathlib import Path
import numpy as np
import pandas as pd


ROOT = Path(".")

MEMMAP_PATH = ROOT / "promoters/memmap/promoter_up200_uint8.memmap"
MEMMAP_INDEX_PATH = ROOT / "promoters/memmap/promoter_up200_index.tsv"
INPUT_TABLE = ROOT / "labels/build/promoter_model_input_core.tsv"

OUT_PATH = ROOT / "features/sequence/promoter_architecture_features_v2.tsv"

SEQ_LEN = 200
CHUNK_SIZE = 50000

# index 0..199 corresponds to relative positions -200..-1
DISTAL_SLICE = slice(0, 120)     # [-200, -81]
PROXIMAL_SLICE = slice(120, 200) # [-80, -1]


def max_run_of_value(arr, value):
    best = 0
    cur = 0
    for x in arr:
        if x == value:
            cur += 1
            if cur > best:
                best = cur
        else:
            cur = 0
    return int(best)


def dinuc_entropy(arr):
    valid = arr[arr <= 3]
    if len(valid) < 2:
        return 0.0

    counts = np.zeros(16, dtype=np.int64)
    left = valid[:-1]
    right = valid[1:]
    pair_idx = left * 4 + right
    uniq, cnt = np.unique(pair_idx, return_counts=True)
    counts[uniq] = cnt

    total = counts.sum()
    if total == 0:
        return 0.0

    p = counts[counts > 0] / total
    return float(-(p * np.log2(p)).sum())


def basic_stats(arr):
    arr = np.asarray(arr, dtype=np.uint8)
    valid = arr[arr <= 3]
    n = len(valid)

    if n == 0:
        return {
            "gc_content": np.nan,
            "cpg_density": np.nan,
            "cpg_obs_exp": np.nan,
            "gc_skew": np.nan,
            "dinuc_entropy": np.nan,
            "max_polyA_run": np.nan,
            "max_polyT_run": np.nan,
        }

    a = int((valid == 0).sum())
    c = int((valid == 1).sum())
    g = int((valid == 2).sum())
    t = int((valid == 3).sum())

    gc = g + c
    gc_content = gc / n

    if n >= 2:
        cp = int(((valid[:-1] == 1) & (valid[1:] == 2)).sum())
        cpg_density = cp / (n - 1)
    else:
        cp = 0
        cpg_density = 0.0

    if c > 0 and g > 0:
        cpg_obs_exp = (cp * n) / (c * g)
    else:
        cpg_obs_exp = 0.0

    denom = g + c
    gc_skew = (g - c) / denom if denom > 0 else 0.0

    return {
        "gc_content": float(gc_content),
        "cpg_density": float(cpg_density),
        "cpg_obs_exp": float(cpg_obs_exp),
        "gc_skew": float(gc_skew),
        "dinuc_entropy": float(dinuc_entropy(valid)),
        "max_polyA_run": float(max_run_of_value(valid, 0)),
        "max_polyT_run": float(max_run_of_value(valid, 3)),
    }


def compute_features(seq):
    seq = np.asarray(seq, dtype=np.uint8)

    overall = basic_stats(seq)
    distal = basic_stats(seq[DISTAL_SLICE])
    proximal = basic_stats(seq[PROXIMAL_SLICE])

    cpg_island_like_flag = int(
        (overall["gc_content"] >= 0.55)
        and (overall["cpg_obs_exp"] >= 0.60)
    )

    return {
        "gc_content": overall["gc_content"],
        "cpg_density": overall["cpg_density"],
        "cpg_obs_exp": overall["cpg_obs_exp"],
        "gc_skew": overall["gc_skew"],
        "dinuc_entropy": overall["dinuc_entropy"],
        "max_polyA_run": overall["max_polyA_run"],
        "max_polyT_run": overall["max_polyT_run"],

        "distal_gc_content": distal["gc_content"],
        "proximal_gc_content": proximal["gc_content"],
        "gc_content_delta_prox_minus_dist": proximal["gc_content"] - distal["gc_content"],

        "distal_cpg_density": distal["cpg_density"],
        "proximal_cpg_density": proximal["cpg_density"],
        "cpg_density_delta_prox_minus_dist": proximal["cpg_density"] - distal["cpg_density"],

        "distal_cpg_obs_exp": distal["cpg_obs_exp"],
        "proximal_cpg_obs_exp": proximal["cpg_obs_exp"],

        "distal_dinuc_entropy": distal["dinuc_entropy"],
        "proximal_dinuc_entropy": proximal["dinuc_entropy"],

        "cpg_island_like_flag": float(cpg_island_like_flag),
    }


def main():
    if not MEMMAP_PATH.exists():
        raise FileNotFoundError(f"Missing memmap file: {MEMMAP_PATH}")
    if not MEMMAP_INDEX_PATH.exists():
        raise FileNotFoundError(f"Missing memmap index file: {MEMMAP_INDEX_PATH}")
    if not INPUT_TABLE.exists():
        raise FileNotFoundError(f"Missing input table: {INPUT_TABLE}")

    mm_index = pd.read_csv(MEMMAP_INDEX_PATH, sep="\t", low_memory=False)
    core = pd.read_csv(INPUT_TABLE, sep="\t", low_memory=False, usecols=["record_id", "gene_id_base"])

    merged = core.merge(mm_index[["record_id", "row_index"]], on="record_id", how="inner")
    if merged["record_id"].duplicated().any():
        dup_n = int(merged["record_id"].duplicated().sum())
        raise ValueError(f"Duplicated record_id after merge: {dup_n}")

    merged = merged.sort_values("record_id").reset_index(drop=True)

    n_rows_mm = len(mm_index)
    mm = np.memmap(
        MEMMAP_PATH,
        dtype="uint8",
        mode="r",
        shape=(n_rows_mm, SEQ_LEN),
    )

    rows = []
    total = len(merged)

    for left in range(0, total, CHUNK_SIZE):
        right = min(left + CHUNK_SIZE, total)
        chunk = merged.iloc[left:right]

        row_indices = chunk["row_index"].to_numpy(dtype=np.int64)
        seqs = np.asarray(mm[row_indices], dtype=np.uint8)

        for i, row in enumerate(chunk.itertuples(index=False)):
            feats = compute_features(seqs[i])
            rows.append({
                "record_id": row.record_id,
                "gene_id_base": row.gene_id_base,
                **feats,
            })

        print(f"[arch_v2] {right}/{total} done")

    out_df = pd.DataFrame(rows)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(OUT_PATH, sep="\t", index=False)

    print("architecture v2 feature build finished")
    print(f"out: {OUT_PATH}")
    print(f"rows: {len(out_df)}")
    print(f"columns: {len(out_df.columns)}")


if __name__ == "__main__":
    main()
