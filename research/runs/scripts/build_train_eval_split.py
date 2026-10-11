"""
Build train/validation/test split manifests for promoter_atlas.

This script performs gene-level splitting for promoter model inputs and maps the
resulting split assignment back to promoter rows.

Formal targets
--------------
- Score_E
- Score_S
- Score_B

Key rules
---------
1. Split unit is gene_id_base, not promoter rows.
2. Score_S missing must remain missing and must never be rewritten as 0.
3. Anchor genes ACTB, GAPDH, and RPL8 are forced into train.
4. Input tables are never overwritten.
5. All paths are relative to the project root.
6. Promoter-level primary key is record_id.

Outputs
-------
- labels/releases/gene_split_manifest.tsv
- labels/releases/promoter_split_core.tsv
- labels/releases/promoter_split_extended.tsv
- labels/qc/train_eval_split_summary.tsv
- runs/train_eval_split.log
"""

from pathlib import Path
import pandas as pd
import numpy as np


ROOT = Path(".")

CORE_CANDIDATES = [
    "labels/releases/promoter_model_input_core.tsv",
    "labels/build/promoter_model_input_core.tsv",
    "data/processed/promoter_model_input_core.tsv",
]

EXTENDED_CANDIDATES = [
    "labels/releases/promoter_model_input_extended.tsv",
    "labels/build/promoter_model_input_extended.tsv",
    "data/processed/promoter_model_input_extended.tsv",
]

OUTPUT_GENE = Path("labels/releases/gene_split_manifest.tsv")
OUTPUT_CORE = Path("labels/releases/promoter_split_core.tsv")
OUTPUT_EXTENDED = Path("labels/releases/promoter_split_extended.tsv")
OUTPUT_QC = Path("labels/qc/train_eval_split_summary.tsv")
OUTPUT_LOG = Path("runs/train_eval_split.log")

RANDOM_SEED = 20260314
ANCHOR_GENES = {"ACTB", "GAPDH", "RPL8"}


def pick_existing(paths):
    for rel in paths:
        path = ROOT / rel
        if path.exists():
            return path
    raise FileNotFoundError(
        "No input file found. Checked:\n" + "\n".join(paths)
    )


def detect_column(df, candidates, label, required=True):
    for col in candidates:
        if col in df.columns:
            return col
    if required:
        raise KeyError(f"Missing required column group: {label}. Candidates: {candidates}")
    return None


def normalize_gene_id(series):
    return series.astype(str).str.replace(r"\.\d+$", "", regex=True)


def first_non_null(series):
    x = series.dropna()
    return x.iloc[0] if len(x) else np.nan


def safe_qbin(series, q, missing_label="NA"):
    values = pd.to_numeric(series, errors="coerce")
    out = pd.Series([missing_label] * len(series), index=series.index, dtype="object")

    mask = values.notna()
    if mask.sum() == 0:
        return out

    unique_n = values[mask].nunique()
    if unique_n <= 1:
        out.loc[mask] = "0"
        return out

    q_eff = min(int(q), int(unique_n))
    bins = pd.qcut(values[mask], q=q_eff, labels=False, duplicates="drop")
    out.loc[mask] = bins.astype(str)
    return out


def split_counts(n, train_frac=0.8, val_frac=0.1, test_frac=0.1):
    if n <= 4:
        return n, 0, 0

    n_val = int(round(n * val_frac))
    n_test = int(round(n * test_frac))

    if n >= 10:
        n_val = max(n_val, 1)
        n_test = max(n_test, 1)

    n_train = n - n_val - n_test

    if n_train < 1:
        deficit = 1 - n_train
        move_from_val = min(deficit, n_val)
        n_val -= move_from_val
        deficit -= move_from_val
        move_from_test = min(deficit, n_test)
        n_test -= move_from_test
        n_train = n - n_val - n_test

    return n_train, n_val, n_test


def prepare_table(df):
    gene_col = detect_column(
        df,
        ["gene_id_base", "gene_id", "ensembl_gene_id", "gencode_gene_id", "gene"],
        "gene_id",
        required=True,
    )
    record_col = detect_column(
        df,
        ["record_id"],
        "record_id",
        required=True,
    )
    symbol_col = detect_column(
        df,
        ["gene_symbol", "gene_name", "symbol", "external_gene_name"],
        "gene_symbol",
        required=False,
    )

    required_scores = ["Score_E", "Score_S", "Score_B"]
    missing_scores = [col for col in required_scores if col not in df.columns]
    if missing_scores:
        raise KeyError(f"Missing formal score columns: {missing_scores}")

    out = pd.DataFrame()
    out["gene_id_base"] = normalize_gene_id(df[gene_col])
    out["gene_symbol"] = df[symbol_col].astype(str) if symbol_col else np.nan
    out["record_id"] = df[record_col].astype(str)
    out["Score_E"] = pd.to_numeric(df["Score_E"], errors="coerce")
    out["Score_S"] = pd.to_numeric(df["Score_S"], errors="coerce")
    out["Score_B"] = pd.to_numeric(df["Score_B"], errors="coerce")

    return out, gene_col, record_col, symbol_col


def assert_no_invalid_score_s_zero_encoding(df, table_name):
    if "Score_S" not in df.columns:
        raise KeyError(f"{table_name} missing Score_S")

    score_s = pd.to_numeric(df["Score_S"], errors="coerce")

    if score_s.notna().sum() == 0:
        raise ValueError(f"{table_name} has no valid Score_S values")

    zero_frac = (score_s == 0).mean()
    if zero_frac > 0.20:
        print(
            f"[WARN] {table_name}: Score_S has unusually many zeros "
            f"({zero_frac:.4f}). Please confirm these are real ranked values "
            f"and not old missing-value encoding."
        )


def assert_gene_label_consistency(gene_label_rows):
    for col in ["Score_E", "Score_S", "Score_B"]:
        nunique = gene_label_rows.groupby("gene_id_base")[col].nunique(dropna=True)
        bad = nunique[nunique > 1]
        if len(bad) > 0:
            raise ValueError(
                f"Inconsistent {col} detected within gene_id_base. "
                f"Conflicting gene count: {len(bad)}"
            )


def assert_record_id_unique(df, table_name):
    if "record_id" not in df.columns:
        raise KeyError(f"{table_name} missing record_id")

    dup_mask = df["record_id"].astype(str).duplicated(keep=False)
    dup_n = int(dup_mask.sum())
    if dup_n > 0:
        raise ValueError(f"{table_name} has duplicated record_id rows: {dup_n}")


def build_gene_table(core_prepared, extended_prepared):
    merged = pd.concat(
        [
            core_prepared[["gene_id_base", "gene_symbol", "Score_E", "Score_S", "Score_B"]],
            extended_prepared[["gene_id_base", "gene_symbol", "Score_E", "Score_S", "Score_B"]],
        ],
        ignore_index=True,
    )

    assert_gene_label_consistency(merged)

    gene_df = merged.groupby("gene_id_base", as_index=False).agg(
        {
            "gene_symbol": first_non_null,
            "Score_E": first_non_null,
            "Score_S": first_non_null,
            "Score_B": first_non_null,
        }
    )

    return gene_df


def assign_split(gene_df):
    gene_df = gene_df.copy()

    gene_df["s_defined"] = np.where(gene_df["Score_S"].notna(), "S1", "S0")
    gene_df["e_bin"] = safe_qbin(gene_df["Score_E"], 10)
    gene_df["b_bin"] = safe_qbin(gene_df["Score_B"], 5)
    gene_df["s_bin"] = safe_qbin(gene_df["Score_S"], 5)
    gene_df.loc[gene_df["s_defined"] == "S0", "s_bin"] = "NA"

    gene_df["stratum"] = (
        gene_df["s_defined"]
        + "|E" + gene_df["e_bin"]
        + "|B" + gene_df["b_bin"]
        + "|S" + gene_df["s_bin"]
    )

    gene_df["split"] = ""

    anchor_mask = gene_df["gene_symbol"].fillna("").isin(ANCHOR_GENES)
    gene_df.loc[anchor_mask, "split"] = "train"

    rng = np.random.default_rng(RANDOM_SEED)

    non_anchor = gene_df.loc[~anchor_mask].copy()
    for _, group in non_anchor.groupby("stratum", dropna=False):
        idx = group.index.to_numpy().copy()
        rng.shuffle(idx)
        n_train, n_val, n_test = split_counts(len(idx))
        gene_df.loc[idx[:n_train], "split"] = "train"
        gene_df.loc[idx[n_train:n_train + n_val], "split"] = "val"
        gene_df.loc[idx[n_train + n_val:], "split"] = "test"

    if not gene_df["split"].isin(["train", "val", "test"]).all():
        bad_n = (~gene_df["split"].isin(["train", "val", "test"])).sum()
        raise ValueError(f"Unassigned split rows detected: {bad_n}")

    non_anchor_train = gene_df.index[
        (gene_df["split"] == "train") & (~anchor_mask)
    ].tolist()
    for target in ["val", "test"]:
        if (gene_df["split"] == target).sum() == 0 and len(non_anchor_train) > 0:
            move_idx = non_anchor_train.pop()
            gene_df.loc[move_idx, "split"] = target

    return gene_df


def map_gene_split_to_promoters(df, gene_col, record_col, split_map):
    out = pd.DataFrame()
    out["gene_id_base"] = normalize_gene_id(df[gene_col])
    out["record_id"] = df[record_col].astype(str)
    out["split"] = out["gene_id_base"].map(split_map)

    if out["split"].isna().any():
        missing_n = int(out["split"].isna().sum())
        raise ValueError(f"Promoter rows with missing split mapping: {missing_n}")

    if out["record_id"].duplicated(keep=False).any():
        dup_n = int(out["record_id"].duplicated(keep=False).sum())
        raise ValueError(f"Split output has duplicated record_id rows: {dup_n}")

    return out


def summarize_distribution(frame, score_cols, level_name):
    rows = []
    split_order = ["train", "val", "test"]
    total_n = len(frame)

    for split in split_order:
        sub = frame.loc[frame["split"] == split].copy()
        n = len(sub)
        rows.append(
            {
                "level": level_name,
                "metric": "row_count",
                "split": split,
                "value": float(n),
                "note": "",
            }
        )
        rows.append(
            {
                "level": level_name,
                "metric": "row_fraction",
                "split": split,
                "value": 0.0 if total_n == 0 else float(n / total_n),
                "note": "",
            }
        )

        for col in score_cols:
            vals = pd.to_numeric(sub[col], errors="coerce")
            rows.append(
                {
                    "level": level_name,
                    "metric": f"{col}_non_missing",
                    "split": split,
                    "value": float(vals.notna().sum()),
                    "note": "",
                }
            )
            rows.append(
                {
                    "level": level_name,
                    "metric": f"{col}_mean",
                    "split": split,
                    "value": float(vals.mean()) if vals.notna().sum() else np.nan,
                    "note": "",
                }
            )
            rows.append(
                {
                    "level": level_name,
                    "metric": f"{col}_median",
                    "split": split,
                    "value": float(vals.median()) if vals.notna().sum() else np.nan,
                    "note": "",
                }
            )

    return pd.DataFrame(rows)


def write_log(log_path, items):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "w", encoding="utf-8") as handle:
        for key, value in items:
            handle.write(f"{key}\t{value}\n")


def main():
    core_path = pick_existing(CORE_CANDIDATES)
    extended_path = pick_existing(EXTENDED_CANDIDATES)

    core_df = pd.read_csv(core_path, sep="\t", low_memory=False)
    extended_df = pd.read_csv(extended_path, sep="\t", low_memory=False)

    assert_no_invalid_score_s_zero_encoding(core_df, "core_input")
    assert_no_invalid_score_s_zero_encoding(extended_df, "extended_input")
    assert_record_id_unique(core_df, "core_input")
    assert_record_id_unique(extended_df, "extended_input")

    core_prepared, core_gene_col, core_record_col, _ = prepare_table(core_df)
    extended_prepared, extended_gene_col, extended_record_col, _ = prepare_table(extended_df)

    gene_df = build_gene_table(core_prepared, extended_prepared)
    gene_df = assign_split(gene_df)

    split_map = gene_df.set_index("gene_id_base")["split"]

    core_split = map_gene_split_to_promoters(
        core_df,
        gene_col=core_gene_col,
        record_col=core_record_col,
        split_map=split_map,
    )

    extended_split = map_gene_split_to_promoters(
        extended_df,
        gene_col=extended_gene_col,
        record_col=extended_record_col,
        split_map=split_map,
    )

    OUTPUT_GENE.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_CORE.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_EXTENDED.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_QC.parent.mkdir(parents=True, exist_ok=True)

    gene_df[[
        "gene_id_base",
        "gene_symbol",
        "Score_E",
        "Score_S",
        "Score_B",
        "split",
    ]].to_csv(OUTPUT_GENE, sep="\t", index=False)

    core_split.to_csv(OUTPUT_CORE, sep="\t", index=False)
    extended_split.to_csv(OUTPUT_EXTENDED, sep="\t", index=False)

    gene_level_from_core = core_prepared.merge(
        core_split[["record_id", "split"]],
        on=["record_id"],
        how="left",
    )
    gene_level_from_extended = extended_prepared.merge(
        extended_split[["record_id", "split"]],
        on=["record_id"],
        how="left",
    )

    qc_gene = summarize_distribution(
        gene_df,
        score_cols=["Score_E", "Score_S", "Score_B"],
        level_name="gene",
    )
    qc_core = summarize_distribution(
        gene_level_from_core,
        score_cols=["Score_E", "Score_S", "Score_B"],
        level_name="core_promoter",
    )
    qc_extended = summarize_distribution(
        gene_level_from_extended,
        score_cols=["Score_E", "Score_S", "Score_B"],
        level_name="extended_promoter",
    )

    qc = pd.concat([qc_gene, qc_core, qc_extended], ignore_index=True)
    qc.to_csv(OUTPUT_QC, sep="\t", index=False)

    log_items = [
        ("core_input", core_path),
        ("extended_input", extended_path),
        ("random_seed", RANDOM_SEED),
        ("split_unit", "gene_id_base"),
        ("promoter_primary_key", "record_id"),
        ("formal_targets", "Score_E,Score_S,Score_B"),
        ("score_s_missing_policy", "keep_missing_not_zero"),
        ("anchor_force_train", ",".join(sorted(ANCHOR_GENES))),
        ("stratification", "s_defined + E_decile + B_quintile + S_quintile_if_defined"),
        ("output_gene_manifest", OUTPUT_GENE),
        ("output_core_split", OUTPUT_CORE),
        ("output_extended_split", OUTPUT_EXTENDED),
        ("output_qc", OUTPUT_QC),
        ("gene_rows", len(gene_df)),
        ("core_rows", len(core_split)),
        ("extended_rows", len(extended_split)),
    ]
    write_log(OUTPUT_LOG, log_items)

    print("Train/eval split build finished.")
    print(f"gene manifest: {OUTPUT_GENE}")
    print(f"core split: {OUTPUT_CORE}")
    print(f"extended split: {OUTPUT_EXTENDED}")
    print(f"qc summary: {OUTPUT_QC}")
    print(f"log file: {OUTPUT_LOG}")


if __name__ == "__main__":
    main()
