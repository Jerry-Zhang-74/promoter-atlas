"""
Run pilot motif scanning v1 for promoter_atlas.

This script performs a small-scale motif scan pilot on the core up200 promoter set.

Scope
-----
- use only motifs with match_status == exact_unique
- use core promoter memmap (up200)
- sample a small split-stratified subset
- scan both strands
- output pilot hits / agg / summary tables

Outputs
-------
- features/motif/pilot/promoter_motif_hits_pilot_v1.tsv
- features/motif/pilot/promoter_motif_agg_pilot_v1.tsv
- features/motif/pilot/promoter_motif_scan_summary_v1.tsv
"""

from pathlib import Path
import math
import numpy as np
import pandas as pd


ROOT = Path(".")

INPUT_TABLE = ROOT / "labels/build/promoter_model_input_core.tsv"
SPLIT_TABLE = ROOT / "labels/releases/promoter_split_core.tsv"

MEMMAP_PATH = ROOT / "promoters/memmap/promoter_up200_uint8.memmap"
MEMMAP_INDEX_PATH = ROOT / "promoters/memmap/promoter_up200_index.tsv"

JASPAR_TXT = ROOT / "data/external/jaspar/JASPAR2026_CORE_vertebrates_non-redundant_pfms_jaspar.txt"
PWM_MANIFEST = ROOT / "features/motif/motif_pwm_manifest_v1.tsv"

OUT_DIR = ROOT / "features/motif/pilot"
HITS_OUT = OUT_DIR / "promoter_motif_hits_pilot_v1.tsv"
AGG_OUT = OUT_DIR / "promoter_motif_agg_pilot_v1.tsv"
SUMMARY_OUT = OUT_DIR / "promoter_motif_scan_summary_v1.tsv"

SEED = 20260315
SEQ_LEN = 200
TRAIN_N = 1000
VAL_N = 200
TEST_N = 200

BASES = ["A", "C", "G", "T"]
BASE_TO_ROW = {"A": 0, "C": 1, "G": 2, "T": 3}
COMP_INDEX = np.array([3, 2, 1, 0, 4], dtype=np.uint8)


def reverse_complement_index(seq_idx):
    return COMP_INDEX[seq_idx[::-1]]


def parse_jaspar_batch(path):
    motif_map = {}
    header = None
    lines = []

    def finalize(h, block_lines):
        if h is None:
            return None, None
        parts = h.split(maxsplit=1)
        matrix_id = parts[0].strip()

        rows = {}
        for line in block_lines:
            line = line.strip()
            if not line:
                continue
            base = line[0]
            if base not in {"A", "C", "G", "T"}:
                continue
            left = line.find("[")
            right = line.find("]")
            nums = line[left + 1:right].strip().split()
            rows[base] = np.array([float(x) for x in nums], dtype=np.float64)
        if len(rows) != 4:
            raise ValueError(f"Incomplete PWM rows for {matrix_id}")
        return matrix_id, rows

    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.rstrip("\n")
            if not line:
                continue
            if line.startswith(">"):
                matrix_id, rows = finalize(header, lines)
                if matrix_id is not None:
                    motif_map[matrix_id] = rows
                header = line[1:].strip()
                lines = []
            else:
                lines.append(line)

    matrix_id, rows = finalize(header, lines)
    if matrix_id is not None:
        motif_map[matrix_id] = rows

    return motif_map


def pfm_to_log_pwm(pfm_rows, pseudocount=0.5, bg=0.25):
    mat = np.vstack([pfm_rows[b] for b in BASES]).astype(np.float64)
    col_sums = mat.sum(axis=0, keepdims=True)
    ppm = (mat + pseudocount) / (col_sums + 4 * pseudocount)
    pwm = np.log2(ppm / bg)

    col_max = pwm.max(axis=0)
    col_min = pwm.min(axis=0)

    return {
        "pwm": pwm,
        "length": pwm.shape[1],
        "score_max": float(col_max.sum()),
        "score_min": float(col_min.sum()),
    }


def scan_one_sequence(seq_idx, motif_obj, thr_loose, thr_strict):
    pwm = motif_obj["pwm"]
    k = motif_obj["length"]
    score_max = motif_obj["score_max"]
    score_min = motif_obj["score_min"]

    if len(seq_idx) < k:
        return {
            "count_loose": 0,
            "count_strict": 0,
            "best_score_raw": np.nan,
            "best_score_norm": np.nan,
            "best_start": np.nan,
            "best_end": np.nan,
            "best_strand": "",
            "covered_bp_loose": 0,
            "covered_bp_strict": 0,
            "distal_count_loose": 0,
            "mid_count_loose": 0,
            "proximal_count_loose": 0,
            "distal_count_strict": 0,
            "mid_count_strict": 0,
            "proximal_count_strict": 0,
        }

    loose_intervals = []
    strict_intervals = []

    best_score_raw = None
    best_score_norm = None
    best_start = None
    best_strand = None

    for strand_name, arr in [("+", seq_idx), ("-", reverse_complement_index(seq_idx))]:
        n_windows = len(arr) - k + 1
        for i in range(n_windows):
            window = arr[i:i+k]
            if np.any(window > 3):
                continue
            score = 0.0
            for pos in range(k):
                score += pwm[int(window[pos]), pos]
            score_norm = (score - score_min) / (score_max - score_min) if score_max > score_min else np.nan

            if (best_score_raw is None) or (score > best_score_raw):
                best_score_raw = float(score)
                best_score_norm = float(score_norm)
                best_start = int(i)
                best_strand = strand_name

            if score_norm >= thr_loose:
                loose_intervals.append((i, i + k, strand_name))
            if score_norm >= thr_strict:
                strict_intervals.append((i, i + k, strand_name))

    def covered_bp(intervals):
        if not intervals:
            return 0
        merged = []
        for s, e, _ in sorted(intervals):
            if not merged or s > merged[-1][1]:
                merged.append([s, e])
            else:
                merged[-1][1] = max(merged[-1][1], e)
        return int(sum(e - s for s, e in merged))

    def zone_name(start):
        rel = start - SEQ_LEN
        if -200 <= rel <= -101:
            return "distal"
        if -100 <= rel <= -41:
            return "mid"
        if -40 <= rel <= -1:
            return "proximal"
        return "other"

    zone_keys = [
        "distal_count_loose", "mid_count_loose", "proximal_count_loose",
        "distal_count_strict", "mid_count_strict", "proximal_count_strict",
    ]
    zone_counts = {k: 0 for k in zone_keys}

    for s, e, strand in loose_intervals:
        z = zone_name(s)
        if z in {"distal", "mid", "proximal"}:
            zone_counts[f"{z}_count_loose"] += 1

    for s, e, strand in strict_intervals:
        z = zone_name(s)
        if z in {"distal", "mid", "proximal"}:
            zone_counts[f"{z}_count_strict"] += 1

    return {
        "count_loose": int(len(loose_intervals)),
        "count_strict": int(len(strict_intervals)),
        "best_score_raw": best_score_raw,
        "best_score_norm": best_score_norm,
        "best_start": best_start,
        "best_end": (best_start + k) if best_start is not None else np.nan,
        "best_strand": best_strand if best_strand is not None else "",
        "covered_bp_loose": covered_bp(loose_intervals),
        "covered_bp_strict": covered_bp(strict_intervals),
        **zone_counts,
    }


def prepare_pilot_df():
    df = pd.read_csv(INPUT_TABLE, sep="\t", low_memory=False)
    split = pd.read_csv(SPLIT_TABLE, sep="\t", low_memory=False)
    mm_index = pd.read_csv(MEMMAP_INDEX_PATH, sep="\t", low_memory=False)

    df = df.merge(split[["record_id", "split"]], on="record_id", how="inner")
    df = df.merge(mm_index[["record_id", "row_index"]], on="record_id", how="inner")

    rng = np.random.default_rng(SEED)

    parts = []
    for split_name, n_keep in [("train", TRAIN_N), ("val", VAL_N), ("test", TEST_N)]:
        sub = df.loc[df["split"] == split_name].copy()
        if len(sub) == 0:
            continue
        if len(sub) > n_keep:
            idx = rng.choice(sub.index.to_numpy(), size=n_keep, replace=False)
            sub = sub.loc[idx].copy()
        parts.append(sub)

    pilot_df = pd.concat(parts, ignore_index=True)
    pilot_df = pilot_df.sort_values(["split", "record_id"]).reset_index(drop=True)
    return pilot_df


def load_sequences_and_motifs():
    mm_index = pd.read_csv(MEMMAP_INDEX_PATH, sep="\t", low_memory=False)
    n_rows = len(mm_index)

    mm = np.memmap(
        MEMMAP_PATH,
        dtype="uint8",
        mode="r",
        shape=(n_rows, SEQ_LEN),
    )

    manifest = pd.read_csv(PWM_MANIFEST, sep="\t", low_memory=False)
    manifest = manifest.loc[manifest["match_status"] == "exact_unique"].copy()

    jaspar_map = parse_jaspar_batch(JASPAR_TXT)

    motif_objs = {}
    for row in manifest.itertuples(index=False):
        matrix_id = str(row.jaspar_matrix_id)
        if matrix_id not in jaspar_map:
            raise KeyError(f"Matrix ID not found in JASPAR batch file: {matrix_id}")
        motif_objs[row.motif_id] = {
            **pfm_to_log_pwm(jaspar_map[matrix_id]),
            "matrix_id": matrix_id,
            "thr_loose": float(row.threshold_loose_value),
            "thr_strict": float(row.threshold_strict_value),
        }

    return mm, manifest, motif_objs


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    pilot_df = prepare_pilot_df()
    mm, manifest, motif_objs = load_sequences_and_motifs()

    hit_rows = []
    agg_rows = []

    for p in pilot_df.itertuples(index=False):
        seq_idx = np.asarray(mm[int(p.row_index)], dtype=np.uint8)

        agg_row = {
            "record_id": p.record_id,
            "gene_id_base": p.gene_id_base,
            "split": p.split,
        }

        for motif_row in manifest.itertuples(index=False):
            motif_id = motif_row.motif_id
            obj = motif_objs[motif_id]

            res = scan_one_sequence(
                seq_idx=seq_idx,
                motif_obj=obj,
                thr_loose=obj["thr_loose"],
                thr_strict=obj["thr_strict"],
            )

            hit_rows.append(
                {
                    "record_id": p.record_id,
                    "gene_id_base": p.gene_id_base,
                    "split": p.split,
                    "motif_id": motif_id,
                    "matrix_id": obj["matrix_id"],
                    **res,
                }
            )

            agg_row[f"{motif_id}_count_loose"] = res["count_loose"]
            agg_row[f"{motif_id}_count_strict"] = res["count_strict"]
            agg_row[f"{motif_id}_presence_strict"] = int(res["count_strict"] > 0)
            agg_row[f"{motif_id}_density_loose"] = res["count_loose"] / SEQ_LEN
            agg_row[f"{motif_id}_best_score_norm"] = res["best_score_norm"]
            agg_row[f"{motif_id}_best_pos_rel_tss"] = (
                (res["best_start"] - SEQ_LEN) if pd.notna(res["best_start"]) else np.nan
            )
            agg_row[f"{motif_id}_distal_count_loose"] = res["distal_count_loose"]
            agg_row[f"{motif_id}_mid_count_loose"] = res["mid_count_loose"]
            agg_row[f"{motif_id}_proximal_count_loose"] = res["proximal_count_loose"]
            agg_row[f"{motif_id}_distal_count_strict"] = res["distal_count_strict"]
            agg_row[f"{motif_id}_mid_count_strict"] = res["mid_count_strict"]
            agg_row[f"{motif_id}_proximal_count_strict"] = res["proximal_count_strict"]

        agg_rows.append(agg_row)

    hits_df = pd.DataFrame(hit_rows)
    agg_df = pd.DataFrame(agg_rows)

    summary_rows = []
    for motif_id, sub in hits_df.groupby("motif_id"):
        summary_rows.append(
            {
                "motif_id": motif_id,
                "n_promoters": int(sub["record_id"].nunique()),
                "strict_presence_rate": float((sub["count_strict"] > 0).mean()),
                "loose_presence_rate": float((sub["count_loose"] > 0).mean()),
                "mean_count_loose": float(sub["count_loose"].mean()),
                "mean_count_strict": float(sub["count_strict"].mean()),
                "mean_best_score_norm": float(sub["best_score_norm"].mean()),
            }
        )
    summary_df = pd.DataFrame(summary_rows).sort_values("motif_id").reset_index(drop=True)

    hits_df.to_csv(HITS_OUT, sep="\t", index=False)
    agg_df.to_csv(AGG_OUT, sep="\t", index=False)
    summary_df.to_csv(SUMMARY_OUT, sep="\t", index=False)

    print("pilot motif scan finished")
    print(f"pilot_rows: {len(pilot_df)}")
    print(f"motif_count: {len(manifest)}")
    print(f"hits_out: {HITS_OUT}")
    print(f"agg_out: {AGG_OUT}")
    print(f"summary_out: {SUMMARY_OUT}")


if __name__ == "__main__":
    main()
