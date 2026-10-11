"""
Run full motif scanning v1 for promoter_atlas.

This version uses chunked NumPy vectorization for speed.

Outputs
-------
- features/motif/promoter_motif_hits_v1_exact_unique.tsv
- features/motif/promoter_motif_agg_v1_exact_unique.tsv
- features/motif/promoter_motif_scan_summary_v1_exact_unique.tsv
"""

from pathlib import Path
import csv
import time
import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view


ROOT = Path(".")

INPUT_TABLE = ROOT / "labels/build/promoter_model_input_core.tsv"
MEMMAP_PATH = ROOT / "promoters/memmap/promoter_up200_uint8.memmap"
MEMMAP_INDEX_PATH = ROOT / "promoters/memmap/promoter_up200_index.tsv"

JASPAR_TXT = ROOT / "data/external/jaspar/JASPAR2026_CORE_vertebrates_non-redundant_pfms_jaspar.txt"
PWM_MANIFEST = ROOT / "features/motif/motif_pwm_manifest_v1.tsv"

OUT_DIR = ROOT / "features/motif"
HITS_OUT = OUT_DIR / "promoter_motif_hits_v1_exact_unique.tsv"
AGG_OUT = OUT_DIR / "promoter_motif_agg_v1_exact_unique.tsv"
SUMMARY_OUT = OUT_DIR / "promoter_motif_scan_summary_v1_exact_unique.tsv"

SEQ_LEN = 200
BASES = ["A", "C", "G", "T"]
COMP_INDEX = np.array([3, 2, 1, 0, 4], dtype=np.uint8)

CHUNK_SIZE = 4096
PROGRESS_EVERY_CHUNKS = 1


def reverse_complement_batch(x):
    return COMP_INDEX[x[:, ::-1]]


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


def prepare_df():
    df = pd.read_csv(INPUT_TABLE, sep="\t", low_memory=False)
    mm_index = pd.read_csv(MEMMAP_INDEX_PATH, sep="\t", low_memory=False)

    required_input = ["record_id", "gene_id_base"]
    required_index = ["record_id", "row_index"]

    miss_input = [c for c in required_input if c not in df.columns]
    miss_index = [c for c in required_index if c not in mm_index.columns]

    if miss_input:
        raise KeyError(f"Missing input columns: {miss_input}")
    if miss_index:
        raise KeyError(f"Missing memmap index columns: {miss_index}")

    df = df.merge(mm_index[["record_id", "row_index"]], on="record_id", how="inner")

    if df["record_id"].duplicated().any():
        dup_n = int(df["record_id"].duplicated().sum())
        raise ValueError(f"Duplicated record_id after merge: {dup_n}")

    return df.sort_values("record_id").reset_index(drop=True)


def load_memmap_and_motifs():
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
    manifest = manifest.sort_values("motif_id").reset_index(drop=True)

    jaspar_map = parse_jaspar_batch(JASPAR_TXT)

    motif_objs = {}
    for row in manifest.itertuples(index=False):
        matrix_id = str(row.jaspar_matrix_id)
        if matrix_id not in jaspar_map:
            raise KeyError(f"Matrix ID not found in JASPAR batch file: {matrix_id}")

        obj = pfm_to_log_pwm(jaspar_map[matrix_id])

        lookup = np.full((5, obj["length"]), -1e9, dtype=np.float64)
        lookup[0:4, :] = obj["pwm"]

        max_score = obj["score_max"]
        min_score = obj["score_min"]
        denom = max_score - min_score if max_score > min_score else np.nan

        motif_objs[row.motif_id] = {
            "matrix_id": matrix_id,
            "length": obj["length"],
            "lookup": lookup,
            "score_max": max_score,
            "score_min": min_score,
            "score_denom": denom,
            "thr_loose": float(row.threshold_loose_value),
            "thr_strict": float(row.threshold_strict_value),
        }

    return mm, manifest, motif_objs


def init_summary(manifest):
    stats = {}
    for motif_id in manifest["motif_id"].tolist():
        stats[motif_id] = {
            "n_promoters": 0,
            "strict_present_n": 0,
            "loose_present_n": 0,
            "sum_count_loose": 0.0,
            "sum_count_strict": 0.0,
            "sum_best_score_norm": 0.0,
        }
    return stats


def motif_windows_scores(batch_idx, motif_obj):
    k = motif_obj["length"]
    lookup = motif_obj["lookup"]
    pos_idx = np.arange(k)

    fw = sliding_window_view(batch_idx, window_shape=k, axis=1)
    rv = sliding_window_view(reverse_complement_batch(batch_idx), window_shape=k, axis=1)

    fw_scores = lookup[fw, pos_idx].sum(axis=2)
    rv_scores = lookup[rv, pos_idx].sum(axis=2)

    return fw_scores, rv_scores


def build_zone_masks(k):
    n_win = SEQ_LEN - k + 1
    starts = np.arange(n_win)
    rel = starts - SEQ_LEN

    distal = ((rel >= -200) & (rel <= -101)).astype(np.int32)
    mid = ((rel >= -100) & (rel <= -41)).astype(np.int32)
    proximal = ((rel >= -40) & (rel <= -1)).astype(np.int32)

    return {
        "distal": distal,
        "mid": mid,
        "proximal": proximal,
    }


def covered_bp_from_hits(hit_mask, k):
    bsz, n_win = hit_mask.shape
    cover = np.zeros((bsz, SEQ_LEN), dtype=bool)
    for offset in range(k):
        cover[:, offset:offset + n_win] |= hit_mask
    return cover.sum(axis=1).astype(np.int32)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print("full motif scan v1 started")
    print("=" * 72)

    df = prepare_df()
    mm, manifest, motif_objs = load_memmap_and_motifs()
    summary = init_summary(manifest)

    print("promoter count:", len(df))
    print("motif count:", len(manifest))
    print("sequence length:", SEQ_LEN)
    print("chunk size:", CHUNK_SIZE)
    print("=" * 72)

    hit_columns = [
        "record_id", "gene_id_base", "motif_id", "matrix_id",
        "count_loose", "count_strict",
        "best_score_raw", "best_score_norm",
        "best_start", "best_end", "best_strand",
        "covered_bp_loose", "covered_bp_strict",
        "distal_count_loose", "mid_count_loose", "proximal_count_loose",
        "distal_count_strict", "mid_count_strict", "proximal_count_strict",
    ]

    agg_columns = ["record_id", "gene_id_base"]
    for motif_id in manifest["motif_id"].tolist():
        agg_columns.extend([
            f"{motif_id}_count_loose",
            f"{motif_id}_count_strict",
            f"{motif_id}_presence_strict",
            f"{motif_id}_density_loose",
            f"{motif_id}_best_score_norm",
            f"{motif_id}_best_pos_rel_tss",
            f"{motif_id}_distal_count_loose",
            f"{motif_id}_mid_count_loose",
            f"{motif_id}_proximal_count_loose",
            f"{motif_id}_distal_count_strict",
            f"{motif_id}_mid_count_strict",
            f"{motif_id}_proximal_count_strict",
        ])

    total_n = len(df)
    n_chunks = (total_n + CHUNK_SIZE - 1) // CHUNK_SIZE
    start_time = time.time()

    with open(HITS_OUT, "w", encoding="utf-8", newline="") as hits_f, \
         open(AGG_OUT, "w", encoding="utf-8", newline="") as agg_f:

        hits_writer = csv.DictWriter(hits_f, fieldnames=hit_columns, delimiter="\t")
        agg_writer = csv.DictWriter(agg_f, fieldnames=agg_columns, delimiter="\t")

        hits_writer.writeheader()
        agg_writer.writeheader()

        for chunk_idx in range(n_chunks):
            left = chunk_idx * CHUNK_SIZE
            right = min((chunk_idx + 1) * CHUNK_SIZE, total_n)

            chunk_df = df.iloc[left:right].copy().reset_index(drop=True)
            row_idx = chunk_df["row_index"].to_numpy(dtype=np.int64)
            batch_idx = np.asarray(mm[row_idx], dtype=np.uint8)

            agg_rows = [
                {
                    "record_id": rid,
                    "gene_id_base": gid,
                }
                for rid, gid in zip(
                    chunk_df["record_id"].tolist(),
                    chunk_df["gene_id_base"].tolist(),
                )
            ]

            record_ids = chunk_df["record_id"].tolist()
            gene_ids = chunk_df["gene_id_base"].tolist()

            for motif_row in manifest.itertuples(index=False):
                motif_id = motif_row.motif_id
                obj = motif_objs[motif_id]
                k = obj["length"]

                fw_scores, rv_scores = motif_windows_scores(batch_idx, obj)
                zone_masks = build_zone_masks(k)

                denom = obj["score_denom"]
                fw_norm = (fw_scores - obj["score_min"]) / denom
                rv_norm = (rv_scores - obj["score_min"]) / denom

                loose_fw = fw_norm >= obj["thr_loose"]
                loose_rv = rv_norm >= obj["thr_loose"]
                strict_fw = fw_norm >= obj["thr_strict"]
                strict_rv = rv_norm >= obj["thr_strict"]

                loose_any = loose_fw | loose_rv
                strict_any = strict_fw | strict_rv

                count_loose = loose_fw.sum(axis=1) + loose_rv.sum(axis=1)
                count_strict = strict_fw.sum(axis=1) + strict_rv.sum(axis=1)

                best_fw_idx = fw_scores.argmax(axis=1)
                best_rv_idx = rv_scores.argmax(axis=1)

                best_fw_score = fw_scores[np.arange(len(chunk_df)), best_fw_idx]
                best_rv_score = rv_scores[np.arange(len(chunk_df)), best_rv_idx]

                choose_fw = best_fw_score >= best_rv_score
                best_score_raw = np.where(choose_fw, best_fw_score, best_rv_score)
                best_start = np.where(choose_fw, best_fw_idx, best_rv_idx)
                best_strand = np.where(choose_fw, "+", "-")
                best_score_norm = (best_score_raw - obj["score_min"]) / denom
                best_end = best_start + k

                covered_bp_loose = covered_bp_from_hits(loose_any, k)
                covered_bp_strict = covered_bp_from_hits(strict_any, k)

                distal_count_loose = (loose_any * zone_masks["distal"]).sum(axis=1)
                mid_count_loose = (loose_any * zone_masks["mid"]).sum(axis=1)
                proximal_count_loose = (loose_any * zone_masks["proximal"]).sum(axis=1)

                distal_count_strict = (strict_any * zone_masks["distal"]).sum(axis=1)
                mid_count_strict = (strict_any * zone_masks["mid"]).sum(axis=1)
                proximal_count_strict = (strict_any * zone_masks["proximal"]).sum(axis=1)

                for i in range(len(chunk_df)):
                    hits_writer.writerow(
                        {
                            "record_id": record_ids[i],
                            "gene_id_base": gene_ids[i],
                            "motif_id": motif_id,
                            "matrix_id": obj["matrix_id"],
                            "count_loose": int(count_loose[i]),
                            "count_strict": int(count_strict[i]),
                            "best_score_raw": float(best_score_raw[i]),
                            "best_score_norm": float(best_score_norm[i]),
                            "best_start": int(best_start[i]),
                            "best_end": int(best_end[i]),
                            "best_strand": best_strand[i],
                            "covered_bp_loose": int(covered_bp_loose[i]),
                            "covered_bp_strict": int(covered_bp_strict[i]),
                            "distal_count_loose": int(distal_count_loose[i]),
                            "mid_count_loose": int(mid_count_loose[i]),
                            "proximal_count_loose": int(proximal_count_loose[i]),
                            "distal_count_strict": int(distal_count_strict[i]),
                            "mid_count_strict": int(mid_count_strict[i]),
                            "proximal_count_strict": int(proximal_count_strict[i]),
                        }
                    )

                    agg_rows[i][f"{motif_id}_count_loose"] = int(count_loose[i])
                    agg_rows[i][f"{motif_id}_count_strict"] = int(count_strict[i])
                    agg_rows[i][f"{motif_id}_presence_strict"] = int(count_strict[i] > 0)
                    agg_rows[i][f"{motif_id}_density_loose"] = float(count_loose[i] / SEQ_LEN)
                    agg_rows[i][f"{motif_id}_best_score_norm"] = float(best_score_norm[i])
                    agg_rows[i][f"{motif_id}_best_pos_rel_tss"] = int(best_start[i] - SEQ_LEN)
                    agg_rows[i][f"{motif_id}_distal_count_loose"] = int(distal_count_loose[i])
                    agg_rows[i][f"{motif_id}_mid_count_loose"] = int(mid_count_loose[i])
                    agg_rows[i][f"{motif_id}_proximal_count_loose"] = int(proximal_count_loose[i])
                    agg_rows[i][f"{motif_id}_distal_count_strict"] = int(distal_count_strict[i])
                    agg_rows[i][f"{motif_id}_mid_count_strict"] = int(mid_count_strict[i])
                    agg_rows[i][f"{motif_id}_proximal_count_strict"] = int(proximal_count_strict[i])

                s = summary[motif_id]
                s["n_promoters"] += len(chunk_df)
                s["strict_present_n"] += int((count_strict > 0).sum())
                s["loose_present_n"] += int((count_loose > 0).sum())
                s["sum_count_loose"] += float(count_loose.sum())
                s["sum_count_strict"] += float(count_strict.sum())
                s["sum_best_score_norm"] += float(best_score_norm.sum())

            for row in agg_rows:
                agg_writer.writerow(row)

            if ((chunk_idx + 1) % PROGRESS_EVERY_CHUNKS == 0) or (chunk_idx + 1 == n_chunks):
                done = right
                elapsed = time.time() - start_time
                rate = done / elapsed if elapsed > 0 else 0.0
                remain = total_n - done
                eta = remain / rate if rate > 0 else 0.0
                print(
                    f"[scan] chunk {chunk_idx + 1}/{n_chunks} | "
                    f"{done}/{total_n} promoters | "
                    f"{rate:.1f} promoters/s | "
                    f"elapsed {elapsed/60:.1f} min | "
                    f"ETA {eta/60:.1f} min"
                )

    summary_rows = []
    for motif_id in sorted(summary.keys()):
        s = summary[motif_id]
        n = max(int(s["n_promoters"]), 1)
        summary_rows.append(
            {
                "motif_id": motif_id,
                "n_promoters": n,
                "strict_presence_rate": s["strict_present_n"] / n,
                "loose_presence_rate": s["loose_present_n"] / n,
                "mean_count_loose": s["sum_count_loose"] / n,
                "mean_count_strict": s["sum_count_strict"] / n,
                "mean_best_score_norm": s["sum_best_score_norm"] / n,
            }
        )

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(SUMMARY_OUT, sep="\t", index=False)

    total_elapsed = time.time() - start_time
    print("=" * 72)
    print("full motif scan finished")
    print("promoter_count:", len(df))
    print("motif_count:", len(manifest))
    print(f"total_time_min: {total_elapsed/60:.2f}")
    print(f"hits_out: {HITS_OUT}")
    print(f"agg_out: {AGG_OUT}")
    print(f"summary_out: {SUMMARY_OUT}")
    print("=" * 72)


if __name__ == "__main__":
    main()
