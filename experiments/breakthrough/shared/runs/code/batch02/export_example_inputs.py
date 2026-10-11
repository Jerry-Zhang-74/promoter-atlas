#!/usr/bin/env python3
"""Export example inference inputs for the loadable gene-level Score_E predictor.

Produces an .npz (gene embeddings and sequence features) plus a JSON manifest with
the chosen gene ids, their splits, and the expected scores recorded from the saved
per-gene predictions. No labels are required to run the predictor; expected scores
are included only so a user can verify a clean re-implementation.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from b2_common import sha256_file, atomic_write_json  # noqa: E402

CACHE = ROOT / "data" / "breakthrough_20261002"
OUT = ROOT / "data" / "breakthrough_20261002" / "batch02"
REPORT = ROOT / "reports" / "breakthrough_20261002" / "batch02"
MODEL_DIR = ROOT / "models" / "breakthrough_20261002" / "batch02"


def load_gene_arrays(gene_ids, rows, emb, seq):
    """Return dict gene_id -> (k,512) embeddings and (k,44) seqfeats for the requested genes."""
    emb_out, seq_out = {}, {}
    for g in gene_ids:
        m = rows["gene_id"].values == g
        idx = rows["idx"].values[m].astype(np.int64)
        emb_out[g] = np.asarray(emb[idx], dtype=np.float32)
        seq_out[g] = np.asarray(seq[idx], dtype=np.float32)
    return emb_out, seq_out


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows = pd.read_csv(CACHE / "trackA_rows.tsv", sep="\t")
    emb = np.load(CACHE / "trackA_embeddings.f32", mmap_mode="r")
    seq = np.load(CACHE / "trackA_seqfeats.f32", mmap_mode="r")

    # deterministic example selection: one per split, varying promoter count
    counts = rows.groupby("gene_id").size()
    split_of = rows.groupby("gene_id")["split"].first()
    example = {}
    # val: gene with the most promoters
    val_genes = counts[split_of == "val"].sort_values(ascending=False)
    example["val"] = val_genes.index[0]
    # test: first single-promoter gene
    test_single = counts[(split_of == "test") & (counts == 1)]
    example["test"] = test_single.index[0]
    # train: first two-promoter gene
    train_two = counts[(split_of == "train") & (counts == 2)]
    example["train"] = train_two.index[0]

    emb_out, seq_out = load_gene_arrays([example[k] for k in ["train", "val", "test"]], rows, emb, seq)

    npz_path = OUT / "trackA_example_inputs.npz"
    save = {}
    for k in ["train", "val", "test"]:
        g = example[k]
        save[f"emb_{k}"] = emb_out[g]
        save[f"seq_{k}"] = seq_out[g]
    np.savez_compressed(npz_path, **save)

    # expected scores from the saved per-gene predictions (where present)
    expected = {}
    for split_name, pred_file in [("val", REPORT / "trackA_val_predictions.tsv"),
                                  ("test", REPORT / "trackA_test_predictions.tsv")]:
        if pred_file.exists():
            p = pd.read_csv(pred_file, sep="\t")
            g = example[split_name]
            row = p[p["gene_id"] == g]
            if len(row):
                expected[split_name] = {"gene_id": g, "selected_E_pred": float(row["selected_E_pred"].iloc[0]),
                                        "Score_E": float(row["Score_E"].iloc[0])}

    manifest = {
        "written_at": "example_inputs",
        "model_dir": str(MODEL_DIR),
        "example_genes": {k: {"gene_id": example[k], "split": k, "n_promoters": int(counts[example[k]])}
                          for k in ["train", "val", "test"]},
        "expected_scores_from_saved_predictions": expected,
        "npz_sha256": sha256_file(npz_path),
        "note": "expected scores are only for verification and are not read by the predictor",
    }
    atomic_write_json(OUT / "trackA_example_inputs.json", manifest)
    print(json.dumps(manifest, indent=2, ensure_ascii=True), flush=True)


if __name__ == "__main__":
    main()
