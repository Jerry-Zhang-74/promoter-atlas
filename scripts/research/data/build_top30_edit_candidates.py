#!/usr/bin/env python3
"""Round 7: ISM edit candidates for the top30 natural promoters (MPRA model).

For each top30 200 bp promoter, run full single-base ISM with the MPRA-fine-tuned
model and export the top-N increase/decrease edits with locked predictions.

Output:
    design/output/top30_edit_candidates.tsv
"""

import importlib.util
import numpy as np
import pandas as pd
import torch
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CKPT = ROOT / "models/checkpoints/mpra_finetune.pt"
MM200 = ROOT / "promoters/memmap/promoter_up200_uint8.memmap"
MM200_IDX = ROOT / "promoters/memmap/promoter_up200_index.tsv"
TOP30 = ROOT / "design/output/top30_hkg_gene_promoters.tsv"
OUT = ROOT / "design/output/top30_edit_candidates.tsv"
TOP_N = 10
BASE = {0: "A", 1: "C", 2: "G", 3: "T", 4: "N"}


def main():
    spec = importlib.util.spec_from_file_location("ft", ROOT / "runs/scripts/train_mpra_finetune.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    ck = torch.load(CKPT, map_location="cpu", weights_only=False)
    model = mod.ConvBaseline(1, feat_dim=0).to(DEVICE)
    model.load_state_dict(ck["state_dict"])
    model.eval()

    idx = pd.read_csv(MM200_IDX, sep="\t").set_index("record_id")["row_index"]
    mm = np.memmap(MM200, dtype="uint8", mode="r", shape=(len(idx), 200))
    top = pd.read_csv(TOP30, sep="\t")

    rows = []
    for _, r in top.iterrows():
        rec = r["record_id"]
        if rec not in idx.index:
            continue
        ref = np.asarray(mm[idx[rec]], dtype=np.int64).copy()
        refseq = "".join(BASE[c] for c in ref)
        variants, metas = [], []
        for i in range(200):
            for alt in range(4):
                if alt == ref[i]:
                    continue
                v = ref.copy()
                v[i] = alt
                variants.append(v)
                metas.append((i, BASE[ref[i]], BASE[alt]))
        x = np.stack([ref] + variants).astype(np.uint8)
        preds = []
        with torch.no_grad():
            for i in range(0, len(x), 512):
                xb = torch.from_numpy(x[i:i + 512]).to(DEVICE)
                preds.append(model(xb).cpu().numpy()[:, 0])
        p = np.concatenate(preds)
        ref_p = p[0]
        d = p[1:] - ref_p
        order = np.argsort(-d)
        for o in order[:TOP_N]:
            pos, rb, ab = metas[o]
            rows.append({"record_id": rec, "gene_name": r["gene_name"],
                         "pos_in_window": pos, "ref": rb, "alt": ab,
                         "delta_pred_log10": round(float(d[o]), 4),
                         "ref_pred_log10": round(float(ref_p), 4)})
        for o in order[-TOP_N:][::-1]:
            pos, rb, ab = metas[o]
            rows.append({"record_id": rec, "gene_name": r["gene_name"],
                         "pos_in_window": pos, "ref": rb, "alt": ab,
                         "delta_pred_log10": round(float(d[o]), 4),
                         "ref_pred_log10": round(float(ref_p), 4)})
        print("done", r["gene_name"], "ref_pred=%.3f" % ref_p, flush=True)

    res = pd.DataFrame(rows)
    res.to_csv(OUT, sep="\t", index=False)
    print("wrote", OUT, "rows =", len(res))


if __name__ == "__main__":
    main()
