#!/usr/bin/env python3
"""Round 7: graded-strength designed promoter library v1 (first designed sequences).

Templates: top10 natural HKG 200 bp cores + top10 strong synthetic MPRA test cores.
For each template: rank all single-base edits by predicted activity delta with the
MPRA-fine-tuned model; keep the template + top-2 up + top-2 down edits.
Tiers are assigned by predicted activity (absolute, weighted-avg equivalent).

Output: design/output/graded_library_v1.tsv (locked predictions)
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
SUP = ROOT / "features/mpra/mpra_supervised.tsv"
OUT = ROOT / "design/output/graded_library_v1.tsv"
BASE = {0: "A", 1: "C", 2: "G", 3: "T", 4: "N"}


def main():
    spec = importlib.util.spec_from_file_location("ft", ROOT / "runs/scripts/train_mpra_finetune.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    ck = torch.load(CKPT, map_location="cpu", weights_only=False)
    model = mod.ConvBaseline(1, feat_dim=0).to(DEVICE)
    model.load_state_dict(ck["state_dict"])
    model.eval()

    def pred_batch(seqs):
        x = np.stack([mod.encode([s]) if False else
                      np.array([{"A": 0, "C": 1, "G": 2, "T": 3, "N": 4}.get(c, 4) for c in s.upper()],
                               dtype=np.uint8)
                      for s in seqs]).astype(np.uint8)
        out = []
        with torch.no_grad():
            for i in range(0, len(x), 512):
                xb = torch.from_numpy(x[i:i + 512]).to(DEVICE)
                out.append(model(xb).cpu().numpy()[:, 0])
        return np.concatenate(out)

    templates = []
    # 10 natural HKG cores (200 bp)
    idx = pd.read_csv(MM200_IDX, sep="\t").set_index("record_id")["row_index"]
    mm = np.memmap(MM200, dtype="uint8", mode="r", shape=(len(idx), 200))
    top = pd.read_csv(TOP30, sep="\t").head(10)
    for _, r in top.iterrows():
        ref = np.asarray(mm[idx[r["record_id"]]], dtype=np.int64)
        templates.append(("nat_" + r["gene_name"], "".join(BASE[c] for c in ref)))
    # 10 strong synthetic cores (164 bp, test split)
    sup = pd.read_csv(SUP, sep="\t")
    syn = sup[(sup["split"] == "test") & (sup["activity"] > 1.0)].head(10)
    for i, (_, r) in enumerate(syn.iterrows()):
        templates.append(("syn_%02d" % i, r["sequence"]))
    # 5 low-activity + 5 very-high-activity synthetic cores to fill the gradient ends
    low = sup[(sup["split"] == "test") & (sup["activity"] < 0.2)].head(5)
    for i, (_, r) in enumerate(low.iterrows()):
        templates.append(("low_%02d" % i, r["sequence"]))
    hi = sup[(sup["split"] == "test") & (sup["activity"] > 1.3)].head(5)
    for i, (_, r) in enumerate(hi.iterrows()):
        templates.append(("hi_%02d" % i, r["sequence"]))

    rows = []
    for name, seq in templates:
        L = len(seq)
        ref_p = float(pred_batch([seq])[0])
        variants, metas = [], []
        for i in range(L):
            for alt in "ACGT":
                if alt == seq[i]:
                    continue
                variants.append(seq[:i] + alt + seq[i + 1:])
                metas.append((i, seq[i], alt))
        p = pred_batch(variants)
        d = p - ref_p
        order = np.argsort(-d)
        rows.append({"design_id": name + "_ref", "template": name, "n_edits": 0,
                     "pred_log10": round(ref_p, 4)})
        for o in order[:2]:
            pos, rb, ab = metas[o]
            rows.append({"design_id": "%s_up%d" % (name, o + 1), "template": name, "n_edits": 1,
                         "edit": "%s%d%s>%s" % (rb, pos, rb, ab),
                         "pred_log10": round(float(p[o]), 4)})
        for o in order[-2:][::-1]:
            pos, rb, ab = metas[o]
            rows.append({"design_id": "%s_dn%d" % (name, len(order) - o), "template": name, "n_edits": 1,
                         "edit": "%s%d%s>%s" % (rb, pos, rb, ab),
                         "pred_log10": round(float(p[o]), 4)})
        print("done", name, "ref=%.3f" % ref_p, flush=True)

    res = pd.DataFrame(rows)
    res["pred_weighted_avg"] = (10 ** res["pred_log10"]).round(2)
    # tiers by predicted weighted-avg activity
    bins = [-np.inf, 2, 4, 8, 16, 32, np.inf]
    labels = ["T1_<2x", "T2_2-4x", "T3_4-8x", "T4_8-16x", "T5_16-32x", "T6_>32x"]
    res["tier"] = pd.cut(res["pred_weighted_avg"], bins=bins, labels=labels)
    res.to_csv(OUT, sep="\t", index=False)
    print("\ntier counts:")
    print(res["tier"].value_counts().to_string())
    print("wrote", OUT, "rows =", len(res))


if __name__ == "__main__":
    main()
