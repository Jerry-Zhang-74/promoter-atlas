# -*- coding: utf-8 -*-
"""BATCH29 target freeze: fixed per-fragment proxy y_i = mean_r log2(1 + a_ir),
a_ir = (r_ir / N_Rr) / (D_i / N_D),  D_i = 1 + P_i.  Library offsets over the FULL table.
Builds rows table + variable-length one-hot codes.  NO estimator, NO model."""
import json, numpy as np, hashlib

B = "reports/breakthrough_20261002/batch29"
DB = "data/breakthrough_20261002/batch29"
SEL = DB + "/selected_rows.json"
FAA = DB + "/sure29_assay_oriented.fa"
MAN = B + "/split_manifest.tsv"
PASSA = B + "/passA.json"

passA = json.load(open(PASSA, encoding="utf-8"))
N_D = passA["library_totals_fulltable"]["N_D_1plusP"]
N_R1 = passA["library_totals_fulltable"]["N_R1"]
N_R2 = passA["library_totals_fulltable"]["N_R2"]
sel = {r["key"]: r for r in json.load(open(SEL, encoding="utf-8"))["rows"]}

def load_fa(p):
    d = {}; k = None
    for line in open(p, encoding="utf-8"):
        line = line.strip()
        if not line: continue
        if line.startswith(">"):
            parts = line[1:].split("_"); k = "%s:%s:%s" % (parts[0], parts[1], parts[2])
        else: d[k] = line
    return d
assay = load_fa(FAA)

man = []
with open(MAN, encoding="utf-8") as f:
    next(f)
    for line in f:
        key, ch, st, en, sd, g, b = line.rstrip("\n").split("\t")
        man.append((key, ch, int(st), int(en), sd, g, b))

BASE = {c: i for i, c in enumerate("ACGT")}
maxlen = 0
rows = []
codes_list = []
for key, ch, st, en, sd, g, b in man:
    r = sel[key]; s = assay[key]
    r1 = r["r1"]; r2 = r["r2"]; P = r["P"]; ip = r["ipcr"]
    D = 1 + P
    a1 = (r1 / N_R1) / (D / N_D)
    a2 = (r2 / N_R2) / (D / N_D)
    y = 0.5 * (np.log2(1 + a1) + np.log2(1 + a2))
    rows.append({"key": key, "chrom": ch, "start": st, "end": en, "strand": sd, "group": g,
                 "split": b, "r1": r1, "r2": r2, "P": P, "ipcr": ip, "D": D,
                 "len": en - st, "a1": a1, "a2": a2, "y": y,
                 "det1": 1 if r1 > 0 else 0, "det2": 1 if r2 > 0 else 0,
                 "both": 1 if (r1 > 0 and r2 > 0) else 0})
    maxlen = max(maxlen, len(s))
    codes_list.append(s)

codes = np.full((len(rows), maxlen), 4, dtype=np.int8)
lens = np.zeros(len(rows), dtype=np.int32)
bad = []
for i, s in enumerate(codes_list):
    lens[i] = len(s)
    for j, c in enumerate(s):
        v = BASE.get(c, 4)
        if v == 4: bad.append(i)
        codes[i, j] = v
tmp = DB + "/encoded_tmp.npz"
np.savez_compressed(tmp, codes=codes, lens=lens)
import os; os.replace(tmp, DB + "/encoded.npz")

with open(DB + "/rows.tsv", "w", encoding="utf-8", newline="\n") as f:
    cols = ["key", "chrom", "start", "end", "strand", "group", "split", "r1", "r2", "P", "ipcr",
            "D", "len", "a1", "a2", "y", "det1", "det2", "both"]
    f.write("\t".join(cols) + "\n")
    for r in rows:
        f.write("\t".join(str(r[c]) for c in cols) + "\n")

sp = np.array([r["split"] for r in rows], dtype=object)
y = np.array([r["y"] for r in rows])
summ = {}
for s in ("train", "val", "test"):
    m = sp == s
    summ[s] = {"n": int(m.sum()), "y_mean": float(y[m].mean()), "y_std": float(y[m].std()),
               "y_zero_frac": float(np.mean(y[m] == 0)), "n_groups": len(set(r["group"] for r, mm in zip(rows, m) if mm))}
res = {"batch": "batch29", "target": "y_i = mean_r log2(1 + a_ir), a_ir=(r_ir/N_Rr)/(D_i/N_D), D_i=1+P_i",
       "library_offsets_fulltable": {"N_D": N_D, "N_R1": N_R1, "N_R2": N_R2},
       "n_rows": len(rows), "maxlen": maxlen, "n_unknown_base_rows": len(bad),
       "summary_by_split": summ,
       "train_target_variance_positive": bool(y[sp == "train"].std() > 0)}
json.dump(res, open(B + "/target_freeze.json", "w", encoding="utf-8"), indent=2)
print(json.dumps(res, indent=2))
