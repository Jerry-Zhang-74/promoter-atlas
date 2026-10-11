# -*- coding: utf-8 -*-
"""batch29 independent reviewer: replay E1/E2 metrics from SAVED predictions only.
ZERO fits. Independent metric implementation (does not import worker metrics)."""
import json, csv, hashlib
from pathlib import Path
import numpy as np

ROOT = Path("E:/promoter_atlas")
B = ROOT / "reports/breakthrough_20261002/batch29"
ATT = ROOT / "runs/breakthrough_20261002/batch29/attempts"
DB = ROOT / "data/breakthrough_20261002/batch29"

def rank(a):
    a = np.asarray(a, float); order = np.argsort(a, kind="mergesort")
    r = np.empty(len(a)); sa = a[order]; i = 0
    while i < len(a):
        j = i
        while j + 1 < len(a) and sa[j+1] == sa[i]: j += 1
        r[order[i:j+1]] = 0.5*(i+j)+1.0; i = j+1
    return r

def spearman(a, b):
    ra, rb = rank(a), rank(b); ra = ra-ra.mean(); rb = rb-rb.mean()
    d = np.sqrt((ra*ra).sum()*(rb*rb).sum())
    return float((ra*rb).sum()/d) if d > 0 else float("nan")

def mae(a, b): return float(np.mean(np.abs(np.asarray(a)-np.asarray(b))))

def gbmae(yt, yp, g):
    vals = []
    for u in np.unique(g):
        m = g == u; vals.append(np.mean(np.abs(yt[m]-yp[m])))
    return float(np.mean(vals)), int(len(vals))

# rows for group labels
rows = list(csv.DictReader(open(DB/"rows.tsv", encoding="utf-8"), delimiter="\t"))
grp = np.array([r["group"] for r in rows], dtype=object)
sp  = np.array([r["split"] for r in rows], dtype=object)

out = {}
for aid in ("E1_ridge", "E2_cnn_true_seed20261101"):
    z = np.load(ATT/aid/"val_pred.npz")
    idx = z["idx"]; pred = z["pred"]; true = z["true"]
    # checks
    assert set(grp[idx]) == set(grp[idx]), "ids"
    chk_idx_match = bool(np.all(idx == np.where(sp=="val")[0]))
    chk_true_match = bool(np.allclose(true, np.array([float(rows[i]["y"]) for i in idx])))
    res = json.load(open(ATT/aid/"result.json", encoding="utf-8"))
    r_sp = spearman(pred, true); r_ma = mae(pred, true); r_gb, ng = gbmae(true, pred, grp[idx])
    out[aid] = {
        "n_val": int(len(idx)),
        "replay_spearman": r_sp, "replay_mae": r_ma, "replay_group_balanced_mae": r_gb, "replay_groups": ng,
        "reported_spearman": res.get("val_spearman", res.get("best_val_spearman")),
        "reported_mae": res.get("val_mae", res.get("best_val_mae")),
        "reported_group_balanced_mae": res.get("val_group_balanced_mae", res.get("best_val_group_balanced_mae")),
        "idx_equals_val_rows": chk_idx_match,
        "true_matches_rows_y": chk_true_match,
        "pred_sha256": hashlib.sha256(np.ascontiguousarray(pred).tobytes()).hexdigest(),
    }
    for k in ("spearman", "mae", "group_balanced_mae"):
        a = out[aid]["replay_"+k]; b = out[aid]["reported_"+k]
        out[aid]["abs_diff_"+k] = (None if (a is None or b is None) else abs(float(a)-float(b)))

json.dump(out, open(ROOT/"reports/breakthrough_20261002/batch29_review/replay.json", "w", encoding="utf-8"), indent=2)
print(json.dumps(out, indent=2))
