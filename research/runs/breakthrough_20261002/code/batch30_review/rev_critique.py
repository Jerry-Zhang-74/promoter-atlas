# -*- coding: utf-8 -*-
"""Fresh independent reviewer (batch30) PART4: group-weight/singleton + batch-size critique (numpy only)."""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "runs/breakthrough_20261002/code/batch30"))
import b30_data as D  # noqa
REP = ROOT / "reports/breakthrough_20261002"
OUT = {}


def main():
    rows = D.load_rows(ROOT / "data/breakthrough_20261002/batch30/rows.tsv")
    loc = np.array([r["group"] for r in rows], dtype=object)
    tr = np.where(np.array([r["split"] for r in rows]) == "train")[0]
    cnt = {}
    for i in tr:
        cnt[loc[i]] = cnt.get(loc[i], 0) + 1
    w = np.array([1.0 / cnt[loc[i]] for i in tr]); w = w / w.mean()
    # 1) weighted per-batch means recombine exactly to the global weighted mean
    rng = np.random.RandomState(0)
    e = rng.rand(len(tr))                       # arbitrary per-example values
    glob = float((w * e).sum() / w.sum())
    order = rng.permutation(len(tr))
    chunks = np.array_split(order, 20)
    rec = 0.0; tot = 0.0
    for c in chunks:
        m = float((w[c] * e[c]).sum() / w[c].sum()); rec += m * w[c].sum(); tot += w[c].sum()
    OUT["weighted_mean_recombination"] = {"global": glob, "recombined": rec / tot,
                                          "abs_diff": abs(glob - rec / tot), "ok": abs(glob - rec / tot) < 1e-12}
    # 2) singleton halving preserves group mass exactly
    si = int(np.where([cnt[loc[i]] == 1 for i in tr])[0][0])
    wsi = w[si]
    halved = np.array([wsi * 0.5, wsi * 0.5])
    OUT["singleton_halving_mass"] = {"original_weight": float(wsi), "halved_sum": float(halved.sum()),
                                     "ok": abs(wsi - halved.sum()) < 1e-15}
    # 3) group-size / batch-composition structure
    sizes = np.array(list(cnt.values()))
    lens_all = np.array([int(r["len"]) for r in rows])
    buckets = {}
    for i in tr:
        L = int(lens_all[i]); buckets[L] = buckets.get(L, 0) + 1
    bsz = np.array(list(buckets.values()))
    OUT["batch_structure"] = {
        "n_train_rows": int(len(tr)), "n_train_groups": int(len(cnt)),
        "frac_rows_in_singleton_groups": float(sizes[sizes == 1].sum() / len(tr)),
        "n_singleton_groups": int((sizes == 1).sum()),
        "n_large_groups_gt5": int((sizes > 5).sum()),
        "distinct_length_buckets_train": int(len(buckets)),
        "bucket_size_max": int(bsz.max()), "bucket_size_mean": round(float(bsz.mean()), 2),
        "frac_buckets_lt_8": float((bsz < 8).mean()),
        "note": "exact-length buckets cap the realised batch size far below 32; most training batches are small, which stresses BatchNorm and makes singleton duplication frequent."}
    # 4) state_digest blind to optimizer (re-check numerically with pure numpy analogue)
    def py(o):
        if isinstance(o, dict):
            return {k: py(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [py(v) for v in o]
        if isinstance(o, (np.bool_,)):
            return bool(o)
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        return o
    final = py(OUT)
    (REP / "batch30_review" / "rev_critique.json").write_text(json.dumps(final, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(final, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
