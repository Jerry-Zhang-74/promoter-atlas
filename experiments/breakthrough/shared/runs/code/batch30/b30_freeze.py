# -*- coding: utf-8 -*-
"""batch30 SPLIT / INPUT freeze + atomic prefit_gate.json.  ZERO fits.

Applies the bounded alignment-audit confirmed cross-block components (removed WHOLE, union with
the existing interval/exact-RC groups) to the surviving batch29 set.  If no additional removals,
the batch30 split is hash-equivalent to batch29 and may be referenced; otherwise a NEW split is
written and the next baseline must refit.  Freezes train chr1-16 / val chr17-19 / test chr20-22,X,Y.
"""
from __future__ import annotations
import json
import shutil
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import b30_common as C
import b30_data as D

BLOCK = {"train": set("chr%d" % i for i in range(1, 17)),
         "val": {"chr17", "chr18", "chr19"},
         "test": {"chr20", "chr21", "chr22", "chrX", "chrY"}}


def blk(ch):
    for b, s in BLOCK.items():
        if ch in s:
            return b
    return None


def main():
    out_dir = C.DATA30; out_dir.mkdir(parents=True, exist_ok=True)
    # load batch29 frozen manifest (surviving 9013)
    man = []
    with open(C.REP29 / "split_manifest.tsv", encoding="utf-8") as f:
        next(f)
        for line in f:
            k, ch, st, en, sd, g, b = line.rstrip("\n").split("\t")
            man.append({"key": k, "chrom": ch, "start": int(st), "end": int(en), "strand": sd, "group": g, "split": b})
    keys = [m["key"] for m in man]
    k2m = {m["key"]: m for m in man}

    aud = C.read_json(C.REP30 / "homology_alignment_audit.json")
    if aud is None:
        raise SystemExit("homology alignment audit missing; cannot freeze")
    confirmed = aud.get("confirmed", [])
    capped = bool(aud.get("capped")) or int(aud.get("n_candidates_unexamined", 0)) > 0

    # union-find over GROUPS, seeded from manifest group ids
    gids = sorted(set(m["group"] for m in man))
    par = {g: g for g in gids}

    def find(x):
        r = x
        while par[r] != r:
            r = par[r]
        while par[x] != r:
            par[x], x = r, par[x]
        return r

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            par[rb] = ra

    for e in confirmed:
        a, b = e["a"], e["b"]
        if a in k2m and b in k2m:
            union(k2m[a]["group"], k2m[b]["group"])

    # component -> set of blocks
    comp_blocks = defaultdict(set)
    comp_members = defaultdict(list)
    for m in man:
        r = find(m["group"]); comp_blocks[r].add(blk(m["chrom"])); comp_members[r].append(m["key"])
    drop = set()
    n_multiblock = 0
    for r, bs in comp_blocks.items():
        if len(bs) > 1:
            n_multiblock += 1
            drop.update(comp_members[r])
    # batch29 already dropped multi-block components; require >=2 members to call it "new"
    new_removals = sorted(drop)
    kept = [m for m in man if m["key"] not in drop]

    # write batch30 split.tsv
    with open(out_dir / "split.tsv", "w", encoding="utf-8", newline="\n") as f:
        f.write("key\tchrom\tstart\tend\tstrand\tgroup\tsplit\n")
        for m in kept:
            f.write("%s\t%s\t%d\t%d\t%s\t%s\t%s\n" % (m["key"], m["chrom"], m["start"], m["end"], m["strand"], m["group"], m["split"]))

    # build batch30 rows.tsv + native encoded.npz (subset in kept order)
    z = np.load(C.DATA29 / "encoded.npz")
    codes_all, lens_all = z["codes"], z["lens"]
    idx29 = {k: i for i, k in enumerate(keys)}
    keep_idx = np.array([idx29[m["key"]] for m in kept], dtype=np.int64)
    codes = codes_all[keep_idx]; lens = lens_all[keep_idx]
    rows29 = D.load_rows(C.DATA29 / "rows.tsv")
    r29 = {r["key"]: r for r in rows29}
    with open(out_dir / "rows.tsv", "w", encoding="utf-8", newline="\n") as f:
        cols = list(rows29[0].keys())
        f.write("\t".join(cols) + "\n")
        for m in kept:
            r = r29[m["key"]]
            f.write("\t".join(str(r[c]) for c in cols) + "\n")
    np.savez_compressed(out_dir / "encoded.npz.tmp.npz", codes=codes, lens=lens)
    import os
    os.replace(out_dir / "encoded.npz.tmp.npz", out_dir / "encoded.npz")

    counts = {}
    for s in ("train", "val", "test"):
        sub = [m for m in kept if m["split"] == s]
        counts[s] = {"inserts": len(sub), "groups": len(set(m["group"] for m in sub)),
                     "chroms": sorted(set(m["chrom"] for m in sub))}
    gates = {"train_inserts>=1000": counts["train"]["inserts"] >= 1000,
             "train_groups>=500": counts["train"]["groups"] >= 500,
             "val_groups>=100": counts["val"]["groups"] >= 100,
             "test_groups>=100": counts["test"]["groups"] >= 100,
             "val_chroms>=2": len(counts["val"]["chroms"]) >= 2,
             "test_chroms>=2": len(counts["test"]["chroms"]) >= 2}
    res = {"batch": "batch30", "split_rule": "chromosome blocks train chr1-16 / val chr17-19 / test chr20-22,X,Y",
           "n_b29_surviving": len(man), "n_confirmed_alignment_edges": len(confirmed),
           "n_new_multiblock_components": n_multiblock, "n_new_removals": len(new_removals),
           "new_removals": new_removals[:200], "n_kept": len(kept),
           "hash_equivalent_to_b29_split": (len(new_removals) == 0),
           "counts": counts, "gates": gates, "gate_all": all(gates.values()),
           "audit_capped_or_unexamined": capped,
           "split_tsv_sha256": C.sha256_file(out_dir / "split.tsv"),
           "rows_tsv_sha256": C.sha256_file(out_dir / "rows.tsv"),
           "encoded_npz_sha256": C.sha256_file(out_dir / "encoded.npz"),
           "written_at": C.now_local()}
    C.atomic_write_json(out_dir / "split.json", res)
    C.atomic_write_json(C.REP30 / "split_freeze.json", res)
    print(json.dumps({k: v for k, v in res.items() if k != "new_removals"}, indent=1))


if __name__ == "__main__":
    main()
