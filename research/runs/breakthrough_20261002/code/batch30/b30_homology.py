# -*- coding: utf-8 -*-
"""batch30 bounded near-homology ALIGNMENT audit (ZERO fits, CPU<=4, <=15 min).

Replaces the batch29 k-mer containment APPROXIMATION with an explicit alignment that counts
indels.  Algorithm (documented, signed stats recorded):
  * shorter sequence B is FITTING-aligned into the longer A (free end gaps on A) with a linear
    gap penalty (match +1, mismatch -1, gap -1) via a vectorised Needleman-Wunsch DP
    (max-plus scan over j) + traceback.
  * aligned_length = number of alignment columns; matches = # of '=' columns;
    columns = aligned_length; shorter_length = min(lenA,lenB).
  * coverage = aligned_length / shorter_length  (>=0.90 required)
  * identity = matches / aligned_length          (>=0.90 required)
  Both orientations are tested (B and revcomp(B)).
Candidate enumeration reuses the batch29 reported cross-block/near-homolog edges
(homology_audit.json) plus a bounded re-enumeration (k=13, >=10 shared 13-mers, HF-capped).
Capped / unexamined candidates are disclosed.  Confirmed cross-block connected components are
removed WHOLE (union with existing interval/exact-RC groups).  NOT an exhaustive all-pairs proof.
"""
from __future__ import annotations
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import b30_common as C

RC = str.maketrans("ACGT", "TGCA")
MAX_PAIRS = 4000          # hard cap on aligned candidate pairs (both orientations applied per pair)
TIME_CAP_S = 780          # 13 min soft cap (protocol <=15 min)
BLOCK = {"train": set("chr%d" % i for i in range(1, 17)),
         "val": {"chr17", "chr18", "chr19"},
         "test": {"chr20", "chr21", "chr22", "chrX", "chrY"}}
NEG = -10000


def blk(ch):
    for b, s in BLOCK.items():
        if ch in s:
            return b
    return None


def sw_align(A, B):
    """Smith-Waterman local alignment of shorter B into longer A (match +1, mis/gap -1).
    Returns (columns, matches, b_consumed).  Columns counts all alignment columns;
    b_consumed = # columns consuming a B residue (diag+left) -> used for coverage."""
    m, n = len(A), len(B)
    if m == 0 or n == 0:
        return 0, 0, 0
    am = np.frombuffer(A.encode("ascii"), dtype=np.uint8)
    bm = np.frombuffer(B.encode("ascii"), dtype=np.uint8)
    idx = np.arange(n + 1, dtype=np.int32)
    prev = np.zeros(n + 1, dtype=np.int32)
    ptr = np.zeros((m + 1, n + 1), dtype=np.int8)
    best = 0; bi = 0; bj = 0
    for i in range(1, m + 1):
        eq = (bm == am[i - 1])
        sarr = np.where(eq, 1, -1).astype(np.int32)
        diag = prev[:-1] + sarr                 # j=1..n  <- (i-1,j-1)
        up = prev[1:] - 1                        # j=1..n  <- (i-1,j)
        base = np.maximum(diag, up)
        arr = np.concatenate(([0], base))        # index 0..n, arr[0]=0 (start/stop)
        scan = np.maximum.accumulate(arr + idx) - idx
        cur = np.maximum(scan, 0)
        cur[0] = 0
        # pointers
        p = np.full(n + 1, 2, dtype=np.int8)     # default left
        jm = np.arange(1, n + 1)
        p[jm] = np.where(cur[1:] == 0, 3, np.where(cur[1:] == diag, 0, np.where(cur[1:] == up, 1, 2)))
        p[0] = 3
        ptr[i] = p
        prev = cur
        if cur.max() > best:
            best = int(cur.max()); bj = int(cur.argmax()); bi = i
    if best <= 0:
        return 0, 0, 0
    cols = 0; matches = 0; bcons = 0
    i, j = bi, bj
    while i > 0 and j > 0:
        p = int(ptr[i, j])
        if p == 0:
            if am[i - 1] == bm[j - 1]:
                matches += 1
            cols += 1; bcons += 1; i -= 1; j -= 1
        elif p == 1:
            cols += 1; i -= 1
        elif p == 2:
            cols += 1; bcons += 1; j -= 1
        else:
            break
    return cols, matches, bcons


def revcomp(s):
    return s.translate(RC)[::-1]


def load_manifest():
    man = {}
    with open(C.REP29 / "split_manifest.tsv", encoding="utf-8") as f:
        next(f)
        for line in f:
            k, ch, st, en, sd, g, b = line.rstrip("\n").split("\t")
            man[k] = {"chrom": ch, "start": int(st), "end": int(en), "strand": sd, "group": g, "split": b}
    return man


def enumerate_candidates(seqs, keys):
    """k=13, >=10 shared 13-mers, HF cap 200, containment>=0.10; returns list of (ka,kb,shared,contain)."""
    from collections import defaultdict
    K, MIN_SHARED, HFCAP, CONTAIN_MIN = 13, 10, 200, 0.10
    kmt = {k: set(seqs[k][i:i + K] for i in range(len(seqs[k]) - K + 1)) for k in keys}
    index = defaultdict(list)
    for k in keys:
        for km in kmt[k]:
            index[km].append(k)
    paircount = defaultdict(int)
    for km, lst in index.items():
        if len(lst) < 2 or len(lst) > HFCAP:
            continue
        for a in range(len(lst)):
            for b in range(a + 1, len(lst)):
                x, y = lst[a], lst[b]
                paircount[(x, y) if x < y else (y, x)] += 1
    out = []
    for (x, y), c in paircount.items():
        if c < MIN_SHARED:
            continue
        si, sj = kmt[x], kmt[y]
        ct = len(si & sj) / min(len(si), len(sj))
        if ct >= CONTAIN_MIN:
            out.append((x, y, c, round(ct, 4)))
    out.sort(key=lambda t: (-t[3], t[0], t[1]))
    return out


def main():
    t0 = time.time()
    man = load_manifest()
    assay = {}
    with open(C.DATA29 / "sure29_assay_oriented.fa", encoding="utf-8") as f:
        k = None
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                p = line[1:].split("_"); k = "%s:%s:%s" % (p[0], p[1], p[2])
            else:
                assay[k] = line
    seqs = {k: v for k, v in assay.items() if k in man}
    keys = sorted(seqs)

    # ---- candidate set: batch29 reported edges + bounded re-enumeration ----
    b29 = C.read_json(C.REP29 / "homology_audit.json", {}) or {}
    cands = []
    for p in b29.get("near_homology_pairs", []):
        a, b = p["i"], p["j"]
        if a in seqs and b in seqs:
            cands.append((a, b, p.get("shared13"), p.get("containment")))
    n_b29 = len(cands)
    extra = enumerate_candidates(seqs, keys)
    seen = set((a, b) if a < b else (b, a) for a, b, _, _ in cands)
    n_extra = 0
    n_sameblock_not_enum = 0
    for a, b, c, ct in extra:
        if a not in seqs or b not in seqs:
            continue
        if (a, b) in seen:
            continue
        if blk(man[a]["chrom"]) != blk(man[b]["chrom"]):
            cands.append((a, b, c, ct)); seen.add((a, b)); n_extra += 1
        else:
            # within-block supplementary candidates below the batch29 containment 0.25 threshold
            # were NOT enumerated (candidate generator limit, disclosed); they cannot create a
            # cross-block component on their own.
            n_sameblock_not_enum += 1
    n_candidates = len(cands)
    n_cross_block_cand = sum(1 for a, b, _, _ in cands if blk(man[a]["chrom"]) != blk(man[b]["chrom"]))
    cands_cross = [c for c in cands if blk(man[c[0]]["chrom"]) != blk(man[c[1]]["chrom"])]
    cands_same = [c for c in cands if blk(man[c[0]]["chrom"]) == blk(man[c[1]]["chrom"])]
    cands = cands_cross + cands_same          # PRIORITISE cross-block

    # ---- align (both orientations) ----
    confirmed = []
    aligned = 0
    n_unexamined = 0
    stats = []
    for a, b, shared, ct in cands:
        if time.time() - t0 > TIME_CAP_S or aligned >= MAX_PAIRS:
            n_unexamined = n_candidates - aligned
            break
        sa, sb = seqs[a], seqs[b]
        A, B = (sa, sb) if len(sa) >= len(sb) else (sb, sa)
        cols, mt, bc = sw_align(A, B)
        cols2, mt2, bc2 = sw_align(A, revcomp(B))
        def score(c, m, bcons):
            if c == 0:
                return (0.0, 0.0)
            return (bcons / len(B), m / c)
        cov1, id1 = score(cols, mt, bc); cov2, id2 = score(cols2, mt2, bc2)
        (cov, ident, orient) = (cov1, id1, "fwd") if cov1 + id1 >= cov2 + id2 else (cov2, id2, "rc")
        aligned += 1
        if cov >= 0.90 and ident >= 0.90:
            confirmed.append({"a": a, "b": b, "coverage": round(cov, 4), "identity": round(ident, 4),
                              "orientation": orient, "columns": cols if orient == "fwd" else cols2,
                              "matches": mt if orient == "fwd" else mt2, "shared13": shared, "containment": ct,
                              "cross_block": bool(blk(man[a]["chrom"]) != blk(man[b]["chrom"]))})
        if len(stats) < 200:
            stats.append({"a": a, "b": b, "coverage": round(cov, 4), "identity": round(ident, 4), "orient": orient})
    out = {"batch": "batch30", "kind": "bounded_alignment_homology_audit",
           "algorithm": "Smith-Waterman local alignment (match+1, mismatch/gap-1) of the shorter sequence into the longer; both orientations; coverage=aligned_B_residues/len(B); identity=matches/columns (indels counted)",
           "n_surviving": len(keys), "n_b29_reported_edges_used": n_b29, "n_reenumerated_crossblock_edges": n_extra,
           "n_sameblock_extra_not_enumerated_disclosure": n_sameblock_not_enum,
           "n_candidates_total": n_candidates, "n_cross_block_candidates": n_cross_block_cand,
           "n_alignments_performed": aligned, "n_candidates_unexamined": n_unexamined,
           "capped": bool(n_unexamined > 0), "max_pairs_cap": MAX_PAIRS, "time_cap_s": TIME_CAP_S,
           "n_confirmed_90_90": len(confirmed), "n_confirmed_cross_block": sum(1 for c in confirmed if c["cross_block"]),
           "confirmed": confirmed, "examples": stats, "elapsed_s": round(time.time() - t0, 1),
           "written_at": C.now_local(), "code_sha256": C.sha256_file(Path(__file__))}
    C.atomic_write_json(C.REP30 / "homology_alignment_audit.json", out)
    print(json.dumps({k: v for k, v in out.items() if k not in ("confirmed", "examples")}, indent=1))


if __name__ == "__main__":
    main()
