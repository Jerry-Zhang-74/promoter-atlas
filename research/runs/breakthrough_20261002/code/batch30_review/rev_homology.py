# -*- coding: utf-8 -*-
"""Fresh independent reviewer (batch30) PART3: homology independent checks. ZERO fits.
Validates the worker sw_align on synthetic ground-truth controls, independently recomputes
k-mer containment for confirmed vs cross-block candidates, and independently re-enumerates the
cross-block candidate set."""
from __future__ import annotations
import json, random, sys, time
from collections import defaultdict
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "runs/breakthrough_20261002/code/batch30"))
import b30_homology as H   # noqa
RC = str.maketrans("ACGT", "TGCA")
OUT = {}
REP = ROOT / "reports/breakthrough_20261002"


def randseq(n, seed):
    r = random.Random(seed)
    return "".join(r.choice("ACGT") for _ in range(n))


def mutate(s, frac, seed, indels=0):
    r = random.Random(seed); s = list(s)
    for i in range(len(s)):
        if r.random() < frac:
            s[i] = r.choice("ACGT")
    out = "".join(s)
    for _ in range(indels):
        p = r.randint(1, len(out) - 1)
        out = out[:p] + r.choice("ACGT") + out[p:]
    for _ in range(indels):
        p = r.randint(1, len(out) - 2)
        out = out[:p] + out[p + 1:]
    return out


def controls():
    A = randseq(800, 42); B = A[150:650]
    cases = {"exact": B, "sub3": mutate(B, 0.03, 1), "sub15": mutate(B, 0.15, 2),
             "indel10": mutate(B, 0.0, 3, indels=5), "rc": B.translate(RC)[::-1],
             "unrelated": randseq(500, 99)}
    res = {}
    for name, b in cases.items():
        cols, mt, bc = H.sw_align(A, b)
        cols2, mt2, bc2 = H.sw_align(A, b.translate(RC)[::-1])
        def sc(c, m, bc):
            return (bc / len(b), m / c) if c else (0.0, 0.0)
        c1, i1 = sc(cols, mt, bc); c2, i2 = sc(cols2, mt2, bc2)
        res[name] = {"len_b": len(b), "fwd": [round(c1, 4), round(i1, 4)], "rc": [round(c2, 4), round(i2, 4)],
                     "best": [round(max(c1, c2), 4), round(i1 if c1 >= c2 else i2, 4)]}
    return res


def load_seqs():
    man = {}
    with open(REP / "batch29/split_manifest.tsv", encoding="utf-8") as f:
        next(f)
        for line in f:
            k, ch, st, en, sd, g, b = line.rstrip("\n").split("\t")
            man[k] = ch
    seqs = {}
    with open(ROOT / "data/breakthrough_20261002/batch29/sure29_assay_oriented.fa", encoding="utf-8") as f:
        k = None
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                p = line[1:].split("_"); k = "%s:%s:%s" % (p[0], p[1], p[2])
            else:
                seqs[k] = line
    return {k: v for k, v in seqs.items() if k in man}, man


def blocks(ch):
    if ch in set("chr%d" % i for i in range(1, 17)):
        return "train"
    if ch in {"chr17", "chr18", "chr19"}:
        return "val"
    if ch in {"chr20", "chr21", "chr22", "chrX", "chrY"}:
        return "test"
    return None


def reenum_crossblock(seqs, keys, K=13, MIN_SHARED=10, HFCAP=200, CONTAIN=0.10):
    kmt = {k: set(seqs[k][i:i + K] for i in range(len(seqs[k]) - K + 1)) for k in keys}
    index = defaultdict(list)
    for k in keys:
        for km in kmt[k]:
            index[km].append(k)
    pc = defaultdict(int)
    for km, lst in index.items():
        if len(lst) < 2 or len(lst) > HFCAP:
            continue
        for a in range(len(lst)):
            for b in range(a + 1, len(lst)):
                x, y = lst[a], lst[b]
                pc[(x, y) if x < y else (y, x)] += 1
    cross = []
    for (x, y), c in pc.items():
        if c < MIN_SHARED:
            continue
        ct = len(kmt[x] & kmt[y]) / min(len(kmt[x]), len(kmt[y]))
        if ct >= CONTAIN and blocks(x.split(":")[0]) != blocks(y.split(":")[0]):
            cross.append((x, y, c, round(ct, 4)))
    cross.sort(key=lambda t: -t[3])
    return cross


def main():
    t0 = time.time()
    OUT["sw_controls"] = controls()
    seqs, man = load_seqs()
    keys = sorted(seqs)
    audit = json.load(open(REP / "batch30/homology_alignment_audit.json", encoding="utf-8"))
    conf = audit["confirmed"]
    # independent k-mer containment for confirmed vs cross-block candidates
    K = 13
    def kc(a, b):
        sa = set(seqs[a][i:i + K] for i in range(len(seqs[a]) - K + 1))
        sb = set(seqs[b][i:i + K] for i in range(len(seqs[b]) - K + 1))
        return len(sa & sb) / min(len(sa), len(sb))
    conf_kc = [round(kc(c["a"], c["b"]), 3) for c in conf if c["a"] in seqs and c["b"] in seqs]
    OUT["confirmed_pairs"] = {"n": len(conf), "n_cross_block": sum(1 for c in conf if c["cross_block"]),
                              "min_cov": min(c["coverage"] for c in conf), "min_id": min(c["identity"] for c in conf),
                              "kmer_containment_min": min(conf_kc), "kmer_containment_mean": round(sum(conf_kc) / len(conf_kc), 3),
                              "all_in_seqs": all(c["a"] in seqs and c["b"] in seqs for c in conf),
                              "ok": (sum(1 for c in conf if c["cross_block"]) == 0
                                     and min(c["coverage"] for c in conf) >= 0.90
                                     and min(c["identity"] for c in conf) >= 0.90
                                     and min(conf_kc) > 0.2454)}
    cross = reenum_crossblock(seqs, keys)
    OUT["reenumeration"] = {"n_cross_block_candidates": len(cross),
                            "worker_reported": audit["n_reenumerated_crossblock_edges"],
                            "match": len(cross) == audit["n_reenumerated_crossblock_edges"],
                            "max_containment": max(c[3] for c in cross) if cross else None,
                            "ok": len(cross) == audit["n_reenumerated_crossblock_edges"]}
    # align a sample of the most containment-similar cross-block candidates with the worker aligner
    sample = cross[:40]
    hits = []
    for a, b, c, ct in sample:
        A, B = (seqs[a], seqs[b]) if len(seqs[a]) >= len(seqs[b]) else (seqs[b], seqs[a])
        cols, mt, bc = H.sw_align(A, B)
        cols2, mt2, bc2 = H.sw_align(A, B.translate(RC)[::-1])
        cov = max(bc / len(B) if cols else 0, bc2 / len(B) if cols2 else 0)
        idn = max(mt / cols if cols else 0, mt2 / cols2 if cols2 else 0)
        if cov >= 0.9 and idn >= 0.9:
            hits.append({"a": a, "b": b, "containment": ct, "cov": round(cov, 3), "id": round(idn, 3)})
    OUT["crossblock_sample_alignment"] = {"n_sampled": len(sample), "n_90_90_hits": len(hits), "hits": hits,
                                          "elapsed_s": round(time.time() - t0, 1),
                                          "ok": len(hits) == 0}
    out = REP / "batch30_review" / "rev_homology.json"
    out.write_text(json.dumps(OUT, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(OUT, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
