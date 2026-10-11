# -*- coding: utf-8 -*-
"""Fresh independent reviewer (batch30) PART3b: a CORRECT local-alignment implementation to
cross-check the worker sw_align coverage/identity. ZERO fits."""
from __future__ import annotations
import json, random, sys, time
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "runs/breakthrough_20261002/code/batch30"))
import b30_homology as H  # noqa
RC = str.maketrans("ACGT", "TGCA")
REP = ROOT / "reports/breakthrough_20261002"


def randseq(n, seed):
    r = random.Random(seed)
    return "".join(r.choice("ACGT") for _ in range(n))


def correct_sw(A, B):
    """Standard Smith-Waterman (match +1, mis/gap -1) with exact traceback. Returns (cols,matches,bcons)."""
    m, n = len(A), len(B)
    Hm = np.zeros((m + 1, n + 1), dtype=np.int32)
    P = np.zeros((m + 1, n + 1), dtype=np.int8)  # 0 stop,1 diag,2 up(A consumed),3 left(B consumed)
    ai = np.frombuffer(A.encode(), dtype=np.uint8); bi = np.frombuffer(B.encode(), dtype=np.uint8)
    best = 0; bi_i = 0; bj_j = 0
    for i in range(1, m + 1):
        a = ai[i - 1]
        Hi = Hm[i]; Hp = Hm[i - 1]; Pi = P[i]
        for j in range(1, n + 1):
            s = 1 if bi[j - 1] == a else -1
            d = Hp[j - 1] + s; u = Hp[j] - 1; l = Hi[j - 1] - 1
            v = d
            if u > v: v = u
            if l > v: v = l
            if v <= 0:
                Hi[j] = 0; Pi[j] = 0
            else:
                Hi[j] = v
                Pi[j] = 1 if v == d else (2 if v == u else 3)
                if v > best:
                    best = v; bi_i = i; bj_j = j
    cols = matches = bcons = 0
    i, j = bi_i, bj_j
    while i > 0 and j > 0:
        p = int(P[i, j])
        if p == 1:
            if ai[i - 1] == bi[j - 1]:
                matches += 1
            cols += 1; bcons += 1; i -= 1; j -= 1
        elif p == 2:
            cols += 1; i -= 1
        elif p == 3:
            cols += 1; bcons += 1; j -= 1
        else:
            break
    return cols, matches, bcons


def cov_id(A, B):
    cols, mt, bc = correct_sw(A, B)
    cols2, mt2, bc2 = correct_sw(A, B.translate(RC)[::-1])
    def sc(c, m, b):
        return (b / len(B), m / c) if c else (0.0, 0.0)
    c1, i1 = sc(cols, mt, bc); c2, i2 = sc(cols2, mt2, bc2)
    return (c1, i1) if c1 + i1 >= c2 + i2 else (c2, i2)


def main():
    t0 = time.time()
    OUT = {}
    A = randseq(800, 42); B0 = A[150:650]
    def mut(s, frac, seed, ind=0):
        r = random.Random(seed); s = list(s)
        for i in range(len(s)):
            if r.random() < frac:
                s[i] = r.choice("ACGT")
        out = "".join(s)
        for _ in range(ind):
            p = r.randint(1, len(out) - 1); out = out[:p] + r.choice("ACGT") + out[p:]
        for _ in range(ind):
            p = r.randint(1, len(out) - 2); out = out[:p] + out[p + 1:]
        return out
    # controls: my correct aligner vs worker aligner
    ctrl = {}
    for name, b in [("exact", B0), ("sub3", mut(B0, 0.03, 1)), ("sub6", mut(B0, 0.06, 5)),
                    ("sub15", mut(B0, 0.15, 2)), ("indel10", mut(B0, 0.0, 3, 5)),
                    ("rc", B0.translate(RC)[::-1]), ("unrelated", randseq(500, 99))]:
        cc, ci = cov_id(A, b)
        wk1 = H.sw_align(A, b); wk2 = H.sw_align(A, b.translate(RC)[::-1])
        def s2(c, m, bc): return (bc / len(b), m / c) if c else (0.0, 0.0)
        w1 = s2(*wk1); w2 = s2(*wk2)
        wb = w1 if w1[0] + w1[1] >= w2[0] + w2[1] else w2
        ctrl[name] = {"true_expected_id": {"exact": 1.0, "sub3": 0.97, "sub6": 0.94, "sub15": 0.85,
                                           "indel10": 0.98, "rc": 1.0, "unrelated": 0.25}[name],
                      "correct_cov_id": [round(cc, 3), round(ci, 3)],
                      "worker_cov_id": [round(wb[0], 3), round(wb[1], 3)]}
    OUT["controls_correct_vs_worker"] = ctrl
    seqs = {}
    man = {}
    with open(REP / "batch29/split_manifest.tsv", encoding="utf-8") as f:
        next(f)
        for line in f:
            p = line.rstrip("\n").split("\t"); man[p[0]] = p[1]
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
    seqs = {k: v for k, v in seqs.items() if k in man}
    audit = json.load(open(REP / "batch30/homology_alignment_audit.json", encoding="utf-8"))
    conf = audit["confirmed"]
    res = []
    skipped = 0
    work_budget_s = 150.0
    pairs = []
    for c in conf:
        a, b = c["a"], c["b"]
        if a not in seqs or b not in seqs:
            continue
        A2, B2 = (seqs[a], seqs[b]) if len(seqs[a]) >= len(seqs[b]) else (seqs[b], seqs[a])
        pairs.append((len(A2) * len(B2), c, A2, B2))
    pairs.sort(key=lambda t: t[0])
    for cells, c, A2, B2 in pairs:
        if cells > 600000 or time.time() - t0 > work_budget_s:
            skipped += 1
            continue
        cov, idn = cov_id(A2, B2)
        res.append({"a": c["a"], "b": c["b"], "worker_cov": c["coverage"], "worker_id": c["identity"],
                    "correct_cov": round(cov, 3), "correct_id": round(idn, 3), "cross_block": c["cross_block"]})
    npass = sum(1 for r in res if r["correct_cov"] >= 0.90 and r["correct_id"] >= 0.90)
    OUT["confirmed_recheck_correct_aligner"] = {"n_checked": len(res), "n_skipped_too_large": skipped,
                                                "n_still_90_90": npass,
                                                "n_cross_block": sum(1 for r in res if r["cross_block"]),
                                                "min_correct_id": min(r["correct_id"] for r in res) if res else None,
                                                "elapsed_s": round(time.time() - t0, 1),
                                                "ok": sum(1 for r in res if r["cross_block"]) == 0}
    OUT["confirmed_detail_sample"] = res[:15]
    (REP / "batch30_review" / "rev_homology2.json").write_text(json.dumps(OUT, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(OUT, ensure_ascii=False, indent=1)[:4000])


if __name__ == "__main__":
    main()
