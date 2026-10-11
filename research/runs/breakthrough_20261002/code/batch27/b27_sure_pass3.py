# -*- coding: utf-8 -*-
"""BATCH27 B pass3: third streaming pass, tier-2 (detection/QC) eligibility cohort.
TIER1 (unchanged, from pass1b): canonical chrom, 200<=len<=2000, finite non-negative counts,
input=iPCR+plasmid+1 >= 10.
TIER2 (predeclared AFTER schema inspection, BEFORE any scoring; a measurement-detection QC rule,
NOT an activity-magnitude rule): biorep1 cDNA sum >= 1 AND biorep2 cDNA sum >= 1.
Emits the tier-2 cohort (coords + labels), its genome-wide overlap-component group graph, and stats.
NO fits, NO model, NO scoring."""
import json, time, zlib, array, gzip
import numpy as np

P = "data/breakthrough_20261002/batch27/source/GSE78709_SuRE-counts-K562_B45_B55_LP170105.txt.gz"
OUTG = "reports/breakthrough_20261002/batch27/sure_tier2_group_graph.json"
TSV = "data/breakthrough_20261002/batch27/sure_tier2_cohort.tsv.gz"
CANON = set(b"chr%d" % i for i in range(1, 23)) | {b"chrX", b"chrY"}
CORD = {c: i for i, c in enumerate(sorted(CANON))}
t0 = time.time()

ch = array.array("i"); st_ = array.array("i"); en_ = array.array("i")
b1 = array.array("i"); b2 = array.array("i"); inp = array.array("i")
lens = []
d = zlib.decompressobj(31); buf = b""; hdr = None; N = 0
n_t1 = 0
n_both = 0
fh = gzip.open(TSV, "wt", encoding="utf-8", newline="")
fh.write("chrom\tstart\tend\tstrand\tinput_iPCR_plus_plasmid_plus1\tbiorep1_cDNA\tbiorep2_cDNA\tlen\n")
with open(P, "rb") as f:
    while True:
        chunk = f.read(1 << 22)
        if not chunk:
            break
        data = d.decompress(chunk)
        if not data:
            continue
        buf += data
        lines = buf.split(b"\n")
        buf = lines.pop()
        for line in lines:
            if not line:
                continue
            if hdr is None:
                hdr = line
                continue
            N += 1
            c = line.split(b"\t")
            if len(c) != 13:
                continue
            chrom = c[0]
            if chrom not in CANON:
                continue
            try:
                st = int(c[1]); en = int(c[2])
                iPCR = int(c[4])
                r1 = int(c[5]) + int(c[6]) + int(c[7]) + int(c[8])
                r2 = int(c[9]) + int(c[10])
                pl = int(c[11]) + int(c[12])
            except Exception:
                continue
            L = en - st
            if not (200 <= L <= 2000):
                continue
            if b"-" in line and min(iPCR, r1, r2, pl) < 0:
                continue
            i_ = iPCR + pl + 1
            if i_ < 10:
                continue
            n_t1 += 1
            if r1 < 1 or r2 < 1:
                continue
            n_both += 1
            idx_ = len(ch)
            ch.append(CORD[chrom]); st_.append(st); en_.append(en)
            b1.append(r1); b2.append(r2); inp.append(i_)
            fh.write("%s\t%d\t%d\t%s\t%d\t%d\t%d\t%d\n" % (chrom.decode(), st, en, c[3].decode(), i_, r1, r2, L))
            if n_both <= 2000000:
                lens.append(L)
        if N % 50000000 < 1 and N:
            print("[pass3] N=%d t1=%d both=%d %.0fs" % (N, n_t1, n_both, time.time() - t0), flush=True)
fh.close()
print("[pass3] stream done N=%d t1=%d both=%d %.0fs" % (N, n_t1, n_both, time.time() - t0), flush=True)

n = len(ch)
parent = array.array("i", range(n))
def find(x):
    while parent[x] != x:
        parent[x] = parent[parent[x]]
        x = parent[x]
    return x
ca = np.frombuffer(ch, dtype=np.int32)
sa = np.frombuffer(st_, dtype=np.int32)
ea = np.frombuffer(en_, dtype=np.int32)
n_edges = 0
for ci in range(len(CORD)):
    idx = np.nonzero(ca == ci)[0]
    if len(idx) < 2:
        continue
    order = idx[np.argsort(sa[idx], kind="mergesort")]
    cur_max = -1; cur_i = -1
    for i in order.tolist():
        if sa[i] < cur_max:
            a = find(cur_i); b = find(i)
            if a != b:
                parent[b] = a
                n_edges += 1
        if ea[i] > cur_max:
            cur_max = ea[i]; cur_i = i
print("[pass3] union-find done %.0fs" % (time.time() - t0), flush=True)

roots = {}
for i in range(n):
    roots.setdefault(find(i), 0)
    roots[find(i)] += 1
sizes = sorted(roots.values(), reverse=True)
import collections
size_hist = collections.Counter(sizes)
la = np.array(lens, dtype=np.int64)
res = {
    "tier1_eligible_genomewide": n_t1,
    "tier2_cohort_size": n,
    "tier2_rule": "biorep1 cDNA sum >= 1 AND biorep2 cDNA sum >= 1 (detection/QC; no activity-magnitude criterion)",
    "tier2_fraction_of_tier1": round(n / n_t1, 6) if n_t1 else None,
    "n_overlap_components": len(roots),
    "n_singleton_components": size_hist.get(1, 0),
    "component_size_hist_top": {str(k): v for k, v in sorted(size_hist.items())[:20]},
    "component_size_max": sizes[0] if sizes else 0,
    "n_overlap_edges": n_edges,
    "len_min": int(la.min()), "len_median": int(np.median(la)), "len_max": int(la.max()),
    "len_hist_100bp": {str(int(k)): int(v) for k, v in zip(*np.unique(la // 100 * 100, return_counts=True))},
    "n_chrom_present": int(len(np.unique(ca))),
    "biorep1_cDNA_total": int(sum(b1)), "biorep2_cDNA_total": int(sum(b2)),
    "elapsed_s": round(time.time() - t0, 1),
}
json.dump(res, open(OUTG, "w", encoding="utf-8"), indent=2, default=str)
print(json.dumps(res, indent=2, default=str))
print("[pass3] DONE %.0fs" % (time.time() - t0), flush=True)
