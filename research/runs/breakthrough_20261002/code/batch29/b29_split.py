# -*- coding: utf-8 -*-
"""BATCH29 split + grouping (label-free).

Groups = connected components of:
  (a) overlapping genomic intervals (transitive, max-end sweep, irrespective of orientation),
  (b) exact / reverse-complement sequence duplicates,
  (c) confirmed near-homologs (k-mer candidate + containment verification, bounded).
Predeclared chromosome blocks: train chr1-16 / val chr17-19 / test chr20-22,X,Y.
Any component touching >1 block is DROPPED from all splits.  NO model, NO labels used."""
import json, time, os, hashlib
from collections import defaultdict

B = "reports/breakthrough_20261002/batch29"
SEL = "data/breakthrough_20261002/batch29/selected_rows.json"
FAA = "data/breakthrough_20261002/batch29/sure29_assay_oriented.fa"
K = 13
MIN_SHARED = 10          # candidate 13-mer shared count
CONTAIN_MIN = 0.25       # ~= 90% identity at k=13 (0.9^13 = 0.254)
HFCAP = 200              # skip 13-mers occurring in >HFCAP sequences (low-complexity / repeats)

t0 = time.time()
rows = json.load(open(SEL, encoding="utf-8"))["rows"]
key2row = {r["key"]: r for r in rows}

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
usable = [k for k in key2row if k in assay]     # DNA-retrieved, ACGT-only
usable.sort()
idx = {k: i for i, k in enumerate(usable)}
n = len(usable)
parent = list(range(n))
def find(x):
    while parent[x] != x:
        parent[x] = parent[parent[x]]; x = parent[x]
    return x
def union(a, b):
    ra, rb = find(a), find(b)
    if ra != rb: parent[rb] = ra

# (a) interval-overlap components (max-end sweep per chromosome, orientation-agnostic)
by_chrom = defaultdict(list)
for k in usable:
    r = key2row[k]
    by_chrom[r["chrom"]].append((r["start"], r["end"], k))
n_overlap_merges = 0
for ch, lst in by_chrom.items():
    lst.sort()
    cur_end = -1; cur_key = None
    for st, en, k in lst:
        if st < cur_end:                       # overlaps current component
            if find(idx[k]) != find(idx[cur_key]):
                union(idx[cur_key], idx[k]); n_overlap_merges += 1
            if en > cur_end: cur_end = en; cur_key = k
        else:
            cur_end = en; cur_key = k

# (b) exact / RC duplicates (assay-oriented)
RC = str.maketrans("ACGT", "TGCA")
seen = {}; rcseen = {}; n_dup_merges = 0
for k in usable:
    s = assay[k]
    if s in seen:
        if find(idx[k]) != find(idx[seen[s]]): union(idx[k], idx[seen[s]]); n_dup_merges += 1
    else: seen[s] = k
    r = s.translate(RC)[::-1]
    if r in rcseen:
        if find(idx[k]) != find(idx[rcseen[r]]): union(idx[k], idx[rcseen[r]]); n_dup_merges += 1
    else: rcseen[r] = k

# (c) near-homology: 13-mer index, bounded
kmt = {}
for k in usable:
    s = assay[k]
    kmt[idx[k]] = set(s[i:i + K] for i in range(len(s) - K + 1))
    if time.time() - t0 > 600:                 # hard 10-min cap on homology phase
        break
kmer_index = defaultdict(list)
for i, ks in kmt.items():
    for km in ks: kmer_index[km].append(i)
paircount = defaultdict(int)
pairops = 0
for km, lst in kmer_index.items():
    if len(lst) < 2 or len(lst) > HFCAP: continue
    for a in range(len(lst)):
        for b in range(a + 1, len(lst)):
            paircount[(lst[a], lst[b])] += 1
            pairops += 1
    if pairops > 40000000: break
cand = [(p, c) for p, c in paircount.items() if c >= MIN_SHARED]
n_cand = len(cand)
n_hom_merges = 0; paired = []
for (i, j), c in cand:
    si, sj = kmt.get(i, set()), kmt.get(j, set())
    if not si or not sj: continue
    contain = len(si & sj) / min(len(si), len(sj))
    if contain >= CONTAIN_MIN:
        paired.append((i, j, c, round(contain, 4)))
        if find(i) != find(j): union(i, j); n_hom_merges += 1

# components
comp = defaultdict(list)
for k in usable: comp[find(idx[k])].append(k)

BLOCK = {"train": set("chr%d" % i for i in range(1, 17)),
         "val": {"chr17", "chr18", "chr19"},
         "test": {"chr20", "chr21", "chr22", "chrX", "chrY"}}
def blk(ch):
    for b, s in BLOCK.items():
        if ch in s: return b
    return None

excluded_comp = 0; excluded_keys = 0
man = []          # key, chrom,start,end,strand, group, split
for root, members in comp.items():
    blocks = set(blk(key2row[k]["chrom"]) for k in members)
    if len(blocks) > 1:
        excluded_comp += 1; excluded_keys += len(members); continue
    b = blocks.pop()
    members_sorted = sorted(members, key=lambda k: idx[k])
    gid = "B29C%06d" % idx[members_sorted[0]]
    for k in members_sorted:
        r = key2row[k]
        man.append((k, r["chrom"], r["start"], r["end"], r["strand"], gid, b))
man.sort(key=lambda x: idx[x[0]])

def stats(sp):
    sub = [m for m in man if m[6] == sp]
    return {"inserts": len(sub), "groups": len(set(m[5] for m in sub)),
            "chroms": sorted(set(m[1] for m in sub))}

counts = {s: stats(s) for s in ("train", "val", "test")}
gates = {
    "train_inserts>=1000": counts["train"]["inserts"] >= 1000,
    "train_groups>=500": counts["train"]["groups"] >= 500,
    "val_groups>=100": counts["val"]["groups"] >= 100,
    "test_groups>=100": counts["test"]["groups"] >= 100,
    "val_chroms>=2": len(counts["val"]["chroms"]) >= 2,
    "test_chroms>=2": len(counts["test"]["chroms"]) >= 2,
}
res = {"batch": "batch29",
 "grouping_rule": "connected components of interval overlap + exact/RC dup + confirmed near-homolog",
 "near_homology_method": {"k": K, "candidate_min_shared_kmers": MIN_SHARED,
   "containment_threshold": CONTAIN_MIN,
   "identity_note": "containment ~ identity^k; threshold 0.25 approx 90%% identity at k=13",
   "HF_cap_skipped_kmers": HFCAP,
   "residual": "k-mer/containment approximation, capped pair search and HF-kmer skipping; NOT a full all-pairs Smith-Waterman guarantee"},
 "split_rule": "chromosome blocks train chr1-16, val chr17-19, test chr20-22,X,Y",
 "n_selected": len(rows), "n_usable_sequence": n,
 "n_interval_overlap_merges": n_overlap_merges,
 "n_exact_or_rc_dup_merges": n_dup_merges,
 "n_near_homolog_candidates": n_cand, "n_near_homolog_merges": n_hom_merges,
 "n_homolog_pairs": len(paired),
 "n_multiblock_components_excluded": excluded_comp,
 "n_inserts_excluded_multiblock": excluded_keys,
 "counts": counts, "gates": gates,
 "gate_all": all(gates.values()),
 "elapsed_s": round(time.time() - t0, 1)}
json.dump(res, open(B + "/split.json", "w", encoding="utf-8"), indent=2)
json.dump({"near_homology_pairs": [{"i": usable[i], "j": usable[j], "shared13": c, "containment": ct}
                                   for i, j, c, ct in paired[:5000]]},
          open(B + "/homology_audit.json", "w", encoding="utf-8"), indent=1)
with open(B + "/split_manifest.tsv", "w", encoding="utf-8", newline="\n") as f:
    f.write("key\tchrom\tstart\tend\tstrand\tgroup\tsplit\n")
    for k, ch, st, en, sd, g, b in man:
        f.write("%s\t%s\t%d\t%d\t%s\t%s\t%s\n" % (k, ch, st, en, sd, g, b))
print(json.dumps(res, indent=2))
