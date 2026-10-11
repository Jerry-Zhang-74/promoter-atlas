# -*- coding: utf-8 -*-
"""Batch38 step1: read ALL 36000 physical design records (preserve dups) and characterise headers."""
import gzip, collections, re, json, os
FA = "data/breakthrough_20261002/batch35/mapt/gsm9640449_VariantMPRAdesignFile.fa.gz"
recs = []
cur = None; buf = []
with gzip.open(FA, "rt", encoding="utf-8") as f:
    for ln in f:
        ln = ln.rstrip("\n")
        if ln.startswith(">"):
            if cur is not None: recs.append((cur, "".join(buf)))
            cur = ln[1:].strip(); buf = []
        else: buf.append(ln)
    recs.append((cur, "".join(buf)))
hdrs = [h for h, s in recs]
print("n_records", len(recs), "n_distinct_headers", len(set(hdrs)))
c = collections.Counter(hdrs)
dups = {k: v for k, v in c.items() if v > 1}
print("n_dup_headers", len(dups), "extra_records", sum(v - 1 for v in dups.values()))
byh = collections.defaultdict(list)
for h, s in recs: byh[h].append(s)
bad = [h for h in dups if len(set(byh[h])) > 1]
print("dup_headers_with_conflicting_seq", len(bad))
lens = collections.Counter(len(s) for h, s in recs)
print("lengths", dict(lens))
pat = collections.Counter()
for h in set(hdrs):
    pat[re.sub(r'\d+', 'N', h)] += 1
print("--- patterns ---")
for k, v in pat.most_common(40): print(v, k)
# sample dup headers
print("--- sample dup headers ---")
for h in list(dups)[:10]: print(dups[h], h)
