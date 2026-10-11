# -*- coding: utf-8 -*-
"""Batch38 step2: KV SNV identity audit against COMPLETE unmodified design records (36000, dups preserved).
Does NOT assume <chrpos>_referenceseq is the known allele; tests every hypothesis."""
import gzip, collections, re, csv, json, hashlib
FA = "data/breakthrough_20261002/batch35/mapt/gsm9640449_VariantMPRAdesignFile.fa.gz"
KV = "data/breakthrough_20261002/batch36/mapt/cohort_KV_SNV_HEK293FT.tsv"

def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""): h.update(b)
    return h.hexdigest()

# load ALL physical records, preserving multiplicity
recs = []
with gzip.open(FA, "rt", encoding="utf-8") as f:
    cur = None; buf = []
    for ln in f:
        ln = ln.rstrip("\n")
        if ln.startswith(">"):
            if cur is not None: recs.append((cur, "".join(buf)))
            cur = ln[1:].strip(); buf = []
        else: buf.append(ln)
    recs.append((cur, "".join(buf)))
byh = collections.defaultdict(list)
for h, s in recs: byh[h].append(s)
# index by position prefix for KV-like SNV records: SNV_chrX_pos_*
pos_index = collections.defaultdict(list)
for h in set(byh):
    m = re.match(r'^SNV_(chr[\w]+)_(\d+)_(.+)$', h)
    if m: pos_index[(m.group(1), m.group(2))].append(h)

rows = list(csv.DictReader(open(KV, encoding="utf-8"), delimiter="\t"))
print("KV rows", len(rows))
print("KV sha256", sha256_file(KV))
print("design sha256", sha256_file(FA))

def base_at(s, i): return s[i] if 0 <= i < len(s) else "-"

n = 0
sample = []
statuses = collections.Counter()
detail = []
for r in rows:
    oid = r["original_id"]; alt_h = r["design_header"]
    m = re.match(r'^SNV_(chr[\w]+)_(\d+)_altseq_([ACGT])_to_([ACGT])$', alt_h)
    ref_h = re.sub(r'_altseq_[ACGT]_to_[ACGT]$', '_referenceseq', alt_h)
    alt_seq = byh.get(alt_h, [None])[0]
    ref_seq = byh.get(ref_h, [None])[0]
    row = {"original_id": oid, "alt_h": alt_h, "ref_h": ref_h}
    if alt_seq is None or ref_seq is None:
        statuses["MISSING_RECORD"] += 1; row["status"] = "MISSING_RECORD"; detail.append(row); continue
    diffs = [i for i in range(min(len(alt_seq), len(ref_seq))) if alt_seq[i] != ref_seq[i]]
    L = len(alt_seq)
    row.update({"len_alt": L, "len_ref": len(ref_seq), "n_diff": len(diffs),
                "diff_pos": diffs[:5], "base_ref_tbl": r["base_ref"], "base_alt_tbl": r["base_alt"],
                "hdr_ref": m.group(3), "hdr_alt": m.group(4)})
    # bases at various offsets
    for off in (123, 124, 125, 126):
        row["alt@%d" % off] = base_at(alt_seq, off)
        row["ref@%d" % off] = base_at(ref_seq, off)
    if len(diffs) == 0:
        st = "IDENTICAL_ALT_EQ_REF"
    elif len(diffs) == 1:
        st = "SINGLE_DIFF@%d" % diffs[0]
    else:
        st = "MULTI_DIFF@%s" % (diffs[:3],)
    row["status"] = st
    statuses[st] += 1
    # alt base at the diff position equals header alt?
    if diffs:
        d0 = diffs[0]
        row["altbase@diff_eq_hdr_alt"] = (alt_seq[d0] == m.group(4))
        row["refbase@diff_eq_hdr_ref"] = (ref_seq[d0] == m.group(3))
    detail.append(row)
    n += 1

print("--- status counts (alt vs derived referenceseq header) ---")
for k, v in statuses.most_common(): print(v, k)

# positional index: what other records exist at the same position?
multi_oligo = 0
examples = []
for r in rows[:0]: pass
print("--- example rows ---")
for row in detail[:8]:
    print(json.dumps({k: row.get(k) for k in ["original_id","ref_h","n_diff","diff_pos","hdr_ref","hdr_alt",
        "base_ref_tbl","base_alt_tbl","alt@124","ref@124","alt@125","ref@125","status","altbase@diff_eq_hdr_alt","refbase@diff_eq_hdr_ref"]}, ensure_ascii=False))
# aggregate alt/ref base at 125 vs header allele
agg = collections.Counter()
for row in detail:
    if "alt@125" not in row: continue
    agg[("alt@125==hdr_alt", row["alt@125"] == row["hdr_alt"])] += 1
    agg[("ref@125==hdr_ref", row["ref@125"] == row["hdr_ref"])] += 1
    agg[("ref@125==hdr_alt", row["ref@125"] == row["hdr_alt"])] += 1
    agg[("alt@125==hdr_ref", row["alt@125"] == row["hdr_ref"])] += 1
    agg[("ref@125==genome_ref_tbl", row["ref@125"] == row["base_ref_tbl"])] += 1
print("--- aggregates at offset125 ---")
for k, v in sorted(agg.items()): print(v, k)
# save detail
json.dump({"statuses": dict(statuses), "n": len(rows)}, open("data/breakthrough_20261002/batch38/step2_status.json","w"), indent=1)
with open("data/breakthrough_20261002/batch38/kv_identity_detail.tsv","w",encoding="utf-8",newline="") as f:
    keys = ["original_id","alt_h","ref_h","status","n_diff","len_alt","len_ref","hdr_ref","hdr_alt",
            "base_ref_tbl","base_alt_tbl","alt@123","ref@123","alt@124","ref@124","alt@125","ref@125","alt@126","ref@126",
            "altbase@diff_eq_hdr_alt","refbase@diff_eq_hdr_ref"]
    w = csv.DictWriter(f, fieldnames=keys, delimiter="\t", lineterminator="\n", extrasaction="ignore")
    w.writeheader()
    for row in detail: w.writerow(row)
print("wrote kv_identity_detail.tsv")
