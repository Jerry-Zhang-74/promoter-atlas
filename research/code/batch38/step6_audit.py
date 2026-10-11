# -*- coding: utf-8 -*-
"""Batch38 A: frozen-source identity & contrast audit of the KV SNV arm.
Deterministic rules frozen on SEQUENCE IDENTITY ONLY (no labels, no cached scores).
Sources: author design FASTA (36000 physical records), author variant_reference_key.tsv,
author design.rmd sequence-generation code, author heks_snvs.R bcalm call chain.
ZERO fits / ZERO GPU."""
import gzip, collections, re, csv, json, hashlib, os, datetime

ROOT = "."
FA   = "data/breakthrough_20261002/batch35/mapt/gsm9640449_VariantMPRAdesignFile.fa.gz"
KV   = "data/breakthrough_20261002/batch36/mapt/cohort_KV_SNV_HEK293FT.tsv"
SAT  = "data/breakthrough_20261002/batch36/mapt/cohort_SAT_HEK293FT.tsv"
KEY  = "data/breakthrough_20261002/batch38/variant_reference_key.tsv"
OUTD = "data/breakthrough_20261002/batch38"

def sha256f(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""): h.update(b)
    return h.hexdigest()
def now(): return datetime.datetime.now(datetime.timezone.utc).astimezone().isoformat()

# ---- load ALL design records (dups preserved) ----
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
dup_with_conflict = sum(1 for h in byh if len(set(byh[h])) > 1)

# ---- author key ----
key = {}
with open(KEY, encoding="utf-8") as f:
    for r in csv.DictReader(f, delimiter="\t"):
        key[r["id"]] = r["control"]

# ---- frozen deterministic rules ----
# For alt header H: derive ref header by replacing trailing _altseq.../ _altseqN... with _referenceseq,
# then REQUIRE equality with author key control. Differences analysed.
def derive_ref(h):
    return re.sub(r'_altseq\d*_[ACGT]+_to_[ACGT]+$', '_referenceseq', h)

kv = list(csv.DictReader(open(KV, encoding="utf-8"), delimiter="\t"))
cls = collections.Counter()
key_agree = key_disagree = key_missing = 0
rows_out = []
diffpos_hist = collections.Counter()
placement = collections.Counter()
ref124_eq_reftbl = 0; alt125_eq_alttbl = 0
for r in kv:
    ah = r["design_header"]; R = r["base_ref"]; A = r["base_alt"]
    dh = derive_ref(ah)
    kc = key.get(ah)
    if kc is None: key_missing += 1
    elif kc == dh: key_agree += 1
    else: key_disagree += 1
    alt_s = byh.get(ah, [None])[0]
    ref_s = byh.get(kc or dh, [None])[0]
    o = {"original_id": r["original_id"], "alt_header": ah, "key_control": kc, "derived_ref": dh,
         "base_ref": R, "base_alt": A}
    if alt_s is None or ref_s is None:
        cls["MISSING_RECORD"] += 1; o["class"] = "MISSING_RECORD"; rows_out.append(o); continue
    diffs = [i for i in range(min(len(alt_s), len(ref_s))) if alt_s[i] != ref_s[i]]
    diffpos_hist[tuple(diffs[:3])] += 1
    o["n_diff"] = len(diffs); o["diff_pos"] = diffs[:3]
    o["alt124"] = alt_s[124]; o["alt125"] = alt_s[125]; o["ref124"] = ref_s[124]; o["ref125"] = ref_s[125]
    # placement of the ALT allele in the alt oligo
    positions_of_A_in_alt = [i for i in range(len(alt_s)) if alt_s[i] == A and (i >= len(ref_s) or ref_s[i] != A)]
    o["alt_allele_offsets"] = positions_of_A_in_alt[:4]
    ref124_eq_reftbl += (ref_s[124] == R)
    alt125_eq_alttbl += (alt_s[125] == A)
    if len(diffs) == 0:
        c = "IDENTICAL_ALT_EQ_CONTROL"
    elif diffs == [125]:
        c = "SINGLE_SUB_AT_125"
    elif diffs == [124]:
        c = "SINGLE_SUB_AT_124"
    else:
        c = "OTHER_MULTI_DIFF"
    o["class"] = c; cls[c] += 1
    placement[tuple(positions_of_A_in_alt[:1])] += 1
    rows_out.append(o)

print("KV rows", len(kv))
print("key coverage: agree", key_agree, "disagree", key_disagree, "missing", key_missing)
print("class:", dict(cls))
print("diff-pos histogram:", dict(diffpos_hist))
print("ALT-allele placement (first offset):", dict(placement))
print("ref[124]==base_ref:", ref124_eq_reftbl, "/", len(kv))
print("alt[125]==base_alt:", alt125_eq_alttbl, "/", len(kv))
print("dup headers with conflicting seq:", dup_with_conflict)

# ---- SAT mixed 20-row control check (sign/control convention from author call chain) ----
sat = list(csv.DictReader(open(SAT, encoding="utf-8"), delimiter="\t"))
import random
random.seed(38)
snv = [r for r in sat if r["deletion"] == "0"]
dele = [r for r in sat if r["deletion"] == "1"]
sample = random.sample(snv, 14) + random.sample(dele, 6)
sat_checks = []
for r in sample:
    ah = r["design_header"]; ph = "maptprom_chr17_%s_referenceseq" % r["pos"]
    a = byh.get(ah, [None])[0]; rf = byh.get(ph, [None])[0]
    s = {"original_id": r["original_id"], "deletion": r["deletion"], "base_ref": r["base_ref"], "base_alt": r["base_alt"]}
    if a and rf:
        d = [i for i in range(min(len(a), len(rf))) if a[i] != rf[i]]
        s.update({"n_diff": len(d), "diff_pos": d[:6], "alt124": a[124] if len(a) > 124 else None,
                  "ref124": rf[124] if len(rf) > 124 else None, "len_alt": len(a), "len_ref": len(rf)})
        if r["deletion"] == "1":
            s["base_is_centered_5bp_del"] = (len(rf) - len(a) == 5) and (a == rf[:122] + rf[127:])
    s["key_control"] = key.get(ah)
    sat_checks.append(s)
print("SAT sample:", json.dumps(sat_checks[:3], ensure_ascii=False))

json.dump({"class": dict(cls), "key_agree": key_agree, "key_disagree": key_disagree, "key_missing": key_missing,
           "diffpos_hist": {str(k): v for k, v in diffpos_hist.items()},
           "placement": {str(k): v for k, v in placement.items()},
           "ref124_eq_base_ref": ref124_eq_reftbl, "alt125_eq_base_alt": alt125_eq_alttbl,
           "dup_conflict": dup_with_conflict, "sat_checks": sat_checks,
           "sha": {"design_fasta": sha256f(FA), "kv_cohort": sha256f(KV), "sat_cohort": sha256f(SAT),
                   "author_key": sha256f(KEY), "generated_at": now()}},
          open(os.path.join(OUTD, "step6_summary.json"), "w"), indent=1, ensure_ascii=False)
with open(os.path.join(OUTD, "kv_classification.tsv"), "w", encoding="utf-8", newline="") as f:
    cols = ["original_id","alt_header","key_control","derived_ref","base_ref","base_alt","class","n_diff",
            "diff_pos","alt124","alt125","ref124","ref125","alt_allele_offsets"]
    w = csv.DictWriter(f, fieldnames=cols, delimiter="\t", lineterminator="\n", extrasaction="ignore")
    w.writeheader()
    for o in rows_out: w.writerow(o)
print("wrote kv_classification.tsv + step6_summary.json")
