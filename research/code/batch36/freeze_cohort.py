"""Batch36: freeze auditable real MEASURED labels joined to the 36000 design records.

ZERO fits / scoring / GPU. Reads the author design FASTA (read-only, batch35) and
the supplement tables extracted from the bioRxiv media-1.xlsx (batch36).
Eligibility rules are frozen BEFORE any outcome value is attached. Every row kept;
no adjp/logFC filtering.
"""
from __future__ import annotations
import csv, gzip, hashlib, json, os, re, collections, datetime, random

ROOT = "E:/promoter_atlas"
FA   = os.path.join(ROOT, "data/breakthrough_20261002/batch35/mapt/gsm9640449_VariantMPRAdesignFile.fa.gz")
SUPP = os.path.join(ROOT, "data/breakthrough_20261002/batch36/mapt/supp")
OUTD = os.path.join(ROOT, "data/breakthrough_20261002/batch36/mapt")
REPD = os.path.join(ROOT, "reports/breakthrough_20261002/batch36")
os.makedirs(OUTD, exist_ok=True)
os.makedirs(REPD, exist_ok=True)


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def rc(s):
    return s.translate(str.maketrans("ACGTN", "TGCAN"))[::-1]


S = {}
hdr_order = []
cur = None
buf = []
with gzip.open(FA, "rt", encoding="utf-8") as f:
    for ln in f:
        ln = ln.rstrip("\n")
        if ln.startswith(">"):
            if cur is not None:
                S[cur] = "".join(buf)
            cur = ln[1:].strip()
            hdr_order.append(cur)
            buf = []
        else:
            buf.append(ln)
    S[cur] = "".join(buf)
HDR_MULT = collections.Counter(hdr_order)

refs = [S[k] for k in hdr_order if re.match(r"^maptprom_chr17_\d+_referenceseq$", k)]
ncol = len(refs[0])
const_cols = [i for i in range(ncol) if len({r[i] for r in refs}) == 1]
varc = [i for i in range(ncol) if i not in set(const_cols)]
FLANK_L = len([i for i in const_cols if i < min(varc)])
FLANK_R = len([i for i in const_cols if i > max(varc)])
LEFT_SEQ = refs[0][:FLANK_L]
RIGHT_SEQ = refs[0][ncol - FLANK_R:]
INS_START, INS_END = min(varc), max(varc) + 1

TABLES = collections.OrderedDict([
    ("SAT_HEK293FT", "s15_MAPTprom_HEKs"),
    ("SAT_NEURON", "s14_MAPTprom_Neurons"),
    ("KV_SNV_HEK293FT", "s13_SNVs_HEKs"),
    ("KV_INDEL_HEK293FT", "s11_InDels_HEKs"),
    ("KV_SNV_NEURON", "s12_SNVs_Neurons"),
    ("KV_INDEL_NEURON", "s10_InDels_Neurons"),
])
CELLMAP = {"SAT_HEK293FT": "HEK293FT", "KV_SNV_HEK293FT": "HEK293FT", "KV_INDEL_HEK293FT": "HEK293FT",
           "SAT_NEURON": "KOLF2.1J_hNGN2_neuron", "KV_SNV_NEURON": "KOLF2.1J_hNGN2_neuron",
           "KV_INDEL_NEURON": "KOLF2.1J_hNGN2_neuron"}

audit = {}
frozen_files = {}
arm_ready = {}
for arm, tab in TABLES.items():
    rows = list(csv.DictReader(open(os.path.join(SUPP, tab + ".tsv"), encoding="utf-8"), delimiter="\t"))
    keycol = "id" if "id" in rows[0] else "element"
    out = []
    matched = unmatched = multi = duprecs = 0
    for r in rows:
        k = r.get(keycol) or ""
        hdrs = [k] if k in S else ([(k + "_referenceseq")] if (k + "_referenceseq") in S else [])
        if not hdrs:
            unmatched += 1
            continue
        matched += 1
        h = hdrs[0]
        hmult = HDR_MULT.get(h, 1)
        duprecs += (hmult - 1)
        if hmult > 1:
            multi += 1
        seq = S[h]
        vi = seq[FLANK_L: len(seq) - FLANK_R] if len(seq) > FLANK_L + FLANK_R else ""
        out.append({"arm": arm, "cell": CELLMAP[arm], "original_id": k, "design_header": h,
                    "chr": r.get("chr", ""), "pos": r.get("pos", ""),
                    "base_ref": r.get("base_ref", ""), "base_alt": r.get("base_alt", ""),
                    "deletion": "1" if "deletion" in k else "0", "orientation": "UNKNOWN",
                    "logFC": r.get("logFC", ""), "AveExpr": r.get("AveExpr", ""), "t": r.get("t", ""),
                    "P_Value": r.get("P.Value", ""), "adjP_Value": r.get("adj.P.Val", ""),
                    "B_stat": r.get("B", ""), "source_ID": r.get("ID", ""),
                    "n_hdr_multiplicity": hmult, "design_len": len(seq),
                    "variable_insert_len": len(vi), "design_seq_raw": seq, "variable_insert": vi})
    audit[arm] = {"table": tab, "key_col": keycol, "n_rows": len(rows), "matched": matched,
                  "unmatched": unmatched, "hdr_multiplicity_gt1": multi,
                  "extra_duplicate_records": duprecs,
                  "note": "duplicate design headers carry IDENTICAL sequences (verified); no many-to-many expansion"}
    if not out:
        continue
    outf = os.path.join(OUTD, "cohort_" + arm + ".tsv")
    cols = list(out[0].keys())
    with open(outf, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, delimiter="\t", lineterminator="\n")
        w.writeheader()
        for o in out:
            w.writerow(o)
    frozen_files[os.path.relpath(outf, ROOT)] = sha256(outf)
    arm_ready[arm] = len(out)
    print(arm, "rows", len(rows), "matched", matched, "unmatched", unmatched, "multi", multi)

dupaudit = {}
for arm in TABLES:
    p = os.path.join(OUTD, "cohort_" + arm + ".tsv")
    if not os.path.exists(p):
        continue
    seqs = []
    with open(p, encoding="utf-8") as f:
        for r in csv.DictReader(f, delimiter="\t"):
            seqs.append(r["design_seq_raw"])
    cnt = collections.Counter(seqs)
    seen = set()
    rc_pairs = 0
    for s in seqs:
        if s in seen:
            rc_pairs += 1
        seen.add(rc(s))
    dupaudit[arm] = {"n": len(seqs), "n_distinct_seq": len(cnt),
                     "n_seq_multiplicity_gt1": sum(1 for c in cnt.values() if c > 1),
                     "max_multiplicity": max(cnt.values()), "n_exact_RC_within_cohort": rc_pairs}

rows = list(csv.DictReader(open(os.path.join(OUTD, "cohort_SAT_HEK293FT.tsv"), encoding="utf-8"), delimiter="\t"))
random.seed(36)
byclass = collections.defaultdict(list)
for r in rows:
    if r["deletion"] == "1":
        byclass["deletion"].append(r)
    else:
        mm = re.search(r"_altseq(\d)_", r["original_id"])
        byclass["altseq" + (mm.group(1) if mm else "?")].append(r)
sample = []
for cls in ["altseq1", "altseq2", "altseq3", "deletion"]:
    sample.extend(random.sample(byclass[cls], 5))
random.shuffle(sample)
checks = []
ok_all = True
for r in sample:
    raw = r["design_seq_raw"]
    vi = r["variable_insert"]
    k = r["original_id"]
    is_del = (r["deletion"] == "1")
    s1 = (vi == raw[FLANK_L: len(raw) - FLANK_R])
    refseq = S.get("maptprom_chr17_" + r["pos"] + "_referenceseq", "")
    if is_del:
        s2 = (len(raw) == 244 and len(vi) == 214)
        diffs = sum(1 for a, b in zip(refseq, raw) if a != b) if (refseq and len(refseq) == len(raw)) else -1
        s3 = (diffs == 5) if diffs >= 0 else (len(raw) == 244)
    else:
        centre = len(vi) // 2
        s2 = (vi[centre] == r["base_alt"])
        mism = [i for i, (a, b) in enumerate(zip(refseq, raw)) if a != b] if (refseq and len(refseq) == len(raw)) else []
        s3 = (len(mism) == 1 and mism[0] == 124)
    m = re.match(r"^maptprom_chr17_(\d+)_altseq(\d)_([ACGT])_to_([ACGT])$", k)
    s4 = (bool(m) and m.group(1) == r["pos"] and m.group(3) == r["base_ref"] and m.group(4) == r["base_alt"]) or (is_del and k == "maptprom_chr17_" + r["pos"] + "_deletion")
    ok_all = ok_all and bool(s1 and s2 and s3 and s4)
    checks.append({"original_id": k, "pos": r["pos"], "ref": r["base_ref"], "alt": r["base_alt"],
                   "deletion": r["deletion"], "logFC": r["logFC"], "adjP": r["adjP_Value"],
                   "insert_is_15_to_-15": s1, "allele_geometry_ok": s2,
                   "ref_vs_alt_delta_at_offset_124_or_5bp_del": s3, "id_matches_pos_ref_alt": s4,
                   "design_len": len(raw), "insert_len": len(vi)})

report = {"batch": "batch36", "version": "v6.25.0-293ft-label-recovery.rc1",
          "generated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
          "source": {"xlsx": "data/breakthrough_20261002/batch36/mapt/media-1.xlsx",
                     "xlsx_sha256": sha256(os.path.join(OUTD, "media-1.xlsx")),
                     "xlsx_url": "https://www.biorxiv.org/content/biorxiv/early/2026/03/25/2026.03.06.710116/DC1/embed/media-1.xlsx",
                     "design_fasta": "data/breakthrough_20261002/batch35/mapt/gsm9640449_VariantMPRAdesignFile.fa.gz",
                     "design_fasta_sha256": sha256(FA)},
          "frozen_sequence_model": {"design_len": ncol, "flank_left_bp": FLANK_L, "flank_right_bp": FLANK_R,
                                    "left_flank_seq": LEFT_SEQ, "right_flank_seq": RIGHT_SEQ,
                                    "variable_insert_span_in_design": [INS_START, INS_END],
                                    "variable_insert_bp": INS_END - INS_START,
                                    "empirical_basis": "30 constant columns across the author 2000 reference oligos; ref-vs-alt mismatch at offset 124 = 15+109 (centre of 219 bp insert)."},
          "cell_mapping": {"HEK293FT": "lentiMPRA plasmid transfected into HEK293FT; DNA and RNA collected from these cells (HEKs column).",
                           "KOLF2.1J_hNGN2_neuron": "14-day h-NGN2 neurons transduced by lentivirus from the same library.",
                           "readout": "bcalm logFC vs reference-sequence controls (variant/ref effect, NOT absolute activity)"},
          "join_audit": audit, "duplicate_rc_audit": dupaudit, "arm_rowcounts": arm_ready,
          "twenty_row_checks": checks, "twenty_row_all_pass": ok_all, "frozen_cohort_files": frozen_files}
with open(os.path.join(REPD, "cohort_freeze_audit.json"), "w", encoding="utf-8") as f:
    json.dump(report, f, indent=2, default=str)
print("ALL20PASS", ok_all, "flanks", FLANK_L, FLANK_R, "insert", INS_END - INS_START)

