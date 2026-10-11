"""Batch36 Part C: minimal cohort FASTA + manifests + readiness + status.

ZERO fits / scoring / GPU. Consumes the frozen cohorts produced by freeze_cohort.py.
"""
from __future__ import annotations
import csv, hashlib, json, os, collections, datetime

ROOT = "E:/promoter_atlas"
OUTD = os.path.join(ROOT, "data/breakthrough_20261002/batch36/mapt")
REPD = os.path.join(ROOT, "reports/breakthrough_20261002/batch36")
os.makedirs(REPD, exist_ok=True)
ARMS = ["SAT_HEK293FT", "SAT_NEURON", "KV_SNV_HEK293FT", "KV_INDEL_HEK293FT",
        "KV_SNV_NEURON", "KV_INDEL_NEURON"]


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


# ---- FASTA of verified variable inserts (15+15 constant flanks stripped) ----
fasta_files = {}
for arm in ARMS:
    src = os.path.join(OUTD, "cohort_" + arm + ".tsv")
    dst = os.path.join(OUTD, "cohort_" + arm + "_inserts.fasta")
    n = 0
    with open(src, encoding="utf-8") as f, open(dst, "w", encoding="utf-8", newline="\n") as g:
        for r in csv.DictReader(f, delimiter="\t"):
            hdr = "|".join([arm, r["cell"], r["original_id"], "pos=" + r["pos"],
                            "ref=" + (r["base_ref"] or "NA"), "alt=" + (r["base_alt"] or "NA"),
                            "del=" + r["deletion"], "logFC=" + r["logFC"], "adjP=" + r["adjP_Value"]])
            g.write(">" + hdr + "\n" + r["variable_insert"] + "\n")
            n += 1
    fasta_files[arm] = {"file": os.path.relpath(dst, ROOT), "n": n, "sha256": sha256(dst)}
    print("fasta", arm, n)

# ---- source / label / sequence manifest (excludes mutable/self) ----
xlsx = os.path.join(OUTD, "media-1.xlsx")
dfa = os.path.join(ROOT, "data/breakthrough_20261002/batch35/mapt/gsm9640449_VariantMPRAdesignFile.fa.gz")
manifest = {
    "batch": "batch36", "version": "v6.25.0-293ft-label-recovery.rc1", "generated_at": now(),
    "study": {"pmid": "41959502", "pmcid": "PMC13060879", "doi": "10.64898/2026.03.06.710116",
              "preprint": "bioRxiv v2 (2026-03-25 posting path); PREPRINT status",
              "title": "Promoter mutagenesis and a massively parallel reporter screen of the MAPT locus"},
    "sources": [
        {"role": "MEASURED_LABELS", "file": os.path.relpath(xlsx, ROOT), "bytes": os.path.getsize(xlsx),
         "sha256": sha256(xlsx), "mimetype": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
         "url": "https://www.biorxiv.org/content/biorxiv/early/2026/03/25/2026.03.06.710116/DC1/embed/media-1.xlsx",
         "sheets_used": ["s10_InDels_Neurons", "s11_InDels_HEKs", "s12_SNVs_Neurons",
                         "s13_SNVs_HEKs", "s14_MAPTprom_Neurons", "s15_MAPTprom_HEKs"],
         "computed_score_sheets_NOT_ground_truth": ["s17_PromoterAI"]},
        {"role": "MEASURED_SEQUENCES", "file": "data/breakthrough_20261002/batch35/mapt/gsm9640449_VariantMPRAdesignFile.fa.gz",
         "bytes": os.path.getsize(dfa), "sha256": sha256(dfa),
         "url": "https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSM9640449",
         "geo": "GSE325670 / GSM9640449", "n_records": 36000, "read_only": True},
    ],
    "not_ground_truth": {"s17_PromoterAI": "computed variant-effect scores (PromoterAI), NOT measured",
                         "gsm9640449_barcodeDictionary": "maps barcode<->oligo only; NOT counts/activity (not downloaded)"},
    "license": "Public NIH preprint supplement + public GEO; local academic research use.",
}

# ---- arm readiness table ----
freeze = json.load(open(os.path.join(REPD, "cohort_freeze_audit.json"), encoding="utf-8"))
dup = freeze["duplicate_rc_audit"]
armstat = {}
for arm in ARMS:
    p = os.path.join(OUTD, "cohort_" + arm + ".tsv")
    rows = list(csv.DictReader(open(p, encoding="utf-8"), delimiter="\t"))
    adj = [float(r["adjP_Value"]) for r in rows if r["adjP_Value"] not in ("", "NA")]
    npos = len({(r["chr"], r["pos"]) for r in rows})
    bychr = collections.defaultdict(list)
    for c, po in {(r["chr"], r["pos"]) for r in rows}:
        bychr[c].append(int(po))
    nclust = 0
    for c, ps in bychr.items():
        ps.sort(); nclust += 1
        for i in range(1, len(ps)):
            if ps[i] - ps[i - 1] > 2000:
                nclust += 1
    armstat[arm] = {"n_variants": len(rows), "n_significant_adjp_le_0.05": sum(1 for a in adj if a <= 0.05),
                    "n_distinct_positions": npos, "n_2kb_region_clusters": nclust,
                    "n_chroms": len(bychr), "cell": rows[0]["cell"],
                    "label": "bcalm logFC (variant vs reference control)"}

readiness = []
for arm in ARMS:
    s = armstat[arm]
    single = s["n_2kb_region_clusters"] == 1
    verdict = "READY_WITHIN_LOCUS" if single else "READY_MULTIREGION"
    readiness.append({
        "arm": arm, "cell": s["cell"], "n_variants": s["n_variants"],
        "n_significant_adjp_le_0.05": s["n_significant_adjp_le_0.05"],
        "n_distinct_positions": s["n_distinct_positions"], "n_2kb_region_clusters": s["n_2kb_region_clusters"],
        "split_possible": ("NO single 2kb locus" if single else "proposed parent-region group split"),
        "verdict": verdict,
        "domain": "regulatory VARIANT effect (CRE reporter), NOT autonomous promoter activity",
        "labels_recovered": True, "sequences_recovered": True})

with open(os.path.join(REPD, "arm_readiness.tsv"), "w", encoding="utf-8", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(readiness[0].keys()), delimiter="\t", lineterminator="\n")
    w.writeheader()
    for r in readiness:
        w.writerow(r)
json.dump({"generated_at": now(), "arms": armstat, "readiness": readiness},
          open(os.path.join(REPD, "label_source_manifest.json"), "w", encoding="utf-8"), indent=2)
json.dump(manifest, open(os.path.join(REPD, "source_manifest.json"), "w", encoding="utf-8"), indent=2)
print("readiness rows", len(readiness))
print(json.dumps(armstat, indent=1))

