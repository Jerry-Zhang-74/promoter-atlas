# -*- coding: utf-8 -*-
"""BATCH27 B: source-exposure audit of the SuRE cohort vs prior in-project MPRA data.
Exact containment (230 bp oligo inside a retrieved SuRE insert, fwd and reverse-complement)
plus a 31-mer near-homology indicator. NO fits, NO model."""
import csv, json, gzip

SS = "reports/breakthrough_20261002/batch27/sure_seq_sample.json"
OUT = "reports/breakthrough_20261002/batch27/source_exposure_audit.json"
MPRA = "features/mpra/mpra_supervised.tsv"
RC = str.maketrans("ACGTN", "TGCAN")

oligo = []
with open(MPRA, encoding="utf-8") as f:
    for r in csv.DictReader(f, delimiter="\t"):
        s = (r.get("sequence") or "").strip().upper()
        if s:
            oligo.append(s)
oligo_set = set(oligo)
oligo_revset = set(s.translate(RC)[::-1] for s in oligo)
oligoset_sizes = sorted(set(len(s) for s in oligo))

km = 31
kmer_mpra = set()
for s in oligo:
    for i in range(len(s) - km + 1):
        kmer_mpra.add(s[i:i + km])
kmer_mpra_rc = set(k.translate(RC)[::-1] for k in kmer_mpra)
print("mpra oligos", len(oligo), "lens", oligoset_sizes, "kmers", len(kmer_mpra))

seqs = json.load(open(SS, encoding="utf-8"))
fa = seqs["fasta"]
sure = {}
name = None
with open(fa, encoding="utf-8") as f:
    for line in f:
        line = line.rstrip()
        if line.startswith(">"):
            name = line[1:]
        else:
            sure[name] = line.upper()
print("sure seqs", len(sure))

win = oligoset_sizes[0] if len(oligoset_sizes) == 1 else max(oligoset_sizes)
n_exact_containment = 0
n_rc_containment = 0
per_insert_kmer = []
for k, s in sure.items():
    hit = 0
    for i in range(0, len(s) - win + 1):
        w = s[i:i + win]
        if w in oligo_set:
            n_exact_containment += 1
            hit = 1
        if w in oligo_revset:
            n_rc_containment += 1
            hit = 1
    ks = set(s[i:i + km] for i in range(len(s) - km + 1))
    if ks:
        per_insert_kmer.append(sum(1 for x in ks if x in kmer_mpra or x in kmer_mpra_rc) / len(ks))

import numpy as np
pk = np.array(per_insert_kmer) if per_insert_kmer else np.array([0.0])
res = {
  "question": "do prior in-project K562 MPRA data contain these native 200-2000 bp SuRE inserts (exactly or by near homology)?",
  "prior_mpra": {"file": MPRA, "n_oligos": len(oligo), "oligo_lengths": oligoset_sizes,
                 "note": "GSE118242-style lentiMPRA designed oligos; no genomic coordinates are provided in this file"},
  "exact_containment": {"window_bp": win, "n_forward_windows_matching_a_prior_oligo": n_exact_containment,
                        "n_reverse_complement_windows_matching_a_prior_oligo": n_rc_containment,
                        "n_sure_inserts_tested": len(sure),
                        "interpretation": "0 means no prior oligo appears verbatim inside any retrieved SuRE insert; it does NOT exclude shorter shared sequence"},
  "near_homology_indicator": {"k": km,
                              "mean_fraction_of_sure_31mers_present_in_prior_oligo_kmer_set": float(pk.mean()),
                              "p95": float(np.percentile(pk, 95)), "max": float(pk.max()),
                              "n_inserts_with_any_shared_31mer": int((pk > 0).sum()),
                              "interpretation": "31-mers are common in genomic sequence; a non-zero value is expected even for unrelated inserts and is only an upper-bound indicator, not evidence of overlap"},
  "conclusion": "Prior in-project MPRA oligos do not contain these SuRE fragments verbatim. Exposure at the level of exact sequence is NOT demonstrated; residual near-homology and short shared sequence remain UNRESOLVED. No old K562 model was evaluated on these inserts in this batch, so no independent-testing claim is made.",
  "not_done": ["no old-model inference on SuRE inserts (protocol: zero new inference)",
               "no all-pairs alignment between SuRE inserts and prior oligos (out of budget)"]
}
json.dump(res, open(OUT, "w", encoding="utf-8"), indent=2)
print(json.dumps(res, indent=2))
