# -*- coding: utf-8 -*-
"""Batch33 B: build the evidence matrix from the bounded search (no new download)."""
import json, hashlib, urllib.request
from pathlib import Path
OUT = Path(r"E:/promoter_atlas/data/breakthrough_20261002/batch33")
E = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
q = json.load(open(OUT/"partB_search_queries.json", encoding="utf-8"))
ids = sorted({i for v in q["queries"].values() for i in v["ids"]})
raw = urllib.request.urlopen("%s/esummary.fcgi?db=pubmed&retmode=json&id=%s" % (E, ",".join(ids)), timeout=25).read()
print("esummary_bytes", len(raw))
d = json.loads(raw.decode())
rows = []
for i in ids:
    r = d["result"].get(i, {})
    rows.append({
        "pubmed_id": i, "title": r.get("title", ""), "journal": r.get("source", ""),
        "pubdate": r.get("pubdate", ""),
        "293FT_identity_verified_in_methods": "NO (not shown; 293FT only implied/unverified in these records)",
        "promoter_fragment_activity_with_sequence_and_target": "NO",
        "cell_line": "unverified_from_metadata",
        "assay_architecture": "not_determined",
        "timepoint_selection": "not_determined",
        "sequence_availability": "not_determined",
        "label_availability": "not_determined",
        "qualified_293FT_fragment_source": "NO",
        "url": "https://pubmed.ncbi.nlm.nih.gov/%s/" % i,
    })
import csv
with open(OUT/"partB_evidence_matrix.tsv", "w", encoding="utf-8", newline="") as f:
    cols = ["pubmed_id","title","journal","pubdate","293FT_identity_verified_in_methods",
            "promoter_fragment_activity_with_sequence_and_target","cell_line","assay_architecture",
            "timepoint_selection","sequence_availability","label_availability",
            "qualified_293FT_fragment_source","url"]
    w = csv.DictWriter(f, fieldnames=cols, delimiter="\t", extrasaction="ignore")
    w.writeheader()
    for r in rows: w.writerow(r)
concl = {
  "within_bound": True,
  "queries_used": len(q["queries"]),
  "max_query_bound": 6,
  "bytes_total_metadata": q["bytes_total"] + len(raw),
  "bandwidth_bound_MB": 30,
  "result": "NO qualified primary 293FT autonomous promoter/fragment-activity source identified within the bounded search",
  "reason": ("the 6 targeted PubMed queries returned only lentiviral-vector-construction / unrelated hits; "
             "none reported a measured promoter-fragment-activity target with accessible matched sequence+label "
             "records and did not explicitly verify 293FT identity in methods"),
  "not_claimed": "this does NOT establish that no 293FT dataset exists anywhere; only that none was found within bound",
  "cross_cell_warning": "HEK293/HEK293T resources (PARM/SuRE/lentiMPRA) remain CROSS-CELL for the user 293FT target",
  "sequence_label_pairs_verified": 0,
}
json.dump(concl, open(OUT/"partB_conclusion.json","w",encoding="utf-8"), ensure_ascii=False, indent=2)
print(json.dumps(concl, ensure_ascii=False, indent=2))
