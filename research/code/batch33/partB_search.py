# -*- coding: utf-8 -*-
"""Batch33 B: ONE bounded primary-source evidence search (<=6 queries, <=30MB,
<=20 min) for actual-293FT autonomous promoter/fragment activity with real
sequence + measured target. Public scientific sources only (NCBI E-utilities).
Records CH-2312evidence; counts metadata records honestly; never counts
figure dots or metadata rows as a full training cohort."""
import json, time, urllib.parse, urllib.request
from pathlib import Path
OUT = Path(r"E:/promoter_atlas/data/breakthrough_20261002/batch33")
OUT.mkdir(parents=True, exist_ok=True)
E = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
def get(url):
    for _ in range(3):
        try:
            with urllib.request.urlopen(url, timeout=20) as r:
                return r.read()
        except Exception as e:
            time.sleep(2)
    return b""
QUERIES = {
 "Q1_promoter_fragment_library": '293FT AND (promoter OR enhancer) AND (fragment OR library) AND (activity OR expression) AND (reporter OR assay)',
 "Q2_MPRA": '293FT AND ("massively parallel reporter assay" OR MPRA OR "massively parallel" OR "lentiviral library")',
 "Q3_random_sequence": '293FT AND ("random sequence" OR "random library" OR "synthetic library") AND (promoter OR enhancer)',
 "Q4_STARR": '293FT AND ("STARR-seq" OR "self-transcribing active regulatory region" OR "capSTARR")',
 "Q5_episomal_integration": '293FT AND (episomal OR integrated OR lentiviral) AND promoter AND (activity OR strength) AND (library OR fragment)',
 "Q6_quantitative_fragment_activity": '293FT AND (promoter OR enhancer) AND (fluorescence OR reporter) AND (truncation OR tiling OR fragment)',
}
res = {}
bytes_total = 0
for name, term in QUERIES.items():
    u = "%s/esearch.fcgi?db=pubmed&retmode=json&retmax=8&sort=relevance&term=%s" % (E, urllib.parse.quote(term))
    raw = get(u); bytes_total += len(raw)
    try:
        d = json.loads(raw.decode("utf-8"))
        cnt = int(d["esearchresult"]["count"]); ids = d["esearchresult"]["idlist"]
    except Exception as e:
        cnt, ids = None, []
    res[name] = {"term": term, "count": cnt, "ids": ids}
    print("%s count=%s ids=%s" % (name, cnt, ids))
    time.sleep(0.4)
json.dump({"queries": res, "bytes_total": bytes_total,
           "note": "metadata/count-level search only; a hit count is NOT a training cohort"},
          open(OUT/"partB_search_queries.json", "w", encoding="utf-8"), ensure_ascii=False, indent=2)
print("bytes_total", bytes_total)
