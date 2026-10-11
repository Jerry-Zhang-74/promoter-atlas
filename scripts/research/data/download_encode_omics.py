#!/usr/bin/env python3
"""Download curated ENCODE omics files: histone marks + DNase + ATAC narrowPeaks (K562/GM12878/HepG2), PRO-cap peaks.

Output: data/omics/raw/{histone,dnase,atac,procap}/<file>
"""
import urllib.request, urllib.parse, ssl, json, subprocess
from pathlib import Path

ROOT = Path(r"E:\promoter_atlas")
RAW = ROOT / "data" / "omics" / "raw"
ctx = ssl.create_default_context()
proxy = urllib.request.ProxyHandler({"https": "http://127.0.0.1:7897", "http": "http://127.0.0.1:7897"})
op = urllib.request.build_opener(proxy)
def get_json(url, timeout=120):
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0", "Accept": "application/json"})
    with op.open(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8", "ignore"))

def curl_dl(url, out, retries=3):
    if out.exists():
        print("  (exists)", out.name, flush=True)
        return True
    for attempt in range(retries):
        rc = subprocess.run(
            ["curl", "-L", "--retry", "5", "--retry-delay", "10", "-C", "-",
             "-x", "http://127.0.0.1:7897", "-sS", "-o", str(out), url],
            capture_output=True, text=True, timeout=None)
        if rc.returncode == 0 and out.exists() and out.stat().st_size > 0:
            print("  DONE", out.name, out.stat().st_size, flush=True)
            return True
        print(f"  retry {attempt+1} rc={rc.returncode} {rc.stderr.strip()[:150]}", flush=True)
    return False

def pick_files(assay, cell, target=None, limit=2):
    params = urllib.parse.urlencode({
        "type": "File", "status": "released", "assay_title": assay,
        "biosample_ontology.term_name": cell, "file_format": "bed",
        "file_format_type": "narrowPeak", "output_type": "peaks",
        "format": "json", "limit": 20,
    })
    q = "https://www.encodeproject.org/search/?" + params
    if target:
        q += "&target.label=" + urllib.parse.quote(target)
    d = get_json(q)
    files = [f for f in d.get("@graph", []) if f.get("file_format") == "bed"]
    def sort_key(f):
        pref = 0 if f.get("preferred_default") is True else 1
        dt = f.get("date_released") or ""
        return (pref, dt)
    files.sort(key=sort_key, reverse=False)
    # prefer preferred_default first
    files.sort(key=lambda f: 0 if f.get("preferred_default") is True else 1)
    return files[:limit]

plan = []
# histones
for cell in ["K562", "GM12878", "HepG2"]:
    for mark in ["H3K4me3", "H3K27ac", "H3K27me3", "H3K4me1", "H3K36me3"]:
        try:
            for f in pick_files("Histone ChIP-seq", cell, mark, limit=2):
                plan.append((f["accession"], f"histone/{cell}_{mark}_{f['accession']}.bed.gz",
                             f"https://www.encodeproject.org/files/{f['accession']}/@@download/{f['accession']}.bed.gz"))
        except Exception as e:
            print("ERR", cell, mark, repr(e)[:120], flush=True)
# dnase + atac
for cell in ["K562", "GM12878", "HepG2"]:
    for assay, folder in [("DNase-seq", "dnase"), ("ATAC-seq", "atac")]:
        try:
            for f in pick_files(assay, cell, limit=2):
                plan.append((f["accession"], f"{folder}/{cell}_{f['accession']}.bed.gz",
                             f"https://www.encodeproject.org/files/{f['accession']}/@@download/{f['accession']}.bed.gz"))
        except Exception as e:
            print("ERR", cell, assay, repr(e)[:120], flush=True)
# pro-cap bidirectional peaks
for acc, label in [("ENCFF106YWP", "K562_ENCSR756TPG"), ("ENCFF227JQX", "K562_ENCSR604EBH")]:
    plan.append((acc, f"procap/{label}_{acc}.bed.gz",
                 f"https://www.encodeproject.org/files/{acc}/@@download/{acc}.bed.gz"))

print("plan:", len(plan), flush=True)
for acc, name, url in plan:
    out = RAW / name
    out.parent.mkdir(parents=True, exist_ok=True)
    print("DL", name, flush=True)
    curl_dl(url, out)
print("DONE", flush=True)
