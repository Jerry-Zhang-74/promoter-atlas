#!/usr/bin/env python3
"""Download ENCODE ATAC-seq narrowPeaks for K562/GM12878/HepG2 (preferred_default)."""
import urllib.request, urllib.parse, ssl, json, subprocess
from pathlib import Path
ROOT = Path(r"E:\promoter_atlas")
RAW = ROOT / "data" / "omics" / "raw" / "atac"
RAW.mkdir(parents=True, exist_ok=True)
ctx = ssl.create_default_context()
proxy = urllib.request.ProxyHandler({"https": "http://127.0.0.1:7897", "http": "http://127.0.0.1:7897"})
op = urllib.request.build_opener(proxy)
def get_json(url, timeout=120):
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0", "Accept": "application/json"})
    with op.open(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8", "ignore"))

exp_map = {"K562": ["ENCSR956DNB", "ENCSR859USB"], "GM12878": ["ENCSR095QNB", "ENCSR637XSC"], "HepG2": ["ENCSR042AWH", "ENCSR291GJU"]}
plan = []
for cell, accs in exp_map.items():
    for acc in accs:
        try:
            d = get_json(f"https://www.encodeproject.org/experiments/{acc}/")
            files = [f for f in d.get("files", []) if f.get("status") == "released"
                     and f.get("file_format") == "bed" and f.get("output_type") in ("peaks", "replicated peaks")]
            files.sort(key=lambda f: 0 if f.get("preferred_default") is True else 1)
            if files:
                f = files[0]
                plan.append((f"{cell}_{f['accession']}.bed.gz",
                             f"https://www.encodeproject.org/files/{f['accession']}/@@download/{f['accession']}.bed.gz"))
        except Exception as e:
            print(cell, acc, "ERR", repr(e)[:100], flush=True)
print("plan:", plan, flush=True)
for name, url in plan:
    out = RAW / name
    if out.exists():
        print("(exists)", name, flush=True)
        continue
    rc = subprocess.run(["curl", "-L", "--retry", "5", "-C", "-", "-x", "http://127.0.0.1:7897",
                         "-sS", "-o", str(out), url], capture_output=True, text=True)
    print("DL", name, rc.returncode, out.stat().st_size if out.exists() else 0, flush=True)
print("DONE", flush=True)
