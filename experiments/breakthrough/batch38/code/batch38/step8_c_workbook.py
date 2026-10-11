# -*- coding: utf-8 -*-
"""Batch38 C: audit workbook s2-s5 (BAC/Oligo arms) labels; extract headers + sample rows."""
import sys, os, zipfile, csv, hashlib, re
sys.path.insert(0, "code/batch36")
import xlsx_min as X
XL = "data/breakthrough_20261002/batch36/mapt/media-1.xlsx"
OUT = "data/breakthrough_20261002/batch38"
z = zipfile.ZipFile(XL)
# sheet name -> part via rels
rels = z.read("xl/_rels/workbook.xml.rels").decode("utf-8")
relmap = dict(re.findall(r'Id="([^"]+)"[^>]*Target="([^"]+)"', rels))
wb = z.read("xl/workbook.xml").decode("utf-8")
sheets = re.findall(r'<sheet[^>]*name="([^"]+)"[^>]*r:id="([^"]+)"', wb)
name2part = {nm: "xl/" + relmap[rid].lstrip("/").replace("xl/", "") if not relmap[rid].startswith("xl/") else relmap[rid] for nm, rid in sheets}
shared = X.read_shared_strings(z)
targets = {"s2": "s2 - BAC MPRA Neurons", "s3": "s3 - BAC MPRA HEKs", "s4": "s4 - OligoMPRA Neurons", "s5": "s5 - OligoMPRA Heks"}
summary = {}
for tag, nm in targets.items():
    part = name2part[nm]
    rows = []
    for row in X.iter_rows(z, part, shared):
        rows.append(row)
        if len(rows) >= 8: break
    # full header + count
    alln = 0
    for _ in X.iter_rows(z, part, shared): alln += 1
    hdr = rows[0] if rows else []
    summary[tag] = {"sheet": nm, "part": part, "n_rows_incl_header": alln, "header": hdr,
                    "sample_rows": rows[1:6]}
    print("=====", tag, nm, "rows", alln)
    print("HEADER:", hdr)
    for r in rows[1:4]: print("  row:", r)
# write csv copies of s2-s5 headers and first rows for evidence
import json
json.dump(summary, open(os.path.join(OUT, "workbook_s2_s5_summary.json"), "w"), indent=1, ensure_ascii=False)
def sha(p):
    h = hashlib.sha256(); h.update(open(p, "rb").read()); return h.hexdigest()
print("xlsx sha", sha(XL))
