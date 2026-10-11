"""Extract s10-s15 (+ s17 computed scores) from media-1.xlsx to TSV.

Dependency-free. Writes to data/breakthrough_20261002/batch36/mapt/supp/.
"""
import sys, os, zipfile, hashlib, json, csv
sys.path.insert(0, r"E:/promoter_atlas/code/batch36")
import xlsx_min as X

XL = r"E:/promoter_atlas/data/breakthrough_20261002/batch36/mapt/media-1.xlsx"
OUT = r"E:/promoter_atlas/data/breakthrough_20261002/batch36/mapt/supp"
os.makedirs(OUT, exist_ok=True)

SHEETS = {
    "s10_InDels_Neurons": "xl/worksheets/sheet11.xml",
    "s11_InDels_HEKs":    "xl/worksheets/sheet12.xml",
    "s12_SNVs_Neurons":   "xl/worksheets/sheet13.xml",
    "s13_SNVs_HEKs":      "xl/worksheets/sheet14.xml",
    "s14_MAPTprom_Neurons": "xl/worksheets/sheet15.xml",
    "s15_MAPTprom_HEKs":  "xl/worksheets/sheet16.xml",
    "s17_PromoterAI":     "xl/worksheets/sheet18.xml",
}

def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()

z = zipfile.ZipFile(XL)
shared = X.read_shared_strings(z)
summary = {}
for name, part in SHEETS.items():
    out = os.path.join(OUT, name + ".tsv")
    n = 0
    with open(out, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, delimiter="\t", lineterminator="\n")
        for row in X.iter_rows(z, part, shared):
            w.writerow(row)
            n += 1
    summary[name] = {"tsv": out, "n_rows_incl_header": n,
                     "sha256": sha256(out), "size": os.path.getsize(out),
                     "worksheet_part": part}
    print(name, n, os.path.getsize(out))
with open(os.path.join(OUT, "_extract_summary.json"), "w", encoding="utf-8") as f:
    json.dump({"source_xlsx": XL, "source_sha256": sha256(XL),
               "extracted_at_utc": __import__("datetime").datetime.utcnow().isoformat()+"Z",
               "sheets": summary}, f, indent=2)
print("xlsx sha256", sha256(XL))
