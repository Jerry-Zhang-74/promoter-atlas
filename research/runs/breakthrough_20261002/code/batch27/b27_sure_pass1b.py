# -*- coding: utf-8 -*-
"""BATCH27 B: fast single streaming pass over the SuRE K562 per-fragment counts file.
NO estimator fits, NO model application. Label-free eligibility + sha256-ordered selection only."""
import json, time, zlib, hashlib, heapq, sys

P = "data/breakthrough_20261002/batch27/source/GSE78709_SuRE-counts-K562_B45_B55_LP170105.txt.gz"
OUT = "reports/breakthrough_20261002/batch27/sure_pass1.json"
SEL = "reports/breakthrough_20261002/batch27/sure_selected_ids.json"
CANON = set(b"chr%d" % i for i in range(1, 23)) | {b"chrX", b"chrY"}
sha = hashlib.sha256
t0 = time.time()

N = 0; parse_fail = 0
chrom_ct = {}; strand_ct = {}; lenhist = {}
n_startok = n_canon = n_lenrange = n_input10 = n_rna_any = 0
elig = 0
heap = []
SEL_CAP = 25000
d = zlib.decompressobj(31)
buf = b""
hdr = None
last_report = 0

def bump(c, k):
    c[k] = c.get(k, 0) + 1

with open(P, "rb") as f:
    while True:
        chunk = f.read(1 << 22)
        if not chunk:
            break
        data = d.decompress(chunk)
        if not data:
            continue
        buf += data
        lines = buf.split(b"\n")
        buf = lines.pop()
        for line in lines:
            if not line:
                continue
            if hdr is None:
                hdr = line
                continue
            N += 1
            c = line.split(b"\t")
            if len(c) != 13:
                parse_fail += 1
                continue
            chrom = c[0]; startb = c[1]; endb = c[2]; strand = c[3]
            bump(chrom_ct, chrom); bump(strand_ct, strand)
            try:
                st = int(startb); en = int(endb)
            except Exception:
                parse_fail += 1
                continue
            L = en - st
            if 0 < L < 4000:
                bump(lenhist, min(L // 100 * 100, 3900))
            if not (st < en):
                continue
            n_startok += 1
            if chrom not in CANON:
                continue
            n_canon += 1
            if not (200 <= L <= 2000):
                continue
            n_lenrange += 1
            neg = b"-" in line
            try:
                iPCR = int(c[4])
                r1 = int(c[5]) + int(c[6]) + int(c[7]) + int(c[8])
                r2 = int(c[9]) + int(c[10])
                pl = int(c[11]) + int(c[12])
            except Exception:
                parse_fail += 1
                continue
            if neg and min(iPCR, r1, r2, pl) < 0:
                continue
            inp = iPCR + pl + 1
            if inp < 10:
                continue
            n_input10 += 1
            rna = r1 + r2
            if rna >= 1:
                n_rna_any += 1
            elig += 1
            key = b"%s:%d-%d:%s" % (chrom, st, en, strand)
            h = int.from_bytes(sha(key).digest()[:8], "big")
            if len(heap) < SEL_CAP:
                heapq.heappush(heap, (-h, key))
            elif h < -heap[0][0]:
                heapq.heapreplace(heap, (-h, key))
            if N - last_report >= 50000000:
                last_report = N
                print("[pass] N=%d elig=%d %.0fs" % (N, elig, time.time() - t0), flush=True)
    buf += d.flush()

json.dump({"file": "GSE78709_SuRE-counts-K562_B45_B55_LP170105.txt.gz",
           "header": (hdr or b"").decode("utf-8", "replace"),
           "n_rows": N, "parse_fail": parse_fail,
           "n_strand": {k.decode(): v for k, v in strand_ct.items()},
           "n_chrom": {k.decode(): v for k, v in chrom_ct.items()},
           "n_start_lt_end": n_startok, "n_canonical_chrom": n_canon,
           "n_len_200_2000": n_lenrange, "n_input_ge10": n_input10,
           "n_rna_ge1": n_rna_any, "n_eligible": elig,
           "len_hist_100bp": {str(k): v for k, v in sorted(lenhist.items())},
           "elapsed_s": round(time.time() - t0, 1)},
          open(OUT, "w", encoding="utf-8"), indent=2)
print("[pass1b] N=%d elig=%d heap=%d %.1fs" % (N, elig, len(heap), time.time() - t0), flush=True)

sel = sorted((h, k) for h, k in heap)
json.dump({"selection_rule": "25000 smallest sha256('chrom:start-end:strand') among eligible measured-fragment rows (label-free, independent of RNA/activity)",
           "n_selected": len(sel),
           "selected": [k.decode() for _, k in sel]},
          open(SEL, "w", encoding="utf-8"), indent=1)
print("[pass1b] selected saved %d" % len(sel), flush=True)
