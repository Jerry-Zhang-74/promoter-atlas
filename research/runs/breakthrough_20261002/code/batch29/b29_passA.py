# -*- coding: utf-8 -*-
"""BATCH29 passA: single streaming pass over the frozen SuRE deposited table.

Parent-approved NEW proxy denominator D_i = 1 + P_i, P_i = plasmid PCRrep1 + PCRrep2
(one fixed pseudocount per unique measured interval; iPCR EXCLUDED from target and eligibility).
Eligibility: canonical chr1-22/X/Y, valid strand/coords, 200<=length<=2000, finite nonnegative
counts, P_i>=1 (equiv D_i>=2).  Retain ALL zero-RNA rows meeting DNA eligibility.
Selection: 10000 lowest sha256('chrom:start-end:strand') among eligible unique intervals,
fully independent of RNA.  NO estimator, NO model."""

import gzip, json, time, hashlib, heapq, os

P_SRC = "data/breakthrough_20261002/batch27/source/GSE78709_SuRE-counts-K562_B45_B55_LP170105.txt.gz"
OUT = "reports/breakthrough_20261002/batch29/passA.json"
SELOUT = "data/breakthrough_20261002/batch29/selected_rows.json"
CANON = set(["chr%d" % i for i in range(1, 23)] + ["chrX", "chrY"])
CAP = 10000

def main():
    t0 = time.time()
    N = 0; pf = 0
    n_badcoord = 0; n_noncanon = 0; n_len = 0; n_neg = 0; n_depth = 0; n_elig = 0; n_dup = 0
    ND = 0; NR1 = 0; NR2 = 0; NIP = 0; NPL = 0
    nP_bins = {"1": 0, "2-3": 0, "4-8": 0, ">=9": 0}
    nP_bins_sel = {"1": 0, "2-3": 0, "4-8": 0, ">=9": 0}
    # eligible-set detection QC
    elig_r1pos = 0; elig_r2pos = 0; elig_both = 0
    heap = []          # (neg_sha_int, key)
    selset = set()
    rowdata = {}       # key -> compact row for currently-selected keys only

    def pbin(p):
        return "1" if p == 1 else ("2-3" if p <= 3 else ("4-8" if p <= 8 else ">=9"))

    with gzip.open(P_SRC, "rt", encoding="utf-8") as f:
        hdr = f.readline().rstrip("\n").split("\t")
        for line in f:
            N += 1
            c = line.rstrip("\n").split("\t")
            if len(c) != 13:
                pf += 1; continue
            try:
                ip = int(c[4])
                r1 = int(c[5]) + int(c[6]) + int(c[7]) + int(c[8])
                r2 = int(c[9]) + int(c[10])
                p1 = int(c[11]); p2 = int(c[12]); P = p1 + p2
                st = int(c[1]); en = int(c[2])
            except Exception:
                pf += 1; continue
            # full-table library totals over the full valid deposited table
            ND += 1 + P; NIP += ip; NPL += P; NR1 += r1; NR2 += r2
            if min(ip, r1, r2, P, st, en) < 0:
                n_neg += 1; continue
            if not (st < en) or c[3] not in ("+", "-"):
                n_badcoord += 1; continue
            if c[0] not in CANON:
                n_noncanon += 1; continue
            L = en - st
            if not (200 <= L <= 2000):
                n_len += 1; continue
            if P < 1:                       # eligibility P_i>=1 (D_i>=2)
                n_depth += 1; continue
            n_elig += 1
            nP_bins[pbin(P)] += 1
            if r1 > 0: elig_r1pos += 1
            if r2 > 0: elig_r2pos += 1
            if r1 > 0 and r2 > 0: elig_both += 1
            key = "%s:%d-%d:%s" % (c[0], st, en, c[3])
            if key in selset:
                n_dup += 1; continue
            h = int(hashlib.sha256(key.encode("utf-8")).hexdigest(), 16)
            if len(heap) < CAP:
                heapq.heappush(heap, (-h, key)); selset.add(key)
                rowdata[key] = [c[0], st, en, c[3], ip, r1, r2, p1, p2, h]
            elif h < -heap[0][0]:
                _, ev = heapq.heapreplace(heap, (-h, key))
                selset.discard(ev); rowdata.pop(ev, None)
                selset.add(key)
                rowdata[key] = [c[0], st, en, c[3], ip, r1, r2, p1, p2, h]
            if N % 30000000 == 0:
                print("[passA] N=%d elig=%d sel=%d %.0fs" % (N, n_elig, len(selset), time.time() - t0), flush=True)

    sel = sorted(rowdata.values(), key=lambda r: r[9])
    sel_r1pos = sum(1 for r in sel if r[5] > 0)
    sel_r2pos = sum(1 for r in sel if r[6] > 0)
    sel_both = sum(1 for r in sel if r[5] > 0 and r[6] > 0)
    sel_zero = sum(1 for r in sel if r[5] == 0 and r[6] == 0)
    for r in sel:
        nP_bins_sel[pbin(r[7] + r[8])] += 1

    rows = []
    for r in sel:
        ch, st, en, sd, ip, r1, r2, p1, p2, h = r
        rows.append({"key": "%s:%d-%d:%s" % (ch, st, en, sd), "chrom": ch, "start": st,
                     "end": en, "strand": sd, "ipcr": ip, "r1": r1, "r2": r2,
                     "pl1": p1, "pl2": p2, "P": p1 + p2, "sha256_12": "%064x" % h})

    res = {
        "batch": "batch29", "pass": "A",
        "source_file": os.path.basename(P_SRC),
        "source_sha256": "8868fa641c2c89bef0c8aa18b6b553030cc7199ea4be4f12c2cd1f3ecd458b85",
        "header": " | ".join(hdr),
        "denominator_definition": "D_i = 1 + P_i (one fixed pseudocount per unique measured interval); P_i = plasmid PCRrep1+PCRrep2; iPCR EXCLUDED",
        "eligibility_rule": {"canonical_chrom": "chr1-22,X,Y", "length_bp": [200, 2000],
                             "P_i_min": 1, "equiv_D_min": 2,
                             "valid": ["start<end", "strand in {+,-}", "counts finite nonnegative"],
                             "rna_thresholded": False,
                             "zero_rna_rows_retained": True},
        "n_rows": N, "n_parse_fail": pf, "n_negative_count": n_neg,
        "n_bad_coord_or_strand": n_badcoord, "n_noncanonical_chrom": n_noncanon,
        "n_length_out_of_range": n_len, "n_below_depth_P_lt_1": n_depth,
        "n_eligible": n_elig, "n_dup_selected_keys_skipped": n_dup,
        "nP_bins_eligible": nP_bins,
        "library_totals_fulltable": {"N_D_1plusP": ND, "N_R1": NR1, "N_R2": NR2,
                                     "N_iPCR": NIP, "N_plasmid_P": NPL},
        "library_offset_note": "N_D=sum_i(1+P_i), N_R1=sum RNA four biorep1 PCR reps, N_R2=sum RNA two biorep2 PCR reps over the FULL valid deposited table (not selection-only); technical PCR reps summed within biological replicate",
        "selection_rule": "lowest sha256('chrom:start-end:strand') among eligible unique intervals; label-free",
        "n_selected": len(rows),
        "selected_QC": {"nP_bins": nP_bins_sel, "r1_pos": sel_r1pos, "r2_pos": sel_r2pos,
                        "both_pos": sel_both, "both_zero": sel_zero,
                        "frac_r1_pos": round(sel_r1pos / max(len(rows), 1), 6),
                        "frac_r2_pos": round(sel_r2pos / max(len(rows), 1), 6),
                        "frac_both_pos": round(sel_both / max(len(rows), 1), 6),
                        "frac_both_zero": round(sel_zero / max(len(rows), 1), 6)},
        "eligible_QC": {"n": n_elig, "r1_pos": elig_r1pos, "r2_pos": elig_r2pos, "both_pos": elig_both,
                        "frac_both_pos": round(elig_both / max(n_elig, 1), 6)},
        "elapsed_s": round(time.time() - t0, 1),
    }
    json.dump(res, open(OUT, "w", encoding="utf-8"), indent=2)
    json.dump({"selection_rule": res["selection_rule"], "n_selected": len(rows),
               "denominator_definition": res["denominator_definition"],
               "eligibility_rule": res["eligibility_rule"],
               "library_totals_fulltable": res["library_totals_fulltable"],
               "rows": rows}, open(SELOUT, "w", encoding="utf-8"), indent=1)
    print(json.dumps(res, indent=2))

if __name__ == "__main__":
    main()
