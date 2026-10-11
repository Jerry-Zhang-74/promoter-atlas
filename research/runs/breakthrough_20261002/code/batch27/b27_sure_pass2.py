# -*- coding: utf-8 -*-
"""BATCH27 B pass2: second streaming pass over the SuRE counts file.
(a) extract the 13 measured count columns for the FROZEN selected cohort (labels; selection unchanged)
(b) build the genome-wide label-free OVERLAP-COMPONENT group graph over ALL eligible inserts.
NO fits, NO model. Eligibility rule identical to pass1b (predeclared)."""
import json, time, zlib, bisect, array, hashlib
import numpy as np

P = "data/breakthrough_20261002/batch27/source/GSE78709_SuRE-counts-K562_B45_B55_LP170105.txt.gz"
SEL = "reports/breakthrough_20261002/batch27/sure_selected_ids.json"
OUTG = "reports/breakthrough_20261002/batch27/sure_group_graph.json"
OUTL = "reports/breakthrough_20261002/batch27/sure_selected_covariates.json"
CANON = set(b"chr%d" % i for i in range(1, 23)) | {b"chrX", b"chrY"}
CORD = {c: i for i, c in enumerate(sorted(CANON))}

sel = json.load(open(SEL, encoding="utf-8"))["selected"]
selkeys = [s.encode() for s in sel]
selset = set(selkeys)
sel_idx = {k: i for i, k in enumerate(selkeys)}
S = []
for s in sel:
    c, rng, sd = s.split(":")
    st, en = rng.split("-")
    S.append((c, int(st), int(en)))
# per-chrom sorted selected intervals + prefix max end
sel_by_chrom = {}
for i, (c, st, en) in enumerate(S):
    sel_by_chrom.setdefault(c, []).append((st, en, i))
selmx = {}
for c, arr in sel_by_chrom.items():
    arr.sort()
    pm = []
    m = -1
    for st, en, i in arr:
        m = max(m, en)
        pm.append(m)
    sel_by_chrom[c] = ([a[0] for a in arr], [a[1] for a in arr], [a[2] for a in arr])
    selmx[c] = pm

t0 = time.time()
chroms = array.array("i"); starts = array.array("i"); ends = array.array("i")
sel_row = {}          # selected index -> row index in arrays
labels = {}           # selected index -> 13 raw columns
n_elig = 0
n_overlap_hits = 0
n_walk_cap = 0
edges_pairs = []      # (sel_index, row_index)
d = zlib.decompressobj(31); buf = b""; hdr = None; N = 0

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
                continue
            chrom = c[0]
            if chrom not in CANON:
                continue
            try:
                st = int(c[1]); en = int(c[2])
            except Exception:
                continue
            L = en - st
            if not (0 < L and 200 <= L <= 2000):
                continue
            try:
                iPCR = int(c[4])
                r1 = int(c[5]) + int(c[6]) + int(c[7]) + int(c[8])
                r2 = int(c[9]) + int(c[10])
                pl = int(c[11]) + int(c[12])
            except Exception:
                continue
            if b"-" in line and min(iPCR, r1, r2, pl) < 0:
                continue
            if iPCR + pl + 1 < 10:
                continue
            key = b"%s:%d-%d:%s" % (chrom, st, en, c[3])
            ri = n_elig
            chroms.append(CORD[chrom]); starts.append(st); ends.append(en)
            n_elig += 1
            if key in selset:
                si = sel_idx[key]
                sel_row[si] = ri
                labels[si] = {"chrom": chrom.decode(), "start": st, "end": en, "strand": c[3].decode(),
                              "counts": [int(x) for x in c[4:13]]}
            sc = sel_by_chrom.get(chrom.decode())
            if sc is not None:
                sst, sen, sidx = sc
                j = bisect.bisect_left(sst, en)
                pm = selmx[chrom.decode()]
                k = j - 1
                steps = 0
                while k >= 0 and pm[k] > st:
                    steps += 1
                    if steps > 500:
                        n_walk_cap += 1
                        break
                    if sen[k] > st:
                        edges_pairs.append((sidx[k], ri))
                        n_overlap_hits += 1
                    k -= 1
        if N % 50000000 < 1 and N:
            print("[pass2] N=%d elig=%d %.0fs" % (N, n_elig, time.time() - t0), flush=True)
    buf += d.flush()

print("[pass2] stream done N=%d elig=%d hits=%d %.0fs" % (N, n_elig, n_overlap_hits, time.time() - t0), flush=True)

# ---- union-find over all eligible intervals (sweep line per chromosome) ----
n = n_elig
parent = array.array("i", range(n))
def find(x):
    while parent[x] != x:
        parent[x] = parent[parent[x]]
        x = parent[x]
    return x

ca = np.frombuffer(chroms, dtype=np.int32)
sa = np.frombuffer(starts, dtype=np.int32)
ea = np.frombuffer(ends, dtype=np.int32)
for ci in range(len(CORD)):
    idx = np.nonzero(ca == ci)[0]
    if len(idx) < 2:
        continue
    order = idx[np.argsort(sa[idx], kind="mergesort")]
    cur_max = -1; cur_i = -1
    for i in order.tolist():
        if sa[i] < cur_max:
            a = find(cur_i); b = find(i)
            if a != b:
                parent[b] = a
        if ea[i] > cur_max:
            cur_max = ea[i]; cur_i = i
print("[pass2] union-find done %.0fs" % (time.time() - t0), flush=True)

root_of_sel = {}
for si, ri in sel_row.items():
    root_of_sel[si] = find(ri)
roots = {}
for si, r in root_of_sel.items():
    roots.setdefault(r, []).append(si)
import collections
sizes = sorted((len(v) for v in roots.values()), reverse=True)
size_hist = collections.Counter(sizes)
n_with_neighbours = sum(1 for v in roots.values() if len(v) > 1)

firstmem = {r: v[0] for r, v in roots.items()}
group_id = {}
for r, mem in roots.items():
    gid = "SURE_G%06d" % min(mem)
    for m in mem:
        group_id[sel[m]] = gid

res = {
    "pass": "pass2",
    "source": "GSE78709_SuRE-counts-K562_B45_B55_LP170105.txt.gz",
    "n_rows_scanned": N,
    "n_eligible_inserts_genomewide": n_elig,
    "n_selected": len(sel),
    "n_selected_found_in_pass2": len(sel_row),
    "group_definition": "connected components of genomic-interval overlap (>=1bp) over ALL eligible inserts (genome-wide), restricted to the components that contain >=1 selected insert",
    "n_groups_containing_selected": len(roots),
    "n_groups_with_>1_selected_member": n_with_neighbours,
    "selected_members_per_group_hist": {str(k): v for k, v in sorted(size_hist.items())},
    "max_selected_per_group": sizes[0] if sizes else 0,
    "n_edges_selected_vs_any_eligible": n_overlap_hits,
    "n_overlap_walk_capped_queries": n_walk_cap,
    "elapsed_s": round(time.time() - t0, 1),
}
json.dump(res, open(OUTG, "w", encoding="utf-8"), indent=2, default=str)
json.dump({"group_of_selected": group_id,
           "n_groups": len(roots)}, open("reports/breakthrough_20261002/batch27/sure_group_assignment.json", "w", encoding="utf-8"), indent=1, default=str)
json.dump({sel[i]: labels[i] for i in labels},
          open(OUTL, "w", encoding="utf-8"), indent=None)
print(json.dumps(res, indent=2, default=str))
print("[pass2] DONE %.0fs" % (time.time() - t0), flush=True)
