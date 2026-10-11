# -*- coding: utf-8 -*-
# batch52: build per-measured-fragment training table + one-hot memmap (uint8, padded 600).
import os, json, gzip, collections, hashlib, time
import numpy as np
ROOT = r"E:/promoter_atlas"
DATA = os.path.join(ROOT, "data/breakthrough_20261002/batch52")
B51  = os.path.join(ROOT, "data/breakthrough_20261002/batch51")
COH  = os.path.join(DATA, "cohort")
PREP = os.path.join(DATA, "prep"); os.makedirs(PREP, exist_ok=True)
BED  = os.path.join(B51, "GSE301246_PARM_Training_Library_v2_MH4_normalized_fold0.bed.gz")
LMAX = 600
T0 = time.time()
def log(*a): print("[%7.1fs]" % (time.time()-T0), *a, flush=True)

# coord_id -> (split, union_id, length, seq)
info = {}
with open(os.path.join(COH, "endpoint_seqs.tsv"), encoding="utf-8") as f:
    f.readline()
    for line in f:
        a = line.rstrip("\n").split("\t"); info[a[0]] = a[5]
log("endpoint seqs", len(info))
meta = {}
with open(os.path.join(COH, "members.tsv"), encoding="utf-8") as f:
    f.readline()
    for line in f:
        a = line.rstrip("\n").split("\t")
        meta[a[0]] = (a[2], a[1], int(a[7]))
log("members", len(meta))

# activity from source (all eligible rows), detect conflicts
act = {}; conflict = set()
with gzip.open(BED, "rt", encoding="utf-8") as f:
    h = f.readline().rstrip("\n").split("\t")
    for line in f:
        a = line.rstrip("\n").split("\t")
        try: v = float(a[5])
        except Exception: continue
        L = int(a[3]) - int(a[2])
        if not (100 <= L <= 600): continue
        cid = "%s:%d-%d:%s" % (a[1], int(a[2]), int(a[3]), a[4])
        if cid in act and act[cid] != v: conflict.add(cid)
        act[cid] = v
log("activities", len(act), "conflicts", len(conflict))

order = []
for cid, seq in info.items():
    if cid in conflict: continue
    if cid not in meta or cid not in act: continue
    order.append(cid)
log("usable rows", len(order))
CODET = np.zeros(256, dtype=np.uint8)
for c, i in zip("ACGT", range(4)): CODET[ord(c)] = i + 1  # 1..4 ; unknown/n -> 0
X = np.lib.format.open_memmap(os.path.join(PREP, "onehot.npy"), mode="w+", dtype=np.uint8,
                              shape=(len(order), LMAX, 4))
rowsf = open(os.path.join(PREP, "rows.tsv"), "w", encoding="utf-8", newline="\n")
rowsf.write("idx\tcoord_id\tunion_id\tsplit\tlength\tactivity\tseqsha\n")
splits = collections.Counter()
for r, cid in enumerate(order):
    seq = info[cid]; sp, uid, L = meta[cid]; v = act[cid]
    arr = np.zeros((LMAX, 4), dtype=np.uint8)
    b = np.frombuffer(seq.encode("ascii"), dtype=np.uint8)
    codes = CODET[b]  # 0..4
    m = codes > 0
    idx = codes[m] - 1
    pos = np.nonzero(m)[0]
    arr[pos, idx] = 1
    X[r] = arr
    splits[sp] += 1
    rowsf.write("%d\t%s\t%s\t%s\t%d\t%.9g\t%s\n" % (r, cid, uid, sp, L, v, hashlib.sha256(seq.encode()).hexdigest()))
rowsf.close()
X.flush(); del X
json.dump({"n": len(order), "splits": dict(splits), "lmax": LMAX, "n_conflict_excluded": len(conflict),
           "onehot_file": "onehot.npy", "rows_file": "rows.tsv"},
          open(os.path.join(PREP, "prep_manifest.json"), "w"), indent=2)
log("DONE", dict(splits))
