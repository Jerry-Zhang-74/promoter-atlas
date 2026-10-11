# -*- coding: utf-8 -*-
# batch52 R0: composition ridge baseline (mono..tetra freq + GC + loglen), weighted ridge alpha=100.
import os, sys, json, time, collections
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from exec_gate import ExecGate
ROOT = r"E:/promoter_atlas"
LEDGER = os.path.join(ROOT, "runs/breakthrough_20261002/batch52/exec_gate_ledger.jsonl")
DATA = os.path.join(ROOT, "data/breakthrough_20261002/batch52")
COH  = os.path.join(DATA, "cohort"); PREP = os.path.join(DATA, "prep")
OUT  = os.path.join(DATA, "models"); os.makedirs(OUT, exist_ok=True)
T0 = time.time()
def log(*a): print("[%7.1fs]" % (time.time()-T0), *a, flush=True)
ALPHA = 100.0
LUT = np.full(256, 4, dtype=np.int64)
for c, i in zip(b"ACGT", range(4)): LUT[c] = i
NF = 4 + 16 + 64 + 256 + 2
OFF = [0, 4, 20, 84, 340]  # block offsets for k=1..4 ; GC at 340, loglen 341

def feats(seq):
    a = LUT[np.frombuffer(seq.encode("ascii"), dtype=np.uint8)]
    L = len(a); v = np.zeros(NF, dtype=np.float64)
    for k in range(1, 5):
        m = L - k + 1
        if m > 0:
            key = np.zeros(m, dtype=np.int64); valid = np.ones(m, dtype=bool)
            for j in range(k):
                sub = a[j:j+m]; key = key * 4 + sub; valid &= (sub < 4)
            cnt = np.bincount(key[valid], minlength=4**k)
            v[OFF[k-1]:OFF[k-1]+4**k] = cnt / m
    v[340] = (seq.count('G') + seq.count('C')) / max(1, L)
    v[341] = np.log(max(1, L))
    return v

seqs = {}
with open(os.path.join(COH, "endpoint_seqs.tsv"), encoding="utf-8") as f:
    f.readline()
    for line in f:
        a = line.rstrip("\n").split("\t"); seqs[a[0]] = a[5]
rows = []
with open(os.path.join(PREP, "rows.tsv"), encoding="utf-8") as f:
    f.readline()
    for line in f:
        a = line.rstrip("\n").split("\t")
        rows.append((a[1], a[2], a[3], int(a[4]), float(a[5])))
log("rows", len(rows), "NF", NF)

splits = collections.defaultdict(list)
for i, r in enumerate(rows): splits[r[2]].append(i)
tr = np.array(splits['TRAIN'])
log("split sizes", {k: len(v) for k, v in splits.items()})

grp = collections.Counter(r[1] for r in rows)
Xtr = np.empty((len(tr), NF)); ytr = np.empty(len(tr))
for j, i in enumerate(tr):
    cid, uid, sp, L, v = rows[i]; Xtr[j] = feats(seqs[cid]); ytr[j] = v
log("built TRAIN features", Xtr.shape)
gate = ExecGate(LEDGER, 4)
log("gate status", gate.status())
token = gate.reserve("batch52-R0")
mu = Xtr.mean(0); sd = Xtr.std(0); sd[sd == 0] = 1.0
Xtr = (Xtr - mu) / sd
wgt = np.array([1.0/grp[rows[i][1]] for i in tr]); wgt = wgt / wgt.mean()
ybar = float((wgt*ytr).sum()/wgt.sum()); xbar = (wgt[:, None]*Xtr).sum(0)/wgt.sum()
Xc = Xtr - xbar; yc = ytr - ybar
A = (Xc * wgt[:, None]).T @ Xc + ALPHA*np.eye(NF)
beta = np.linalg.solve(A, (Xc * wgt[:, None]).T @ yc)
intercept = ybar - float(xbar @ beta)
rmse_tr = float(np.sqrt(((Xtr @ beta + intercept - ytr)**2).mean()))
gate.complete(token, identity="r0-composition-ridge-alpha100")
log("solved ridge; intercept", round(intercept, 4), "TRAIN RMSE", round(rmse_tr, 4), "gate completed")

with open(os.path.join(OUT, "r0_predictions.tsv"), "w", encoding="utf-8", newline="\n") as f:
    f.write("coord_id\tunion_id\tsplit\tlength\tpred_activity\n")
    bs = 20000
    for s in range(0, len(rows), bs):
        ch = rows[s:s+bs]
        X = np.empty((len(ch), NF))
        for j, r in enumerate(ch): X[j] = feats(seqs[r[0]])
        p = (X - mu)/sd @ beta + intercept
        for j, r in enumerate(ch):
            f.write("%s\t%s\t%s\t%d\t%.6f\n" % (r[0], r[1], r[2], r[3], p[j]))
np.save(os.path.join(OUT, "r0_beta.npy"), beta)
np.save(os.path.join(OUT, "r0_mu.npy"), mu); np.save(os.path.join(OUT, "r0_sd.npy"), sd)
json.dump({"model": "R0_composition_ridge", "alpha": ALPHA, "n_features": NF,
           "intercept": intercept, "feat_blocks": "mono4+di16+tri64+tetra256=340,+GC+loglen",
           "scale": "TRAIN mean/std (constant features scale1)",
           "weights": "inverse TRAIN group size normalized mean1",
           "n_train": int(len(tr)), "train_endpoint_rmse": rmse_tr, "unit": "raw author Log2Norm_HEK293"},
          open(os.path.join(OUT, "r0_readout.json"), "w"), indent=2)
np.save(os.path.join(OUT, "r0_intercept.npy"), np.array([intercept]))
log("R0 DONE")
