#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""BATCH31: per-fold PARM outputs on a small saved input; finiteness + reproducibility
(two identical runs) + fivefold dispersion. ZERO fits."""
import os,sys,json,time,hashlib
import numpy as np
HERE=os.path.dirname(os.path.abspath(__file__)); REPO=os.path.dirname(os.path.dirname(HERE))
EXT=os.path.join(REPO,"data/breakthrough_20261002/batch31/_external")
RES=os.path.join(REPO,"data/breakthrough_20261002/batch31/results")
sys.path.insert(0,os.path.join(EXT,"parm_pkg")); sys.path.insert(1,os.path.join(EXT,"pydeps"))
import torch
from PARM.PARM_utils_load_model import load_PARM
from PARM.PARM_predict import get_prediction
from Bio import SeqIO
MODEL_DIR=os.path.join(EXT,"src/HEK293")
def read_fa(p):
    out=[]
    for rec in SeqIO.parse(p,"fasta"): out.append((rec.id,str(rec.seq).upper()))
    return out
def run(models,seqs):
    return np.array([get_prediction([s for _,s in seqs], m, L_max=600).flatten() for m in models])
def main():
    assert torch.cuda.is_available()
    models=[load_PARM(os.path.join(MODEL_DIR,f"HEK293_fold{i}.parm"), filter_size=125, train=False, type_loss="poisson") for i in range(5)]
    small=os.path.join(RES,"small_repro_input.fasta")
    with open(small,"w",encoding="utf-8") as f:
        for i,(id_,s) in enumerate(read_fa(os.path.join(EXT,"src/example_data_input.fasta"))): f.write(f">{id_}\n{s}\n")
        for id_,s in read_fa(os.path.join(REPO,"data/breakthrough_20261002/batch31/inputs/local_supported.fasta"))[:5]: f.write(f">{id_}\n{s}\n")
    seqs=read_fa(small)
    A=run(models,seqs); B=run(models,seqs)
    finite=bool(np.isfinite(A).all())
    repro=float(np.max(np.abs(A-B)))
    disp=np.std(A,axis=0); mean=np.mean(A,axis=0)
    out={"n_seqs":len(seqs),"n_folds":5,"all_finite":finite,
         "max_abs_diff_two_runs":repro,"reproducible":bool(repro==0.0),
         "fold_mean":mean.tolist(),"fold_std":disp.tolist(),
         "ids":[i for i,_ in seqs],
         "note":"fivefold dispersion is model disagreement only, not a calibrated confidence interval",
         "device":"cuda:0","pid":os.getpid(),"ts":time.strftime("%Y-%m-%dT%H:%M:%S%z")}
    json.dump(out,open(os.path.join(RES,"fold_repro_dispersion.json"),"w",encoding="utf-8"),ensure_ascii=False,indent=2)
    print(json.dumps(out,ensure_ascii=False,indent=2))
if __name__=="__main__": main()
