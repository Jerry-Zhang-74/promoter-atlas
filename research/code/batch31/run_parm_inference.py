#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""BATCH31 section A: faithful official PARM HEK293 inference (ZERO fits).
Runs the UNMODIFIED official PARM_predict() from the pinned external source.
"""
import os, sys, json, time, hashlib, subprocess
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
EXT = os.path.join(REPO, "data/breakthrough_20261002/batch31/_external")
PKG = os.path.join(EXT, "parm_pkg")
DEPS = os.path.join(EXT, "pydeps")
MODEL_DIR = os.path.join(EXT, "src/HEK293")
INPUT_DIR = os.path.join(REPO, "data/breakthrough_20261002/batch31/inputs")
RES = os.path.join(REPO, "data/breakthrough_20261002/batch31/results")
os.makedirs(RES, exist_ok=True)
sys.path.insert(0, PKG)
sys.path.insert(1, DEPS)

import torch

def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1<<20), b""): h.update(b)
    return h.hexdigest()

def nvidia_smi():
    try:
        out = subprocess.check_output(
            ["nvidia-smi","--query-gpu=name,memory.used,memory.total,utilization.gpu,temperature.gpu",
             "--format=csv,noheader"], text=True, timeout=20)
        return out.strip()
    except Exception as e:
        return "unavailable: "+str(e)

def proc_creation():
    try:
        import ctypes, ctypes.wintypes
        k=ctypes.windll.kernel32
        h=k.OpenProcess(0x0400, False, os.getpid())
        c=ctypes.c_ulonglong(); e=ctypes.c_ulonglong(); x=ctypes.c_ulonglong(); y=ctypes.c_ulonglong()
        k.GetProcessTimes(h, ctypes.byref(c), ctypes.byref(e), ctypes.byref(x), ctypes.byref(y))
        return c.value
    except Exception as ex:
        return str(ex)

def main():
    from PARM.PARM_predict import PARM_predict, sequence_to_onehot
    from PARM.PARM_utils_load_model import load_PARM
    from PARM.version import __version__ as PARM_VERSION
    assert torch.cuda.is_available(), "CUDA required; refusing silent CPU fallback"
    dev = torch.device("cuda:0")
    meta = {
        "batch":"batch31","section":"A","official_model":"PARM HEK293 equal mean of 5 folds",
        "parm_source_repo":"https://github.com/vansteensellab/PARM",
        "parm_commit_sha":"260a27082942060364c1c4e2393018a0fa128844",
        "parm_version":PARM_VERSION,
        "pid":os.getpid(),"process_creation_winfiletime":str(proc_creation()),
        "started_at":time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "cuda_available":bool(torch.cuda.is_available()),
        "cuda_device_name":torch.cuda.get_device_name(0),
        "device_used":"cuda:0",
        "nvidia_smi_at_start":nvidia_smi(),
        "L_max":600,"filter_size":125,"type_loss":"poisson","n_seqs_per_batch":1,
        "zero_fits":True,"optimizer_steps":0,"gradients":0,
        "weight_sha256":{f:sha256_file(os.path.join(MODEL_DIR,f)) for f in sorted(os.listdir(MODEL_DIR))},
        "input_sha256":{f:sha256_file(os.path.join(INPUT_DIR,f)) for f in sorted(os.listdir(INPUT_DIR))},
    }
    inputs = {
        "hek673_133bp": os.path.join(INPUT_DIR,"hek673_133bp.fasta"),
        "local_supported": os.path.join(INPUT_DIR,"local_supported.fasta"),
    }
    for name, inp in inputs.items():
        outp = os.path.join(RES, f"parm_pred_{name}.tsv")
        if os.path.exists(outp): os.remove(outp)
        t0=time.time()
        PARM_predict(input=inp, output=outp, model_directory=MODEL_DIR,
                     n_seqs_per_batch=1, store_sequence=True, filter_size=125,
                     type_loss="poisson", test_fold=False, L_max=600)
        meta[name] = {"input":inp,"output":outp,"wall_s":round(time.time()-t0,2),
                      "output_sha256":sha256_file(outp)}
    meta["nvidia_smi_at_end"]=nvidia_smi()
    meta["finished_at"]=time.strftime("%Y-%m-%dT%H:%M:%S%z")
    with open(os.path.join(RES,"parm_inference_meta.json"),"w",encoding="utf-8") as f:
        json.dump(meta,f,ensure_ascii=False,indent=2)
    print(json.dumps(meta,ensure_ascii=False,indent=2))
if __name__=="__main__":
    main()
