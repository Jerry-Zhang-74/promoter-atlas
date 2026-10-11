# -*- coding: utf-8 -*-
"""batch30 native FASTA prediction interface (minimal; readiness scope only).

Direct user FASTA in -> exact traceable preprocessing -> native-length forward (cuda:0, batch<=32)
-> structured output.  Uses the SAME native_batch function as TRAIN/VAL/TEST.
Explicit unsupported fields are emitted as null with a reason.  No confidence interval is
fabricated.  This is a per-fragment RNA/plasmid PROXY for K562 SuRE; NOT HEK293T activity.
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import b30_common as C
import b30_data as D
import b30_model as MOD


def predict_fasta(fasta_path, ckpt_path=None, device="cuda:0", model=None):
    import torch
    C.set_thread_limits(4); torch.set_num_threads(4)
    seqs = []
    names = []
    cur = None
    with open(fasta_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                cur = line[1:].strip(); names.append(cur); seqs.append("")
            else:
                if cur is None:
                    raise ValueError("FASTA must start with '>'")
                seqs[-1] += line.upper()
    codes, lens = D.encode_native(seqs)          # exact ACGT check; storage padding never read
    if model is None:
        model = MOD.build_cnn("legnet").to(device)
        if ckpt_path:
            st = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            model.load_state_dict(st["model"] if "model" in st else st)
    model.eval()
    preds = np.zeros(len(seqs), dtype=np.float64)
    with torch.no_grad():
        for t, L, chunk in D.iter_native_batches(codes, lens, np.arange(len(seqs)), device, torch, batch=32):
            p = model(t)[:, 0].detach().cpu().numpy()
            for k, g in enumerate(chunk):
                preds[int(g)] = p[k]
    out = {"batch": "batch30", "n_sequences": len(seqs), "device": device,
           "inputs": [{"name": names[i], "length": int(lens[i]), "y_proxy": float(preds[i])} for i in range(len(seqs))],
           "supported": {"y_proxy": "K562 SuRE per-fragment RNA/plasmid proxy y=mean_r log2(1+a_r), a_r=(r_r/N_Rr)/(D/N_D), D=1+P"},
           "unsupported_fields": {"confidence_interval": None, "HEK293T_activity": None, "gene_E_S_B": None,
                                  "reason": "no calibrated uncertainty; proxy != HEK293T physical activity; not a gene-level model"},
           "written_at": C.now_local()}
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("fasta")
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    res = predict_fasta(a.fasta, a.ckpt)
    if a.out:
        C.atomic_write_json(a.out, res)
    print(json.dumps(res, indent=1))
