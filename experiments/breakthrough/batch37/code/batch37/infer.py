#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Batch37 step 2 - ZERO-fit frozen PARM HEK293 forward on the unique variable inserts.

Pinned official PARM code + 5 official HEK293 fold weights (read-only), official
central zero-pad (L_max=600), equal mean of 5 folds. No fitting/optimizer/gradient.
Genuine CUDA execution with device assertions and contemporaneous NVIDIA timestamps.
"""
import argparse, hashlib, json, os, subprocess, sys, time
import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ASSET = os.path.join(REPO, "data/breakthrough_20261002/batch31/_external")
FROZEN_MAN = os.path.join(REPO, "reports/breakthrough_20261002/batch35/frozen_assets.json")
L_MAX = 600; FILTER = 125; TYPE_LOSS = "poisson"; BATCH = 32
OUTDIR = os.path.join(REPO, "data/breakthrough_20261002/batch37")

def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""): h.update(b)
    return h.hexdigest()

def nvidia_smi():
    try:
        return subprocess.check_output(["nvidia-smi",
            "--query-gpu=name,memory.used,memory.total,utilization.gpu,temperature.gpu",
            "--format=csv,noheader"], text=True, timeout=20).strip()
    except Exception as e:
        return "unavailable: " + repr(e)

def proc_creation_winfiletime():
    try:
        import ctypes
        k = ctypes.windll.kernel32
        h = k.OpenProcess(0x0400, False, os.getpid())
        c = ctypes.c_ulonglong(); e = ctypes.c_ulonglong(); x = ctypes.c_ulonglong(); y = ctypes.c_ulonglong()
        k.GetProcessTimes(h, ctypes.byref(c), ctypes.byref(e), ctypes.byref(x), ctypes.byref(y))
        return str(c.value)
    except Exception as ex:
        return "unavailable: " + repr(ex)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--unique-fasta", default=os.path.join(OUTDIR, "unique_sequences.fasta"))
    ap.add_argument("--out-tsv", default=os.path.join(OUTDIR, "unique_scores.tsv"))
    ap.add_argument("--out-json", default=os.path.join(OUTDIR, "inference_account.json"))
    args = ap.parse_args()
    t0 = time.time()

    with open(FROZEN_MAN, encoding="utf-8") as f: fman = json.load(f)
    for rel, exp in fman["weights_sha256"].items():
        got = sha256_file(os.path.join(ASSET, "src", rel))
        if got != exp:
            print(json.dumps({"status": "ERROR", "error": "weight SHA mismatch", "rel": rel})); return 3

    sys.path.insert(0, os.path.join(ASSET, "parm_pkg"))
    sys.path.insert(1, os.path.join(ASSET, "pydeps"))
    import torch
    from Bio import SeqIO
    from PARM.PARM_utils_load_model import load_PARM
    from PARM.PARM_predict import sequence_to_onehot
    from PARM.version import __version__ as PARM_VERSION

    recs = [(r.id, str(r.seq).upper()) for r in SeqIO.parse(args.unique_fasta, "fasta")]
    print("n_unique", len(recs), flush=True)
    assert len({r[0] for r in recs}) == len(recs), "duplicate seq ids"

    if not torch.cuda.is_available():
        print(json.dumps({"status": "ERROR", "error": "CUDA_UNAVAILABLE_REFUSING_CPU_FALLBACK"})); return 4
    device = torch.device("cuda:0")

    model_dir = os.path.join(ASSET, "src", "HEK293")
    folds = ["HEK293_fold%d.parm" % i for i in range(5)]
    models = []
    for fn in folds:
        m = load_PARM(os.path.join(model_dir, fn), filter_size=FILTER, train=False, type_loss=TYPE_LOSS)
        models.append(m)
    params_cuda = all(p.is_cuda for m in models for p in m.parameters())
    param_dev = str(next(models[0].parameters()).device)
    # probe for input device assertion
    probe = torch.tensor(np.float32(sequence_to_onehot([recs[0][1]], L_max=L_MAX))).permute(0, 2, 1).to(device)
    input_cuda = bool(probe.is_cuda); del probe
    if not (params_cuda and input_cuda):
        print(json.dumps({"status": "ERROR", "error": "DEVICE_ASSERTION_FAILED",
                          "params_cuda": params_cuda, "input_cuda": input_cuda})); return 4

    acc = {"pid": os.getpid(), "pid_creation_winfiletime": proc_creation_winfiletime(),
           "device": "cuda:0", "torch": torch.__version__, "parm_version": PARM_VERSION,
           "filter_size": FILTER, "type_loss": TYPE_LOSS, "L_max": L_MAX, "batch": BATCH,
           "model_params_on_cuda": bool(params_cuda), "model_param_device": param_dev,
           "input_tensor_on_cuda": input_cuda, "n_unique": len(recs), "n_folds": len(models),
           "nvidia_smi_start": nvidia_smi(), "wall_start": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
    acc["nvidia_smi_during"] = nvidia_smi()

    fold_scores = {fn: np.empty(len(recs), dtype=np.float64) for fn in folds}
    t1 = time.time()
    with torch.no_grad():
        for start in range(0, len(recs), BATCH):
            chunk = recs[start:start + BATCH]
            seqs = [c[1] for c in chunk]
            X = torch.tensor(np.float32(sequence_to_onehot(seqs, L_max=L_MAX))).permute(0, 2, 1).to(device)
            assert X.is_cuda, "input not on cuda"
            for fn, m in zip(folds, models):
                out = m(X)
                fold_scores[fn][start:start + len(chunk)] = out.detach().cpu().numpy().reshape(-1)
            if (start // BATCH) % 100 == 0:
                print("  batch", start // BATCH, "/", (len(recs) + BATCH - 1) // BATCH,
                      "elapsed", round(time.time() - t1, 1), flush=True)
    fwd_wall = round(time.time() - t1, 3)
    acc["official_forward_wall_s"] = fwd_wall
    acc["nvidia_smi_after"] = nvidia_smi()
    acc["wall_end"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    acc["elapsed_s"] = round(time.time() - t0, 3)

    mean_score = np.mean([fold_scores[fn] for fn in folds], axis=0)
    cols = ["seq_id", "length"] + folds + ["mean_score"]
    with open(args.out_tsv, "w", encoding="utf-8", newline="") as f:
        f.write("\t".join(cols) + "\n")
        for i, (sid, seq) in enumerate(recs):
            row = [sid, str(len(seq))] + ["%.10g" % fold_scores[fn][i] for fn in folds] + ["%.10g" % mean_score[i]]
            f.write("\t".join(row) + "\n")
    acc["out_tsv"] = args.out_tsv
    acc["out_tsv_sha256"] = sha256_file(args.out_tsv)
    acc["status"] = "OK"
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(acc, f, ensure_ascii=False, indent=2)
    print(json.dumps({"status": "OK", "n_scored": len(recs), "forward_wall_s": fwd_wall,
                      "elapsed_s": acc["elapsed_s"], "out_tsv_sha256": acc["out_tsv_sha256"]}))
    return 0

if __name__ == "__main__":
    sys.exit(main())
