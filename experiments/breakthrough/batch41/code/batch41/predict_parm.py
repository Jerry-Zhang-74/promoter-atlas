#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""batch41 frozen PARM HEK293 inference: official bare-FASTA central-zero-600,
exact 270bp insert input, 5 folds stored separately + equal mean. ZERO fits.

Device is asserted (params on cuda, input tensor on cuda); no CPU fallback.
Actual PID/creation, contemporaneous nvidia-smi telemetry and elapsed are recorded.
Predictions cached resumably, keyed by exact sequence SHA.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import b41_common as C  # noqa: E402


def main():
    t0 = time.time()
    C.set_thread_limits(4)
    import subprocess
    import torch
    if not torch.cuda.is_available():
        C.atomic_write_json(C.DATA / "cache/parm/failed.json",
                            {"error": "CUDA_UNAVAILABLE_REFUSING_SILENT_CPU_FALLBACK"})
        print("ERROR: CUDA unavailable")
        return 4
    device = torch.device("cuda:0")

    # asset SHA verification (identity)
    man = json.load(open(C.PARM_FROZEN_MANIFEST, encoding="utf-8"))
    problems = []
    wsha = {}
    for rel, exp in man["weights_sha256"].items():
        got = C.sha256_file(C.PARM_EXT / "src" / rel)
        wsha[rel] = got
        if got != exp:
            problems.append("weights " + rel)
    for rel, exp in man["source_sha256"].items():
        if C.sha256_file(C.PARM_EXT / rel) != exp:
            problems.append("source " + rel)
    if problems:
        C.atomic_write_json(C.DATA / "cache/parm/failed.json",
                            {"error": "ASSET_SHA_MISMATCH", "problems": problems})
        print("ERROR asset sha", problems)
        return 3

    rows = []
    import csv
    with open(C.DATA / "frozen_input_table.tsv", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh, delimiter="\t"))
    seqs = [r["sequence"] for r in rows]
    shas = [r["sequence_sha256"] for r in rows]
    n = len(seqs)

    sys.path.insert(0, str(C.PARM_EXT / "parm_pkg"))
    sys.path.insert(1, str(C.PARM_EXT / "pydeps"))
    from PARM.PARM_utils_load_model import load_PARM  # noqa: E402
    from PARM.PARM_predict import get_prediction  # noqa: E402
    from PARM.version import __version__ as PARM_VERSION  # noqa: E402

    model_dir = C.PARM_EXT / "src/HEK293"
    fold_scores = np.zeros((C.N_FOLDS, n), dtype=np.float64)
    rt = {"pid": os.getpid(), "pid_creation_winfiletime": C.proc_creation_winfiletime(),
          "started_at": C.now_local(), "nvidia_smi_start": C.nvidia_smi(),
          "device": "cuda:0", "parm_version": PARM_VERSION, "n_seqs_per_batch": C.PARM_BATCH}
    probe_done = False
    for f in range(C.N_FOLDS):
        model = load_PARM(str(model_dir / ("HEK293_fold%d.parm" % f)),
                          filter_size=C.PARM_FILTER, train=False, type_loss=C.PARM_LOSS)
        params_cuda = all(p.is_cuda for p in model.parameters())
        param_dev = str(next(model.parameters()).device)
        if not probe_done:
            from PARM.PARM_predict import sequence_to_onehot
            probe = torch.tensor(np.float32(sequence_to_onehot([seqs[0]], L_max=C.PARM_L_MAX))
                                 ).permute(0, 2, 1).to(device)
            rt["model_params_on_cuda"] = bool(params_cuda)
            rt["model_param_device"] = param_dev
            rt["input_tensor_on_cuda"] = bool(probe.is_cuda)
            del probe
            if not (params_cuda and rt["input_tensor_on_cuda"]):
                C.atomic_write_json(C.DATA / "cache/parm/failed.json",
                                    {"error": "DEVICE_ASSERTION_FAILED", "rt": rt})
                return 4
            probe_done = True
        if f == 0:
            rt["nvidia_smi_during"] = C.nvidia_smi()
        for b0 in range(0, n, C.PARM_BATCH):
            chunk = seqs[b0:b0 + C.PARM_BATCH]
            out = get_prediction(list(chunk), model, L_max=C.PARM_L_MAX)
            out = np.asarray(out).reshape(-1)
            fold_scores[f, b0:b0 + len(chunk)] = out
        del model
        torch.cuda.empty_cache()
        rt.setdefault("fold_wall_s", {})[str(f)] = round(time.time() - t0, 3)
    parm_mean = fold_scores.mean(axis=0)
    rt["nvidia_smi_after"] = C.nvidia_smi()
    rt["elapsed_s"] = round(time.time() - t0, 3)
    rt["total_sequences"] = n
    rt["finite"] = bool(np.isfinite(parm_mean).all())
    if not rt["finite"]:
        C.atomic_write_json(C.DATA / "cache/parm/failed.json", {"error": "NONFINITE", "rt": rt})
        return 5

    cache = {"model": "PARM_HEK293_5fold_equal_mean",
             "input_sha": C.sha256_file(C.DATA / "frozen_input_table.tsv"),
             "code_sha256": C.sha256_file(Path(__file__)),
             "weights_sha256": wsha, "runtime": rt,
             "pred": {shas[i]: {"parm_folds": [float(fold_scores[f][i]) for f in range(C.N_FOLDS)],
                                "parm_mean": float(parm_mean[i]),
                                "parm_fold_spread": float(fold_scores[:, i].std(ddof=0))}
                      for i in range(n)}}
    C.atomic_write_json(C.DATA / "cache/parm/preds.json", cache)
    print(json.dumps({"n": n, "parm_mean_min": float(parm_mean.min()),
                      "parm_mean_max": float(parm_mean.max()),
                      "params_cuda": rt.get("model_params_on_cuda"),
                      "input_cuda": rt.get("input_tensor_on_cuda"),
                      "nvidia_start": rt["nvidia_smi_start"], "nvidia_after": rt["nvidia_smi_after"],
                      "elapsed_s": rt["elapsed_s"]}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
