#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Batch34 A: minimal direct FASTA -> prediction interface over the FROZEN official PARM HEK293 model.

Scope (frozen, runs/breakthrough_20261002/batch34/PROTOCOL.txt): ZERO fits.
Official PARM forward inference ONLY, reused UNMODIFIED from
data/breakthrough_20261002/batch31/_external (read-only; nothing copied into a deliverable).
Exact official naked-FASTA preprocessing (PARM central zero-padding to L_max=600) and the
official equal mean of 5 HEK293 folds. No chosen fold/cell, no RC ensemble, no tiling,
no reference-DNA reconstruction, no truncation/windowing of over-length records.
The numeric score is a PARM HEK293 model score: NOT D28 GFP, NOT 293FT activity, NOT a probability.
It is an exploratory cross-cell ranking descriptor only.
"""
import argparse, hashlib, json, os, subprocess, sys, time, tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
DEFAULT_ASSET_ROOT = os.path.join(REPO, "data", "breakthrough_20261002", "batch31", "_external")
DEFAULT_FROZEN_MANIFEST = os.path.join(REPO, "reports", "breakthrough_20261002", "batch35", "frozen_assets.json")
BATCH = "batch35"
VERSION = "v6.24.0-293ft-cohort-readiness.rc1"
L_MAX = 600
FILTER_SIZE = 125
TYPE_LOSS = "poisson"
N_SEQS_PER_BATCH = 1
CANONICAL = set("ACGT")
OFFICIAL_AMBIGUOUS = set("N")
ELIG_OK = ("Exploratory cross-cell ranking descriptor only: PARM HEK293 model score, "
           "NOT 293FT activity / NOT D28 GFP / NOT a probability. UNVALIDATED for new 293FT families.")
ELIG_UNSUP = ("No numerical score produced: sequence length exceeds the PARM software input limit "
              "(L_max=600). No truncation, windowing or tiling was applied.")
ELIG_INVALID = "No numerical score produced: record failed input validation (see invalid_reason)."


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def nvidia_smi():
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,memory.used,memory.total,utilization.gpu,temperature.gpu",
             "--format=csv,noheader"], text=True, timeout=20)
        return out.strip()
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


def fail(code, payload, out_json):
    payload = dict(payload)
    payload.setdefault("status", "ERROR")
    payload.setdefault("batch", BATCH); payload.setdefault("version", VERSION)
    payload.setdefault("generated_at", time.strftime("%Y-%m-%dT%H:%M:%S%z"))
    if out_json:
        try:
            os.makedirs(os.path.dirname(out_json), exist_ok=True)
            with open(out_json, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)
        except Exception:
            pass
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    sys.exit(code)


def verify_assets(asset_root, frozen_manifest):
    with open(frozen_manifest, "r", encoding="utf-8") as f:
        man = json.load(f)
    problems = []
    for rel, expect in man["weights_sha256"].items():
        p = os.path.join(asset_root, "src", rel)
        if not os.path.exists(p):
            problems.append("missing weight " + rel); continue
        got = sha256_file(p)
        if got != expect:
            problems.append("weight SHA mismatch " + rel + " got " + got[:12] + " want " + expect[:12])
    for rel, expect in man["source_sha256"].items():
        p = os.path.join(asset_root, rel)
        if not os.path.exists(p):
            problems.append("missing source " + rel); continue
        got = sha256_file(p)
        if got != expect:
            problems.append("source SHA mismatch " + rel + " got " + got[:12] + " want " + expect[:12])
    return man, problems


def classify(seq):
    flags = {"contains_N": False, "invalid_chars": ""}
    bad = sorted(set(seq) - CANONICAL - OFFICIAL_AMBIGUOUS)
    if len(seq) == 0:
        return "INVALID", "EMPTY_SEQUENCE", "NA", flags
    if bad:
        flags["invalid_chars"] = "".join(bad)
        return "INVALID", "INVALID_CHARS:" + "".join(bad), "NA", flags
    flags["contains_N"] = bool(set(seq) & OFFICIAL_AMBIGUOUS)
    if len(seq) > L_MAX:
        return "VALID", "", "UNSUPPORTED", flags
    return "VALID", "", "SUPPORTED", flags


def main():
    ap = argparse.ArgumentParser(description="Minimal direct FASTA to frozen PARM HEK293 prediction")
    ap.add_argument("--input", required=True)
    ap.add_argument("--out-json", required=True)
    ap.add_argument("--out-tsv", required=True)
    ap.add_argument("--asset-root", default=DEFAULT_ASSET_ROOT)
    ap.add_argument("--frozen-asset-manifest", default=DEFAULT_FROZEN_MANIFEST)
    ap.add_argument("--target-cell", default="293FT")
    args = ap.parse_args()

    # FIX (batch35, defect#1): create BOTH output parents up front, BEFORE any temporary
    # workdir / mkdtemp is created. The batch34 code created the temp dir inside
    # dirname(out_json) before that directory existed -> FileNotFoundError on a fresh nested
    # output path. We also support out_json and out_tsv having DIFFERENT parent directories.
    out_json_dir = os.path.dirname(os.path.abspath(args.out_json))
    out_tsv_dir = os.path.dirname(os.path.abspath(args.out_tsv))
    os.makedirs(out_json_dir, exist_ok=True)
    os.makedirs(out_tsv_dir, exist_ok=True)

    if not os.path.exists(args.input):
        fail(2, {"error": "INPUT_FILE_NOT_FOUND", "input": args.input}, args.out_json)
    if not os.path.exists(args.asset_root):
        fail(3, {"error": "ASSET_ROOT_NOT_FOUND", "asset_root": args.asset_root}, args.out_json)

    man, problems = verify_assets(args.asset_root, args.frozen_asset_manifest)
    weights_sha = man["weights_sha256"]
    if problems:
        fail(3, {"error": "ASSET_SHA_VERIFICATION_FAILED", "asset_root": args.asset_root,
                 "problems": problems}, args.out_json)

    input_sha = sha256_file(args.input)

    sys.path.insert(0, os.path.join(args.asset_root, "parm_pkg"))
    sys.path.insert(1, os.path.join(args.asset_root, "pydeps"))
    from Bio import SeqIO

    records = []
    for rec in SeqIO.parse(args.input, "fasta"):
        records.append({"id": rec.id, "seq": str(rec.seq).upper()})
    if len(records) == 0:
        fail(2, {"error": "EMPTY_INPUT_NO_RECORDS", "input": args.input, "input_sha256": input_sha}, args.out_json)
    ids = [r["id"] for r in records]
    dups = sorted({i for i in ids if ids.count(i) > 1})
    if dups:
        fail(2, {"error": "DUPLICATE_RECORD_IDS", "input": args.input, "input_sha256": input_sha,
                 "duplicate_ids": dups}, args.out_json)

    per = []
    supported = []
    for r in records:
        validity, reason, support, flags = classify(r["seq"])
        per.append({"id": r["id"], "length": len(r["seq"]), "input_validity": validity,
                    "invalid_reason": reason, "software_length_support": support,
                    "contains_N": flags["contains_N"]})
        if validity == "VALID" and support == "SUPPORTED":
            supported.append(r)

    started = time.time()
    runtime = {"pid": os.getpid(), "pid_creation_winfiletime": proc_creation_winfiletime(),
               "device": "cuda:0", "nvidia_smi_start": nvidia_smi(),
               "n_inferred": len(supported),
               "n_unsupported": sum(1 for p in per if p["software_length_support"] == "UNSUPPORTED"),
               "n_invalid": sum(1 for p in per if p["input_validity"] == "INVALID")}

    scores = {}
    if supported:
        import numpy as np
        import torch
        if not torch.cuda.is_available():
            fail(4, {"error": "CUDA_UNAVAILABLE_REFUSING_SILENT_CPU_FALLBACK"}, args.out_json)
        device = torch.device("cuda:0")
        model_dir = os.path.join(args.asset_root, "src", "HEK293")
        from PARM.PARM_utils_load_model import load_PARM
        from PARM.PARM_predict import PARM_predict, sequence_to_onehot
        from PARM.version import __version__ as PARM_VERSION
        m0 = load_PARM(os.path.join(model_dir, "HEK293_fold0.parm"),
                       filter_size=FILTER_SIZE, train=False, type_loss=TYPE_LOSS)
        params_cuda = all(p.is_cuda for p in m0.parameters())
        param_dev = str(next(m0.parameters()).device)
        del m0
        probe = torch.tensor(np.float32(sequence_to_onehot([supported[0]["seq"]], L_max=L_MAX))).permute(0, 2, 1).to(device)
        input_cuda = bool(probe.is_cuda)
        del probe
        runtime["model_params_on_cuda"] = bool(params_cuda)
        runtime["model_param_device"] = param_dev
        runtime["input_tensor_on_cuda"] = input_cuda
        if not (params_cuda and input_cuda):
            fail(4, {"error": "DEVICE_ASSERTION_FAILED", "params_cuda": params_cuda,
                     "input_cuda": input_cuda}, args.out_json)

        # Defect#1 fix: out_json_dir was created above, so mkdtemp here cannot fail on a fresh path.
        tmpdir = tempfile.mkdtemp(prefix="b35_", dir=out_json_dir)
        valid_fa = os.path.join(tmpdir, "valid_supported.fasta")
        with open(valid_fa, "w", encoding="utf-8") as f:
            for r in supported:
                f.write(">" + r["id"] + "\n" + r["seq"] + "\n")
        valid_out = os.path.join(tmpdir, "valid_supported_pred.tsv")
        runtime["nvidia_smi_during"] = nvidia_smi()
        t0 = time.time()
        PARM_predict(input=valid_fa, output=valid_out, model_directory=model_dir,
                     n_seqs_per_batch=N_SEQS_PER_BATCH, store_sequence=True, filter_size=FILTER_SIZE,
                     type_loss=TYPE_LOSS, test_fold=False, L_max=L_MAX)
        runtime["official_forward_wall_s"] = round(time.time() - t0, 3)
        runtime["nvidia_smi_after"] = nvidia_smi()
        runtime["parm_version"] = PARM_VERSION
        with open(valid_out, "r", encoding="utf-8") as f:
            hdr = f.readline().rstrip("\n").split("\t")
            si = hdr.index("header")
            pi = [i for i, h in enumerate(hdr) if h.startswith("prediction_")][0]
            for line in f:
                parts = line.rstrip("\n").split("\t")
                scores[parts[si]] = float(parts[pi])
        try:
            os.remove(valid_fa); os.remove(valid_out); os.rmdir(tmpdir)
        except OSError:
            pass

    out_records = []
    for p in per:
        sc = scores.get(p["id"]) if p["software_length_support"] == "SUPPORTED" else None
        if p["input_validity"] == "INVALID":
            note = ELIG_INVALID
        elif p["software_length_support"] == "UNSUPPORTED":
            note = ELIG_UNSUP
        else:
            note = ELIG_OK
        out_records.append({
            "id": p["id"], "length": p["length"], "input_validity": p["input_validity"],
            "invalid_reason": p["invalid_reason"],
            "software_length_support": p["software_length_support"], "contains_N": p["contains_N"],
            "score": sc, "model_source_cell": "HEK293", "target_requested_cell": args.target_cell,
            "output_definition": "PARM HEK293 promoter-activity model score (equal mean of 5 official HEK293 folds; official central zero-padding L_max=600)",
            "biological_validation_status": "UNVALIDATED_FOR_293FT", "eligibility_note": note})

    runtime["elapsed_s"] = round(time.time() - started, 3)
    payload = {
        "status": "OK", "batch": BATCH, "version": VERSION,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "input": {"path": os.path.abspath(args.input), "sha256": input_sha, "n_records": len(records)},
        "model": {"name": "PARM", "parm_version": runtime.get("parm_version", "0.2.6"),
                  "source_repo": "https://github.com/vansteensellab/PARM",
                  "commit": "260a27082942060364c1c4e2393018a0fa128844",
                  "folds": 5, "aggregation": "equal mean of 5 HEK293 folds", "model_source_cell": "HEK293",
                  "weights_sha256": weights_sha, "asset_root": os.path.abspath(args.asset_root)},
        "assay": {"output_definition": "PARM HEK293 promoter-activity model score", "L_max": L_MAX,
                  "padding": "official central zero-padding", "filter_size": FILTER_SIZE,
                  "type_loss": TYPE_LOSS, "n_seqs_per_batch": N_SEQS_PER_BATCH},
        "target_requested_cell": args.target_cell,
        "biological_validation_status": "UNVALIDATED_FOR_293FT",
        "eligibility_note": "Numeric score = exploratory cross-cell ranking descriptor only; no 293FT biological validation.",
        "accepted_alphabet": {"canonical": "ACGT (case-insensitive)",
                              "official_ambiguous": "N -> all-zero one-hot (allowed, flagged)",
                              "rejected": "any other character -> record INVALID (never silently removed)"},
        "length_policy": {"supported": "1..600", "unsupported": ">600 -> null score, no truncation/tiling"},
        "runtime": runtime, "records": out_records}

    os.makedirs(os.path.dirname(os.path.abspath(args.out_json)), exist_ok=True)
    os.makedirs(os.path.dirname(os.path.abspath(args.out_tsv)), exist_ok=True)
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    cols = ["id", "length", "input_validity", "software_length_support", "score",
            "model_source_cell", "target_requested_cell", "biological_validation_status",
            "output_definition", "eligibility_note", "invalid_reason", "input_sha256"]
    with open(args.out_tsv, "w", encoding="utf-8", newline="") as f:
        f.write("\t".join(cols) + "\n")
        for r in out_records:
            f.write("\t".join([str(r["id"]), str(r["length"]), r["input_validity"],
                r["software_length_support"],
                ("" if r["score"] is None else repr(r["score"])),
                r["model_source_cell"], r["target_requested_cell"], r["biological_validation_status"],
                r["output_definition"], r["eligibility_note"], r["invalid_reason"], input_sha]) + "\n")
    print(json.dumps({"status": "OK", "out_json": os.path.abspath(args.out_json),
                      "n_records": len(out_records), "n_scored": len(scores),
                      "n_unsupported": runtime["n_unsupported"], "n_invalid": runtime["n_invalid"],
                      "elapsed_s": runtime["elapsed_s"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
