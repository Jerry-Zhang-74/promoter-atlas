# -*- coding: utf-8 -*-
"""batch59 worker: real GPU smoke + external-cwd + boundary tests on the reviewed bundles.

ZERO fit/solve/backward/train. Only small real forward passes (GPU budget <=30s total).
Extracts the FLAT zips into an EXTERNAL dir whose path contains Chinese + spaces, runs the
predictor from a DIFFERENT external cwd, and exercises strict-input boundary failures.
"""
from __future__ import annotations
import json, os, shutil, subprocess, sys, tempfile, time, zipfile

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
REP = os.path.join(ROOT, "reports", "breakthrough_20261002", "batch59")
PY = os.path.join(ROOT, ".venv", "Scripts", "python.exe")
SEQ = os.path.join(ROOT, "data", "breakthrough_20261002", "batch52", "corrected", "cohort_v2", "sequences.fa")
CACHED = os.path.join(ROOT, "reports", "breakthrough_20261002", "batch54", "Nabs_TEST_parent_granted.tsv")
ZIPS = {
    "Nabs": os.path.join(ROOT, "releases", "batch52_first_model", "v6.38.0-nabs-reviewed-flat.zip"),
    "Njoint": os.path.join(ROOT, "releases", "batch52_second_model", "v6.38.0-njoint-reviewed-flat.zip"),
}
KNOWN_TEST = "chr14:56232529-56233023:+"


def env():
    e = dict(os.environ)
    e.update({"PYTHONDONTWRITEBYTECODE": "1", "PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8"})
    return e


def seq_for(coord, n=30000):
    with open(SEQ, "r", encoding="utf-8") as f:
        for i, ln in enumerate(f):
            if ln.startswith(">"):
                if ln.strip()[1:] == coord:
                    return f.readline().strip()
                if i > 2 * n:
                    break
    return None


def run(args, cwd):
    t = time.time()
    p = subprocess.run(args, cwd=cwd, env=env(), capture_output=True, text=True, encoding="utf-8")
    return {"rc": p.returncode, "sec": round(time.time() - t, 3),
            "stdout": p.stdout[-1500:], "stderr": p.stderr[-800:]}


def main():
    tmp = os.path.join(tempfile.gettempdir(), "batch59 外部 测试")
    if os.path.exists(tmp):
        shutil.rmtree(tmp)
    ext = os.path.join(tmp, "解压 模型包")
    os.makedirs(ext)
    res = {"kind": "batch59_smoke_boundary", "external_root": tmp, "gpu_budget_s": 30.0}

    # extract flat zips to external Chinese+space paths (archive full read test)
    bundles = {}
    for arm, z in ZIPS.items():
        dst = os.path.join(ext, arm + " 包")
        with zipfile.ZipFile(z) as zf:
            bad = zf.testzip()
            zf.extractall(dst)
        bundles[arm] = os.path.join(dst, "predict.py")
        res.setdefault("archive_read", {})[arm] = {
            "extracted_to": dst, "testzip_none": bad is None,
            "has_predict": os.path.exists(os.path.join(dst, "predict.py")),
            "has_pinned_core": os.path.isdir(os.path.join(dst, "pinned_core"))}

    # --- GPU test 1: Nabs known TEST record vs cached score (external cwd) ---
    s = seq_for(KNOWN_TEST)
    kf = os.path.join(tmp, "已知 测试 记录.fa")
    with open(kf, "w", encoding="utf-8") as f:
        f.write(">" + KNOWN_TEST + "\n" + s + "\n")
    cached = None
    with open(CACHED, "r", encoding="utf-8") as f:
        next(f)
        for ln in f:
            a = ln.split("\t")
            if a[0] == KNOWN_TEST:
                cached = float(a[2]); break
    out1 = os.path.join(tmp, "nabs_known.tsv")
    r1 = run([PY, "-X", "utf8", bundles["Nabs"], "--bundle", os.path.dirname(bundles["Nabs"]),
              "--fasta", kf, "--out", out1], cwd=tmp)
    pred = None
    if os.path.exists(out1):
        with open(out1, encoding="utf-8") as f:
            next(f)
            pred = float(f.readline().split("\t")[2])
    res["gpu_nabs_known_test"] = {"run": r1, "cached_activity_pred": cached, "release_activity_pred": pred,
                                  "abs_diff": (abs(pred - cached) if pred is not None else None),
                                  "tol_documented": 1e-4,
                                  "match_within_tol": (pred is not None and cached is not None and abs(pred - cached) <= 1e-4)}

    # --- GPU test 2: Njoint on new artificial FASTA (external cwd) ---
    nf = os.path.join(ROOT, "reports", "breakthrough_20261002", "batch59", "software_only_new_inputs.fa")
    out2 = os.path.join(tmp, "njoint_new.tsv")
    r2 = run([PY, "-X", "utf8", bundles["Njoint"], "--bundle", os.path.dirname(bundles["Njoint"]),
              "--fasta", nf, "--out", out2], cwd=tmp)
    nrows = sum(1 for _ in open(out2, encoding="utf-8")) - 1 if os.path.exists(out2) else 0
    res["gpu_njoint_new_fasta"] = {"run": r2, "n_rows": nrows, "ok": r2["rc"] == 0 and nrows == 3}

    # --- boundary tests (strict input, expect rc=2, no silent pass) ---
    bnd = {}
    cases = {}
    cases["len601"] = "A" * 601
    cases["len2000"] = "A" * 2000
    cases["invalid_base"] = "ACGTZACGT"
    bnd_files = {}
    for k, sq in cases.items():
        p = os.path.join(tmp, k + ".fa")
        open(p, "w", encoding="utf-8").write(">r_" + k + "\n" + sq + "\n")
        bnd_files[k] = p
    bnd_files["empty"] = os.path.join(tmp, "empty.fa")
    open(bnd_files["empty"], "w").close()
    bnd_files["duplicate"] = os.path.join(tmp, "dup.fa")
    open(bnd_files["duplicate"], "w", encoding="utf-8").write(">same\nACGTACGT\n>same\nACGTACGT\n")
    for k, p in bnd_files.items():
        out = os.path.join(tmp, "out_" + k + ".tsv")
        r = run([PY, "-X", "utf8", bundles["Nabs"], "--bundle", os.path.dirname(bundles["Nabs"]),
                 "--fasta", p, "--out", out], cwd=tmp)
        bnd[k] = {"rc": r["rc"], "explicit_fail": r["rc"] == 2, "no_output": not os.path.exists(out),
                  "stderr": r["stderr"].strip().splitlines()[-1] if r["stderr"].strip() else ""}
    res["boundary"] = bnd

    total_gpu = r1["sec"] + r2["sec"]
    res["gpu_total_s"] = round(total_gpu, 3)
    res["gpu_within_budget"] = total_gpu <= 30.0
    res["all_boundary_fail"] = all(v["explicit_fail"] and v["no_output"] for v in bnd.values())
    json.dump(res, open(os.path.join(REP, "smoke_boundary.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=2, sort_keys=True)
    print(json.dumps({k: res[k] for k in ("gpu_total_s", "gpu_within_budget", "all_boundary_fail",
                                          "gpu_nabs_known_test", "gpu_njoint_new_fasta", "boundary")},
                     ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
