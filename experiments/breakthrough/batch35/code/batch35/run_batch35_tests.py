# -*- coding: utf-8 -*-
"""Batch35 regression tests for the TWO usability defects fixed in code/batch35/parm_fasta_predict.py.

D#1: a VALID record with a FRESH NESTED output path (json & tsv parents do NOT exist and are
     DIFFERENT directories) must NOT crash. Real CUDA inference; crosscheck the ONE known control
     to the frozen batch31/batch34 fivefold score at tolerance 1e-4.
D#2: invalid-only / fatal failure paths must run with NO GPU available (CUDA_VISIBLE_DEVICES="").
"""
import json, os, shutil, subprocess, sys, time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
WRAP = REPO / "code/batch35/parm_fasta_predict.py"
MAN = REPO / "reports/breakthrough_20261002/batch35/test_manifest.json"
RES = REPO / "data/breakthrough_20261002/batch35/results"
PY = str(REPO / ".venv/Scripts/python.exe")
man = json.loads(MAN.read_text(encoding="utf-8"))
env = dict(os.environ); env["PYTHONUTF8"] = "1"; env["PYTHONIOENCODING"] = "utf-8"

# Truly fresh nested output dirs (deleted first; distinct json/tsv parents, both absent).
nested_json_dir = RES / "fresh_nested/deep/json_out"
nested_tsv_dir = RES / "fresh_nested/separate/tsv_out"
for d in (RES / "fresh_nested",):
    if d.exists():
        shutil.rmtree(d)
json_out = nested_json_dir / "known_control.json"
tsv_out = nested_tsv_dir / "known_control.tsv"

summary = {"batch": "batch35", "version": "v6.24.0-293ft-cohort-readiness.rc1",
           "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "tests": {}}


def run(tag, inp, oj, ot, gpu, extra_env=None):
    e = dict(env)
    if not gpu:
        e["CUDA_VISIBLE_DEVICES"] = ""   # prove no-GPU paths
    e.update(extra_env or {})
    t0 = time.time()
    p = subprocess.run([PY, "-X", "utf8", str(WRAP), "--input", str(inp),
                        "--out-json", str(oj), "--out-tsv", str(ot)],
                       cwd=str(REPO), env=e, capture_output=True, text=True)
    wall = round(time.time() - t0, 2)
    payload = None
    if Path(oj).exists():
        try:
            payload = json.loads(Path(oj).read_text(encoding="utf-8"))
        except Exception:
            payload = None
    entry = {"tag": tag, "gpu_requested": gpu, "exit_code": p.returncode, "wall_s": wall,
             "out_json_exists": Path(oj).exists(), "out_tsv_exists": Path(ot).exists(),
             "payload": payload, "stderr_tail": p.stderr[-400:]}
    summary["tests"][tag] = entry
    return entry


# ---- D#1: fresh nested path + separate parents + real GPU ----
print("D#1 fresh nested output dirs existed before run:", nested_json_dir.exists() or nested_tsv_dir.exists())
known = REPO / man["files"]["known_control.fasta"]["path"]
e = run("fresh_nested_gpu", known, json_out, tsv_out, gpu=True)
spec = man["files"]["known_control.fasta"]["expect"][0]
checks1 = {}
checks1["exit_0"] = (e["exit_code"] == 0)
checks1["json_written_to_fresh_nested"] = json_out.exists()
checks1["tsv_written_to_separate_parent"] = tsv_out.exists()
pl = e["payload"] or {}
rec = next((r for r in pl.get("records", []) if r["id"] == spec["id"]), None)
checks1["record_present"] = rec is not None
if rec:
    sc = rec.get("score")
    d = abs(sc - spec["frozen_batch31_fivefold_score"]) if sc is not None else None
    checks1["crosscheck_within_tol"] = (d is not None and d <= spec["tolerance"])
    checks1["score"] = sc; checks1["abs_diff"] = d; checks1["tolerance"] = spec["tolerance"]
    checks1["validity"] = rec.get("input_validity"); checks1["support"] = rec.get("software_length_support")
    checks1["biological_validation_status"] = rec.get("biological_validation_status")
rt = pl.get("runtime", {})
checks1["params_on_cuda"] = bool(rt.get("model_params_on_cuda"))
checks1["input_on_cuda"] = bool(rt.get("input_tensor_on_cuda"))
checks1["nvidia_smi_reports_gpu"] = "RTX" in str(rt.get("nvidia_smi_start", ""))
checks1["has_pid"] = isinstance(rt.get("pid"), int) and rt.get("pid", 0) > 0
checks1["has_pid_creation"] = str(rt.get("pid_creation_winfiletime", "")).isdigit()
e["checks"] = checks1; e["all_ok"] = all(v for k, v in checks1.items() if isinstance(v, bool))

# ---- D#2: no-GPU invalid-only / failure paths ----
def load_expect(fname):
    return man["files"][fname]["expect"]


e = run("invalid_only_nogpu", REPO / man["files"]["invalid_only.fasta"]["path"],
        RES / "invalid_only.json", RES / "invalid_only.tsv", gpu=False)
pl = e["payload"] or {}
c = {"exit_0": e["exit_code"] == 0, "status_OK": pl.get("status") == "OK",
     "no_gpu_run": True}
recs = {r["id"]: r for r in pl.get("records", [])}
for exp in load_expect("invalid_only.fasta"):
    r = recs.get(exp["id"])
    c["rec_" + exp["id"]] = bool(r and r.get("input_validity") == "INVALID"
                                 and r.get("invalid_reason") == exp["invalid_reason"]
                                 and r.get("score") is None)
c["nothing_inferred"] = pl.get("runtime", {}).get("n_inferred") == 0
e["checks"] = c; e["all_ok"] = all(c.values())

for fname, tag in (("empty.fasta", "empty_nogpu"), ("duplicates.fasta", "duplicates_nogpu")):
    exp = load_expect(fname)[0]
    e = run(tag, REPO / man["files"][fname]["path"], RES / (tag + ".json"), RES / (tag + ".tsv"), gpu=False)
    pl = e["payload"] or {}
    c = {"exit_2": e["exit_code"] == 2, "error_matches": pl.get("error") == exp["global"],
         "no_gpu_run": True}
    if "duplicate_ids" in exp:
        c["duplicate_ids"] = pl.get("duplicate_ids") == exp["duplicate_ids"]
    e["checks"] = c; e["all_ok"] = all(v for v in c.values() if isinstance(v, bool))

all_ok = all(t["all_ok"] for t in summary["tests"].values())
summary["all_ok"] = all_ok
summary["gpu_seconds_total"] = round(sum(t["wall_s"] for k, t in summary["tests"].items()
                                         if t["gpu_requested"]), 3)
RES.mkdir(parents=True, exist_ok=True)
(REPO / "reports/breakthrough_20261002/batch35/test_evidence.json").write_text(
    json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps({"all_ok": all_ok,
                  "D1": summary["tests"]["fresh_nested_gpu"]["checks"],
                  "D2_invalid": summary["tests"]["invalid_only_nogpu"]["checks"],
                  "empty": summary["tests"]["empty_nogpu"]["checks"],
                  "duplicates": summary["tests"]["duplicates_nogpu"]["checks"]}, ensure_ascii=False, indent=2))
