#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Run the frozen batch34 numerical tests through the wrapper and validate vs the frozen manifest."""
import json, os, subprocess, sys, time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
WRAP = REPO / "code" / "batch34" / "parm_fasta_predict.py"
MAN = REPO / "reports" / "breakthrough_20261002" / "batch34" / "test_manifest.json"
RES = REPO / "data" / "breakthrough_20261002" / "batch34" / "results"
RES.mkdir(parents=True, exist_ok=True)
PY = str(REPO / ".venv" / "Scripts" / "python.exe")

man = json.loads(MAN.read_text(encoding="utf-8"))
env = dict(os.environ); env["PYTHONUTF8"] = "1"; env["PYTHONIOENCODING"] = "utf-8"

results = {}
for fname, spec in man["files"].items():
    stem = fname.replace(".fasta", "")
    oj = str(RES / (stem + ".json")); ot = str(RES / (stem + ".tsv"))
    t0 = time.time()
    p = subprocess.run([PY, "-X", "utf8", str(WRAP), "--input", str(REPO / spec["path"]),
                        "--out-json", oj, "--out-tsv", ot],
                       cwd=str(REPO), env=env, capture_output=True, text=True)
    wall = round(time.time() - t0, 2)
    entry = {"file": fname, "input_sha256": spec["sha256"], "exit_code": p.returncode,
             "wall_s": wall, "out_json": str(Path(oj).relative_to(REPO)).replace("\\", "/"), "checks": []}
    payload = None
    if os.path.exists(oj):
        payload = json.loads(Path(oj).read_text(encoding="utf-8"))
    for exp in spec["expect"]:
        if "global" in exp:
            ok = (p.returncode == 2 and payload and payload.get("error") == exp["global"])
            if "duplicate_ids" in exp and payload:
                ok = ok and payload.get("duplicate_ids") == exp["duplicate_ids"]
            entry["checks"].append({"expect": exp["global"], "ok": bool(ok)})
            continue
        if not payload or payload.get("status") != "OK":
            entry["checks"].append({"id": exp["id"], "ok": False, "why": "no OK payload"})
            continue
        rec = next((r for r in payload["records"] if r["id"] == exp["id"]), None)
        if rec is None:
            entry["checks"].append({"id": exp["id"], "ok": False, "why": "record missing"}); continue
        ok = True; detail = {}
        for k in ("input_validity", "software_length_support", "invalid_reason", "contains_N"):
            if k in exp:
                ok = ok and (rec.get(k) == exp[k]); detail[k] = rec.get(k)
        if exp.get("score", "MISS") is None:
            ok = ok and (rec.get("score") is None); detail["score"] = rec.get("score")
        if "batch31_saved_score" in exp:
            sc = rec.get("score")
            d = abs(sc - exp["batch31_saved_score"]) if sc is not None else None
            ok = ok and (d is not None and d <= exp["tolerance"])
            detail["score"] = sc; detail["batch31_saved_score"] = exp["batch31_saved_score"]
            detail["abs_diff"] = d; detail["tolerance"] = exp["tolerance"]
        entry["checks"].append({"id": exp["id"], "ok": bool(ok), "detail": detail})
    entry["all_ok"] = all(c["ok"] for c in entry["checks"])
    results[fname] = entry

summary = {"batch": "batch34", "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
           "manifest": str(MAN.relative_to(REPO)).replace("\\", "/"),
           "all_ok": all(e["all_ok"] for e in results.values()),
           "n_files": len(results), "n_ok": sum(1 for e in results.values() if e["all_ok"]),
           "files": results}
outp = REPO / "reports" / "breakthrough_20261002" / "batch34" / "test_evidence.json"
outp.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps({"all_ok": summary["all_ok"], "n_ok": summary["n_ok"], "n_files": summary["n_files"],
                  "wrote": str(outp)}, ensure_ascii=False))
for f, e in results.items():
    print(f, e["exit_code"], e["all_ok"], [c.get("id", c.get("expect")) for c in e["checks"]])
