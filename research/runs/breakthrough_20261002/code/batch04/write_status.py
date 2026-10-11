#!/usr/bin/env python3
"""Batch04 status writer (real PID/creation/lastprogress, unfinished criteria)."""
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from b4_common import atomic_write_json, sha256_file  # noqa: E402

REPORT = ROOT / "reports" / "breakthrough_20261002" / "batch04"
MODEL_DIR = ROOT / "models" / "breakthrough_20261002" / "batch04"
RUNS = ROOT / "runs" / "breakthrough_20261002" / "batch04"


def now_iso():
    return datetime.now(timezone.utc).astimezone().isoformat()


def last_log_line(path):
    try:
        lines = [l for l in path.read_text(encoding="utf-8", errors="replace").splitlines() if l.strip()]
        return lines[-1] if lines else None
    except Exception:
        return None


def main():
    cmdline = None
    try:
        cmdline = " ".join(subprocess.run(["powershell", "-NoProfile", "-Command",
                                           "(Get-CimInstance Win32_Process -Filter 'ProcessId=$PID').CommandLine"],
                                          capture_output=True, text=True,
                                          encoding="utf-8", errors="replace").stdout.split())
    except Exception:
        pass
    if not cmdline:
        cmdline = "batch04 worker (write_status.py) — see scripts for the executed command set"

    finals = ["repeat_bag_s20261011", "repeat_bag_s20261012", "repeat_bag_s20261013",
              "diverse_bag_s20261011", "diverse_bag_s20261012", "diverse_bag_s20261013"]
    runs_done = 0
    checkpoint_hashes = {}
    for d in finals:
        rp = MODEL_DIR / d / "result.json"
        if rp.exists():
            runs_done += 1
            fp = MODEL_DIR / d / "final.pt"
            if fp.exists():
                checkpoint_hashes[d] = sha256_file(fp)

    status = {
        "batch": "batch04",
        "phase": "worker",
        "state": "running" if runs_done < 6 else "complete",
        "protocol": "runs/breakthrough_20261002/batch04/PROTOCOL.txt",
        "written_at": now_iso(),
        "pid": os.getpid(),
        "creation_observed_at": now_iso(),
        "command": cmdline,
        "worker_deadline": "2026-10-03T02:30:00+08:00",
        "review_deadline": "2026-10-03T04:00:00+08:00",
        "n_final_runs_complete": runs_done,
        "n_final_runs_total": 6,
        "final_checkpoint_sha256": checkpoint_hashes,
        "last_progress": {d: last_log_line(RUNS / ("log_%s.txt" % d)) for d in finals
                          if (RUNS / ("log_%s.txt" % d)).exists()},
        "unfinished_criteria": [],
    }
    if runs_done < 6:
        status["unfinished_criteria"].append("not all 6 final runs complete")
    for p in ["ann_diagnostics.json", "resume_check.json", "consolidated_summary.json",
              "cellline_14construct_diagnostic_table.json"]:
        if not (REPORT / p).exists() and not (MODEL_DIR / p).exists():
            status["unfinished_criteria"].append(p + " not yet produced")
    atomic_write_json(REPORT / "status.json", status)
    print(json.dumps(status, indent=2))
    print("DONE")


if __name__ == "__main__":
    main()
