#!/usr/bin/env python3
"""Batch05 status writer (real PID/creation/lastprogress + scientific fields)."""
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from b5_common import atomic_write_json, sha256_file  # noqa: E402

REPORT = ROOT / "reports" / "breakthrough_20261002" / "batch05"
MODEL_DIR = ROOT / "models" / "breakthrough_20261002" / "batch05"
RUNS = ROOT / "runs" / "breakthrough_20261002" / "batch05"


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
        cmdline = "batch05 worker (write_status_v5.py) — see runs/batch05 for executed command set"

    finals = ["repeat_bag_s20261011", "diverse_bag_s20261011"]
    runs_done = 0
    checkpoint_hashes = {}
    for d in finals:
        rp = MODEL_DIR / d / "result.json"
        if rp.exists():
            runs_done += 1
            fp = MODEL_DIR / d / "final.pt"
            if fp.exists():
                checkpoint_hashes[d] = sha256_file(fp)

    # scientific fields (populated from consolidated results when available)
    sci = {
        "candidate_version": "v6.6.0-scratchgene.rc7 (diverse_bag, primary)",
        "summary_zh": "",
        "key_results": {},
        "next_action": "等待独立评审与父进程验收; 本批次不自行启动下一科学批次",
    }
    cons_path = REPORT / "consolidated_summary_v5.json"
    if cons_path.exists():
        cons = json.loads(cons_path.read_text(encoding="utf-8"))
        rt = cons["arms"]["repeat_bag"]["test"]["full"]
        dt = cons["arms"]["diverse_bag"]["test"]["full"]
        d = cons["paired_b2_minus_b1"]["test"]["delta_spearman"]
        ab = cons["annotation_baseline_comparison"]
        ev = cons["evidence"]
        sci["key_results"] = {
            "test_spearman_B1": rt["spearman"], "test_spearman_B2": dt["spearman"],
            "test_r2_B1": rt["r2"], "test_r2_B2": dt["r2"],
            "test_mae_B1": rt["mae"], "test_mae_B2": dt["mae"],
            "b2_minus_b1_test_spearman": d["base"],
            "component_bootstrap_ci_low": d["ci_low"],
            "component_bootstrap_ci_high": d["ci_high"],
            "best_annotation_baseline_test_spearman": ab["best_baseline_test_spearman"],
            "evidence_threshold_met": ev["meets_improvement_threshold"],
        }
        sci["summary_zh"] = ("冻结序列-组件划分(精确/RC+检出近邻阻断, 非无同源)下, B1=%.4f / B2=%.4f (测试Spearman), "
                             "B2-B1=%.4f (组件bootstrap CI[%.4f, %.4f]), 最优注释基线=%.4f; "
                             "证据阈值(>=.03且下界>0)=%s; 单种子, 回顾性, 非因果。"
                             % (rt["spearman"], dt["spearman"], d["base"], d["ci_low"], d["ci_high"],
                                ab["best_baseline_test_spearman"], ev["meets_improvement_threshold"]))

    status = {
        "batch": "batch05",
        "phase": "worker",
        "state": "running" if runs_done < 2 else "complete",
        "protocol": "runs/breakthrough_20261002/batch05/PROTOCOL.txt",
        "written_at": now_iso(),
        "pid": os.getpid(),
        "creation_observed_at": now_iso(),
        "command": cmdline,
        "worker_deadline": "2026-10-03T05:30:00+08:00",
        "review_deadline": "2026-10-03T07:00:00+08:00",
        "n_final_runs_complete": runs_done,
        "n_final_runs_total": 2,
        "final_checkpoint_sha256": checkpoint_hashes,
        "last_progress": {d: last_log_line(RUNS / ("log_%s.txt" % d)) for d in finals},
        "scientific_candidate_version": sci["candidate_version"],
        "summary_zh": sci["summary_zh"],
        "key_results": sci["key_results"],
        "next_action": sci["next_action"],
        "unfinished_criteria": [],
    }
    if runs_done < 2:
        status["unfinished_criteria"].append("not all 2 final runs complete")
    for p in ["ann_diagnostics_v5.json", "resume_check_v5.json", "consolidated_summary_v5.json",
              "report.txt", "report.html", "version_registry_v5.json"]:
        if not (REPORT / p).exists() and not (MODEL_DIR / p).exists():
            status["unfinished_criteria"].append(p + " not yet produced")
    atomic_write_json(REPORT / "status.json", status)
    print(json.dumps(status, indent=2, ensure_ascii=False))
    print("DONE")


if __name__ == "__main__":
    main()
