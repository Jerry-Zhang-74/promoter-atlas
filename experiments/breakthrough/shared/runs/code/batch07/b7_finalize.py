#!/usr/bin/env python3
"""Batch07 finalize: status.json, sha_manifest, version_registry, README."""
import sys
import os
import json
import datetime
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from b7_common import REPORT, MODEL_DIR, atomic_write_json, sha256_file, CANDIDATE  # noqa: E402

CODE = Path(__file__).resolve().parent


def find_claude_pid():
    try:
        r = subprocess.run(["tasklist"], capture_output=True, timeout=20)
        out = r.stdout.decode("utf-8", errors="replace")
        for line in out.splitlines():
            if "claude.exe" in line.lower():
                parts = line.split()
                if len(parts) >= 2:
                    return int(parts[1])
    except Exception:
        pass
    return None


def main():
    now = datetime.datetime.now().astimezone().isoformat()
    writer_pid = os.getpid()
    claude_pid = find_claude_pid()

    evalj = json.loads((REPORT / "eval_covergate_b7.json").read_text(encoding="utf-8"))
    acc = evalj["acceptance"]
    t = evalj["by_split"]["test"]["hgb_primary_gate"]
    v = evalj["by_split"]["val"]["hgb_primary_gate"]
    rmb = evalj["by_split"]["test"]["random_mask_baseline"]

    val_rel_mae = (v["full"]["mae"] - v["retained"]["mae"]) / v["full"]["mae"]

    status = {
        "batch": "batch07",
        "phase": "worker",
        "candidate_version": CANDIDATE,
        "state": "complete",
        "started_at": "2026-10-03T04:34:00+08:00",
        "updated_at": now,
        "active_worker": {
            "pid": claude_pid if claude_pid else writer_pid,
            "process_name": "claude" if claude_pid else "python",
            "role": ("Claude Code worker harness" if claude_pid else "status writer (short-lived)"),
            "created_at": now,
            "note": ("worker phase COMPLETE; PID observed via tasklist at write time; "
                     "distinguishes the finished batch07 worker from this short-lived "
                     "finalize/status writer"),
        },
        "summary_zh": (
            "batch07 完成：在冻结 batch05 划分与 batch06 features45 上，仅用 outer-train 45 维特征拟合"
            "固定覆盖门控（StandardScaler + LedoitWolf，平方马氏距离 <= train 90% 分位），对冻结 HGB "
            "预测器在 test 上覆盖率 90.23%，但相对 MAE 下降仅 1.40%（未达 5% 阈值），组件 bootstrap "
            "95% 下界 0.000854>0。主接受判定 NEGATIVE（acceptance_threshold_met=false），已如实保留，"
            "未重扫、未事后调阈值。"
        ),
        "key_results": {
            "test_coverage": acc["coverage"],
            "test_relative_mae_reduction": acc["relative_mae_reduction"],
            "test_component_bootstrap_lb": acc["component_bootstrap_lb"],
            "test_full_mae": acc["all_mae"],
            "test_retained_mae": acc["retained_mae"],
            "test_full_spearman": t["full"]["spearman"],
            "test_retained_spearman": t["retained"]["spearman"],
            "val_coverage": v["coverage"],
            "val_relative_mae_reduction": val_rel_mae,
            "random_mask_gate_percentile": rmb["gate_percentile_vs_random"],
            "acceptance_threshold_met": acc["acceptance_threshold_met"],
            "result": "NEGATIVE",
        },
        "next_action": (
            "交付物已落盘：中文报告 report.txt/report.html、figures、raw membership TSV、"
            "next_batch_proposal.json、BATCH08_DRAFT.txt。等待独立审查；科学门控接受为 NEGATIVE，"
            "但项目持续，不停止。"
        ),
        "current_command": "全部完成；无运行中命令。",
        "experiment_complete": True,
        "checkpoint": (
            "preflight PASS + geometry 冻结(SHA c70de2c9) + 评估(acceptance_threshold_met=false) + "
            "推断包装 v6.8.0-covergate.rc1 + 26 项测试全 PASS + 报告 + 提案全部落盘；NEGATIVE 保留。"
        ),
    }
    atomic_write_json(REPORT / "status.json", status)

    # sha manifest (code + report artifacts + figures + model geometry)
    files = {}
    for p in sorted(CODE.glob("b7_*.py")):
        files["runs/breakthrough_20261002/code/batch07/" + p.name] = sha256_file(p)
    for p in sorted((REPORT / "figures").glob("*.png")):
        files["reports/breakthrough_20261002/batch07/figures/" + p.name] = sha256_file(p)
    for name in ["preflight_b7.json", "frozen_config_b7.json", "eval_covergate_b7.json",
                 "report.txt", "report.html", "status.json", "next_batch_proposal.json",
                 "BATCH08_DRAFT.txt"]:
        q = REPORT / name
        if q.exists():
            files["reports/breakthrough_20261002/batch07/" + name] = sha256_file(q)
    q = REPORT / "predictions" / "pred_v6.8.0-covergate.rc1.tsv"
    if q.exists():
        files["reports/breakthrough_20261002/batch07/predictions/pred_v6.8.0-covergate.rc1.tsv"] = sha256_file(q)
    for name in ["covergate_geometry.npz", "covergate_geometry.json"]:
        q = MODEL_DIR / name
        if q.exists():
            files["models/breakthrough_20261002/batch07/" + name] = sha256_file(q)
    atomic_write_json(REPORT / "sha_manifest_b7.json", files)

    # version registry entry for v6.8.0-covergate.rc1
    registry = {
        "candidate_version": CANDIDATE,
        "track": "gene-level coverage gate (Score_E, frozen HGB)",
        "task": "train-only feature-coverage rule applicability diagnostic",
        "parent_model_version": "batch06 HGB",
        "parent_model_sha": "994db28e75a42e9bef239e5f854d1352aeefa89ebaa66676c0c1e652d2e726c6",
        "split_sha": "4fbfccf43038fbe2211c2a21f113b11b38d92cf410bc3591c6bb3e3a42d8a6f5",
        "features45_sha": "c776ecfa74163b5000de8f82aa8961b5aed154f80333d95665530947694c0800",
        "geometry_sha256": evalj["geometry_sha256"],
        "evaluation_protocol_version": "batch07/PROTOCOL.txt v1",
        "code_sha_manifest": "reports/breakthrough_20261002/batch07/sha_manifest_b7.json",
        "metrics": {
            "test_coverage": acc["coverage"],
            "test_relative_mae_reduction": acc["relative_mae_reduction"],
            "test_component_bootstrap_lb": acc["component_bootstrap_lb"],
        },
        "review_status": "pending independent review",
        "decision": "scientific gate acceptance NEGATIVE (acceptance_threshold_met=false)",
        "status": "rejected_for_promotion",
        "created_at": now,
    }
    atomic_write_json(REPORT / "version_registry_b7.json", registry)

    # reproducible artifact README (no false probabilities)
    readme = []
    readme.append("=" * 78)
    readme.append("batch07 可复现工件 README —— v6.8.0-covergate.rc1")
    readme.append("=" * 78)
    readme.append("科学结论: 主接受判定 NEGATIVE（覆盖门控在 90% 覆盖率下相对 MAE 下降 1.40%，未达 5% 阈值）。")
    readme.append("")
    readme.append("复现顺序（全部 CPU，无 GPU，无预测器重训）:")
    readme.append("  1) .venv/Scripts/python.exe -X utf8 runs/breakthrough_20261002/code/batch07/b7_preflight.py")
    readme.append("  2) .venv/Scripts/python.exe -X utf8 runs/breakthrough_20261002/code/batch07/b7_geometry.py")
    readme.append("  3) .venv/Scripts/python.exe -X utf8 runs/breakthrough_20261002/code/batch07/b7_evaluate.py")
    readme.append("  4) .venv/Scripts/python.exe -X utf8 runs/breakthrough_20261002/code/batch07/b7_inference.py")
    readme.append("  5) .venv/Scripts/python.exe -X utf8 runs/breakthrough_20261002/code/batch07/b7_tests.py")
    readme.append("  6) .venv/Scripts/python.exe -X utf8 runs/breakthrough_20261002/code/batch07/b7_figures.py")
    readme.append("  7) .venv/Scripts/python.exe -X utf8 runs/breakthrough_20261002/code/batch07/b7_report.py")
    readme.append("  8) .venv/Scripts/python.exe -X utf8 runs/breakthrough_20261002/code/batch07/b7_proposal.py")
    readme.append("  9) .venv/Scripts/python.exe -X utf8 runs/breakthrough_20261002/code/batch07/b7_finalize.py")
    readme.append("")
    readme.append("输入（冻结，只读）:")
    readme.append("  split_arrays.npz SHA: 4fbfccf4...42d8a6f5")
    readme.append("  features45.npz SHA: c776ecfa...94c0800")
    readme.append("  hgb_baseline.joblib SHA: 994db28e...d2e726c6")
    readme.append("  finalridge.npz SHA: 0e0825d1...1f230fcc")
    readme.append("")
    readme.append("门控几何（本批冻结）:")
    readme.append("  covergate_geometry.npz SHA: " + evalj["geometry_sha256"])
    readme.append("  主阈值（train 90% 分位, 线性插值）= " + ("%.6f" % evalj["primary_gate"]["threshold"]))
    readme.append("")
    readme.append("重要说明（无虚假概率）:")
    readme.append("  - 覆盖分数 = 特征中心性（平方马氏距离），不是校准概率，不是完整密度/OOD 保证。")
    readme.append("  - 保留子集 vs 全样本是不同人群；子集相关性更高不等于模型更好。")
    readme.append("  - 现有 test 被反复复用 => 回顾性/探索性，绝不宣称前瞻性。")
    readme.append("  - 门控未命中不等于不可靠，也不等于生物学分布外。")
    readme.append("  - 审查完整性 PASS 与科学门控接受是两回事。")
    readme.append("=" * 78)
    (REPORT / "README.txt").write_text("\n".join(readme) + "\n", encoding="utf-8")

    print("finalized: status.json, sha_manifest_b7.json, version_registry_b7.json, README.txt")
    print("claude_pid=", claude_pid, "writer_pid=", writer_pid)


if __name__ == "__main__":
    main()
