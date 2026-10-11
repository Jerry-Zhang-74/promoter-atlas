#!/usr/bin/env python3
"""Batch02 UTF-8 encoding audit: Chinese roundtrip + subprocess roundtrip, no U+FFFD."""
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
REPORT = ROOT / "reports" / "breakthrough_20261002" / "batch02"
PY = ROOT / ".venv" / "Scripts" / "python.exe"

TITLE = "突破性实验批次02报告：基因级表达预测模型的可加载交付与MPRA多成员家族鲁棒性验证"
SECTIONS = [
    "本批次对批次01的两个候选增益进行纠正与加固：",
    "任务A：将模型家族与正则化系数的选择全部改为仅使用46618个训练基因的五折内部交叉验证，",
    "并交付一个可独立加载、无需目标标签即可推理的基因级Score_E预测模型。",
    "任务B：在批次01训练分区上做五折分组交叉验证，检验多成员家族上的泛化，",
    "不追求强行成功，如实报告未证实或负向结果。",
]


def main():
    REPORT.mkdir(parents=True, exist_ok=True)
    chinese_sample = "\n".join(SECTIONS)
    checks = {}

    # 1) JSON serialization roundtrip (ensure_ascii=True then decode)
    payload = {"title": TITLE, "sections": SECTIONS}
    js = json.dumps(payload, ensure_ascii=True)
    back = json.loads(js)
    checks["json_roundtrip"] = {"title_exact": back["title"] == TITLE,
                                "sections_exact": back["sections"] == SECTIONS}

    # 2) UTF-8 file roundtrip byte-for-byte
    sample_path = REPORT / "encoding_roundtrip_sample.txt"
    sample_path.write_text(chinese_sample, encoding="utf-8")
    read_back = sample_path.read_text(encoding="utf-8")
    checks["file_roundtrip_bytes_exact"] = read_back == chinese_sample

    # 3) no U+FFFD anywhere
    has_replacement = "�" in TITLE or any("�" in s for s in SECTIONS)
    checks["no_uFFFD"] = not has_replacement

    # 4) subprocess stdout roundtrip (PYTHONIOENCODING=utf-8)
    prog = "import sys; t = " + repr(TITLE) + "; sys.stdout.write(t)"
    env = {"PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8"}
    r = subprocess.run([str(PY), "-X", "utf8", "-c", prog], capture_output=True, text=True,
                       encoding="utf-8", env=env)
    checks["subprocess_stdout_roundtrip"] = {"exact": r.stdout == TITLE,
                                             "no_uFFFD": "�" not in r.stdout,
                                             "returncode": r.returncode}

    audit = {
        "title": TITLE,
        "checks": checks,
        "all_pass": bool(all(v if isinstance(v, bool) else v.get("exact", True) for v in checks.values())
                         and all(v.get("no_uFFFD", True) for v in checks.values() if isinstance(v, dict))),
        "note": "all text files written with encoding='utf-8'; JSON via ensure_ascii=True across process boundaries",
    }
    audit["all_pass"] = bool(
        checks["json_roundtrip"]["title_exact"] and checks["json_roundtrip"]["sections_exact"]
        and checks["file_roundtrip_bytes_exact"] and checks["no_uFFFD"]
        and checks["subprocess_stdout_roundtrip"]["exact"] and checks["subprocess_stdout_roundtrip"]["no_uFFFD"]
        and checks["subprocess_stdout_roundtrip"]["returncode"] == 0)
    (REPORT / "encoding_audit.json").write_text(
        json.dumps(audit, indent=2, ensure_ascii=True), encoding="utf-8")
    print(json.dumps(audit, indent=2, ensure_ascii=True), flush=True)
    print("ENCODING", "PASS" if audit["all_pass"] else "FAIL", flush=True)


if __name__ == "__main__":
    main()
