#!/usr/bin/env python3
"""UTF-8 contract audit: roundtrip Chinese text through file + subprocess.

Writes a small machine-readable audit to reports/breakthrough_20261002/.
Checks: JSON ensure_ascii=True shell boundary roundtrip, explicit utf-8 file
roundtrip byte-for-byte, absence of U+FFFD replacement characters, and flags
likely mojibake. Does not rewrite any historical data.
"""
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "reports" / "breakthrough_20261002"
PY = str(ROOT / ".venv" / "Scripts" / "python.exe")

TITLE = "启动子图谱突破批次01：基因级表示与低代价基线"
BODY = ("本报告为中文汇总，包含汇总表、诚实结论与图表；"
        "所有文件均以 UTF-8 编码读写，未出现替换字符。")
SAMPLE = {"title": TITLE, "body": BODY}


def has_replacement(s):
    return "�" in s


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    results = {"audit": "utf8_roundtrip", "status": "PASS"}

    # 1) JSON ensure_ascii=True serialization -> bytes -> parse roundtrip
    blob = json.dumps(SAMPLE, ensure_ascii=True)
    reparsed = json.loads(blob)
    results["json_ascii_roundtrip_byte_equal"] = bool(reparsed == SAMPLE)
    results["json_ascii_contains_replacement"] = has_replacement(blob)

    # 2) explicit utf-8 file write/read roundtrip
    f = OUT / "encoding_roundtrip_sample.txt"
    f.write_text(TITLE + "\n" + BODY, encoding="utf-8")
    back = f.read_text(encoding="utf-8")
    results["file_utf8_roundtrip_byte_equal"] = bool(back == TITLE + "\n" + BODY)
    results["file_contains_replacement"] = has_replacement(back)

    # 3) subprocess boundary roundtrip (JSON stdin -> child -> JSON stdout)
    child_code = (
        "import json,sys; d=json.load(sys.stdin); "
        "sys.stdout.write(json.dumps({'echo':d}, ensure_ascii=True))"
    )
    proc = subprocess.run(
        [PY, "-X", "utf8", "-c", child_code],
        input=blob, capture_output=True, text=True, encoding="utf-8",
        env={"PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8"},
    )
    echo = json.loads(proc.stdout)
    results["subprocess_roundtrip_equal"] = bool(echo["echo"] == SAMPLE)
    results["subprocess_returncode"] = proc.returncode
    results["subprocess_stderr_has_replacement"] = has_replacement(proc.stderr)

    # 4) mojibake heuristic on the roundtripped title
    title_back = echo["echo"]["title"]
    results["title_no_replacement"] = not has_replacement(title_back)
    results["title_contains_han"] = any("一" <= c <= "鿿" for c in title_back)
    results["sample_title"] = title_back
    results["sample_body"] = echo["echo"]["body"]

    all_ok = all([
        results["json_ascii_roundtrip_byte_equal"],
        results["file_utf8_roundtrip_byte_equal"],
        results["subprocess_roundtrip_equal"],
        not results["json_ascii_contains_replacement"],
        not results["file_contains_replacement"],
        results["title_no_replacement"],
        results["title_contains_han"],
    ])
    results["status"] = "PASS" if all_ok else "FAIL"

    (OUT / "encoding_audit.json").write_text(
        json.dumps(results, ensure_ascii=True, indent=2), encoding="utf-8")
    print(json.dumps(results, ensure_ascii=True, indent=2), flush=True)


if __name__ == "__main__":
    main()

