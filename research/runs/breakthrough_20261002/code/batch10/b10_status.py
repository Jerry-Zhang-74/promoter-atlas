#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Batch10 status.json writer (startup and scientific-stage refreshes).

Real clock time and process identity (PID + creation time via os, no secrets).
Usage: python b10_status.py <stage> <note>
"""
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from b10_common import REPORT_OUT, CANDIDATE_VERSION, BATCH, atomic_write_json  # noqa: E402


def main():
    stage = sys.argv[1] if len(sys.argv) > 1 else "startup"
    note = sys.argv[2] if len(sys.argv) > 2 else ""
    try:
        import psutil  # optional
        proc = psutil.Process(os.getpid())
        create_time = time.strftime("%Y-%m-%dT%H:%M:%S%z", time.localtime(proc.create_time()))
    except Exception:
        create_time = None
    status = {
        "candidate_version": CANDIDATE_VERSION,
        "batch": BATCH,
        "stage": stage,
        "note": note,
        "written_at_local": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "pid": os.getpid(),
        "process_create_time_local": create_time,
    }
    atomic_write_json(REPORT_OUT / "status.json", status)
    print(json.dumps(status, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
