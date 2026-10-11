#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Batch11 status.json writer (startup and scientific-stage refreshes).

Records the WORKER identity (long-lived PID supplied by the controller) with its
creation time verified via Windows GetProcessTimes (ctypes), NOT os.getpid() of a
transient status-writer subprocess. The status-writer subprocess identity is kept
in a separately named field and is not confused with the worker. Real clock only.

Usage: python b11_status.py <stage> [note] [--worker-pid N]
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from b11_common import (REPORT_OUT, CANDIDATE_VERSION, BATCH, atomic_write_json,
                        windows_process_creation_time, fmt_local, now_local)  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", nargs="?", default="startup")
    ap.add_argument("note", nargs="?", default="")
    ap.add_argument("--worker-pid", type=int, default=23116)
    args = ap.parse_args()

    writer_pid = os.getpid()
    worker_ct = windows_process_creation_time(args.worker_pid)
    writer_ct = windows_process_creation_time(writer_pid)

    status = {
        "candidate_version": CANDIDATE_VERSION,
        "batch": BATCH,
        "stage": args.stage,
        "note": args.note,
        "written_at_local": now_local(),
        "worker": {
            "pid": args.worker_pid,
            "pid_source": "supplied by controller (control_17 worker pid)",
            "process_create_time_local": fmt_local(worker_ct),
            "creation_verified_via": "Windows GetProcessTimes (ctypes)",
        },
        "statewriter": {
            "pid": writer_pid,
            "process_create_time_local": fmt_local(writer_ct),
            "role": "transient status-writer subprocess; NOT the worker",
        },
        "distinction_note": "worker creation time is distinct from controller started_at and from this writer subprocess",
    }
    atomic_write_json(REPORT_OUT / "status.json", status)
    print(json.dumps(status, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

