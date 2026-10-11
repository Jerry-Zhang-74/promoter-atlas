#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Batch13 status.json writer.

Records the WORKER identity (long-lived PID supplied by the controller control_19
status.json) with its creation time verified via Windows GetProcessTimes
(ctypes), NOT os.getpid() of a transient status-writer subprocess. The writer
subprocess identity is kept in a separate field. Real clock only, no future
timestamps.

Usage: python b13_status.py <stage> [note]
"""
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from b13_common import REPORT_OUT, CANDIDATE_VERSION, BATCH, atomic_write_json  # noqa: E402

CONTROL_19_STATUS = Path(__file__).resolve().parents[2] / "control_19" / "status.json"


def windows_process_creation_time(pid):
    if os.name != "nt":
        return None
    import ctypes
    from ctypes import wintypes
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    PROCESS_QUERY_LIMITED_INFORMATION = 0x1000

    class FILETIME(ctypes.Structure):
        _fields_ = [("dwLowDateTime", wintypes.DWORD),
                    ("dwHighDateTime", wintypes.DWORD)]

    h = kernel32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, int(pid))
    if not h:
        return None
    ct = FILETIME(); et = FILETIME(); kt = FILETIME(); ut = FILETIME()
    ok = kernel32.GetProcessTimes(h, ctypes.byref(ct), ctypes.byref(et),
                                  ctypes.byref(kt), ctypes.byref(ut))
    kernel32.CloseHandle(h)
    if not ok:
        return None
    t = (ct.dwHighDateTime << 32) | ct.dwLowDateTime
    return t / 1e7 - 11644473600.0


def fmt_local(epoch_s):
    if epoch_s is None:
        return None
    lt = time.localtime(epoch_s)
    return time.strftime("%Y-%m-%dT%H:%M:%S", lt) + (".%06d+08:00" % int((epoch_s % 1) * 1e6))


def main():
    stage = sys.argv[1] if len(sys.argv) > 1 else "startup"
    note = sys.argv[2] if len(sys.argv) > 2 else ""

    worker_pid = None
    if CONTROL_19_STATUS.exists():
        ctl = json.loads(CONTROL_19_STATUS.read_text(encoding="utf-8"))
        worker_pid = ctl.get("worker", {}).get("pid")

    writer_pid = os.getpid()
    worker_ct = windows_process_creation_time(worker_pid) if worker_pid else None
    writer_ct = windows_process_creation_time(writer_pid)

    status = {
        "candidate_version": CANDIDATE_VERSION,
        "batch": BATCH,
        "stage": stage,
        "note": note,
        "written_at_local": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "worker": {
            "pid": worker_pid,
            "pid_source": "control_19 status.json worker.pid",
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
