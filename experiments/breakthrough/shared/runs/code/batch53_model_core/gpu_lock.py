# -*- coding: utf-8 -*-
"""GPU-exclusive process lock with PID + process-creation identity (one RTX5060 cuda:0).

Main models never run two competing training jobs. A short replay evaluator may take the
lock for <=3 min, then MUST release before Njoint trains, and MUST NOT hold it during CPU
work. Identity = (pid, process creation time) so a recycled PID cannot masquerade as the
holder. Bounded handoff: waiter times out rather than blocking forever.
"""
from __future__ import annotations
import errno, json, os, time

from io_utils import atomic_write_json, now_iso

LOCK_FILE = "gpu_exclusive.lock"
HANDOFF_TIMEOUT_S = 3 * 60


def _proc_creation(pid: int):
    if os.name != "nt":
        return None
    try:
        import ctypes
        from ctypes import wintypes
        k = ctypes.windll.kernel32
        h = k.OpenProcess(0x1000, False, int(pid))  # PROCESS_QUERY_LIMITED_INFORMATION
        if not h:
            return None
        ft = wintypes.FILETIME()
        if k.GetProcessTimes(h, ctypes.byref(ft), ctypes.byref(ft), ctypes.byref(ft), ctypes.byref(ft)):
            k.CloseHandle(h)
            return (ft.dwHighDateTime << 32) | ft.dwLowDateTime
        k.CloseHandle(h)
    except Exception:
        return None
    return None


def my_identity() -> dict:
    pid = os.getpid()
    return {"pid": pid, "creation": _proc_creation(pid), "acquired_at": now_iso()}


def _alive(holder: dict) -> bool:
    pid = holder.get("pid")
    if pid is None:
        return False
    c = _proc_creation(pid)
    if c is None:
        return False
    return holder.get("creation") in (None, c)


class GpuLock:
    def __init__(self, lock_dir: str):
        self.path = os.path.join(lock_dir, LOCK_FILE)
        os.makedirs(lock_dir, exist_ok=True)
        self.held = False

    def holder(self):
        if not os.path.exists(self.path):
            return None
        try:
            with open(self.path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {"pid": None, "creation": None}

    def acquire(self, purpose: str, timeout_s: float = HANDOFF_TIMEOUT_S) -> dict:
        t0 = time.time()
        while True:
            try:
                fd = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                ident = my_identity()
                ident["purpose"] = purpose
                os.write(fd, json.dumps(ident).encode("utf-8"))
                os.close(fd)
                self.held = True
                return ident
            except OSError as e:
                if e.errno not in (errno.EEXIST, errno.EACCES, errno.EPERM):
                    raise
            h = self.holder()
            if h is not None and not _alive(h):
                try:
                    os.remove(self.path)   # stale holder -> reclaim
                    continue
                except OSError:
                    pass
            if time.time() - t0 > timeout_s:
                raise TimeoutError("GPU lock handoff timeout; holder=%r" % (h,))
            time.sleep(0.1)

    def release(self) -> None:
        if not self.held:
            return
        h = self.holder()
        if h and h.get("pid") == os.getpid():
            try:
                os.remove(self.path)
            except FileNotFoundError:
                pass
        self.held = False
