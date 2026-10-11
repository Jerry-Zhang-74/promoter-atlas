# -*- coding: utf-8 -*-
"""GPU-exclusive process lock with PID + stable OS process-CREATION identity.

Main models never run two competing training jobs on one RTX5060. A short replay evaluator
may take the lock for <=3 min, then MUST release before Njoint trains, and MUST NOT hold it
during CPU work. Identity = (pid, process creation time) so a recycled PID cannot masquerade
as the holder.

BATCH57 REPAIR D1 (2026-10-04): the batch53 version called GetProcessTimes with ONE shared
FILETIME object for all four out-parameters, so _proc_creation() returned kernel/user CPU
time (which changes within a process lifetime) instead of the creation time. That made
_alive(live_holder) False and let a second waiter reclaim a lock held by a live process.
Fixed by using FOUR DISTINCT FILETIME buffers and reading only lpCreationTime. The lock now
NEVER reclaims a lock whose holder pid is alive and whose creation time cannot be disproved.

Portability: on non-Windows the creation time is read from /proc/<pid>/stat field 22.
"""
from __future__ import annotations
import errno, json, os, time

from io_utils import now_iso

LOCK_FILE = "gpu_exclusive.lock"
HANDOFF_TIMEOUT_S = 3 * 60

_PROCESS_QUERY_LIMITED_INFORMATION = 0x1000


def _ft_to_int(ft) -> int:
    return (int(ft.dwHighDateTime) << 32) | int(ft.dwLowDateTime)


def _posix_creation(pid: int):
    """Linux: field 22 (starttime, clock ticks since boot) from /proc/<pid>/stat."""
    try:
        with open("/proc/%d/stat" % int(pid), "r", encoding="utf-8") as f:
            data = f.read()
        rp = data.rfind(")")
        fields = data[rp + 2:].split()
        return int(fields[19])          # field 22 overall == index 19 after comm
    except Exception:
        return None


def _proc_creation(pid: int):
    """Stable OS process CREATION time (Windows 100ns since 1601; Linux starttime ticks).

    Returns None when the process cannot be queried (usually: it no longer exists)."""
    if os.name != "nt":
        return _posix_creation(pid)
    try:
        import ctypes
        from ctypes import wintypes
        k = ctypes.windll.kernel32
        h = k.OpenProcess(_PROCESS_QUERY_LIMITED_INFORMATION, False, int(pid))
        if not h:
            return None
        try:
            creation = wintypes.FILETIME()
            exit_time = wintypes.FILETIME()
            kernel_time = wintypes.FILETIME()
            user_time = wintypes.FILETIME()
            ok = k.GetProcessTimes(h, ctypes.byref(creation), ctypes.byref(exit_time),
                                   ctypes.byref(kernel_time), ctypes.byref(user_time))
            if not ok:
                return None
            return _ft_to_int(creation)
        finally:
            k.CloseHandle(h)
    except Exception:
        return None


def _pid_exists(pid: int) -> bool:
    if pid is None or int(pid) <= 0:
        return False
    if os.name == "nt":
        try:
            import ctypes
            from ctypes import wintypes
            k = ctypes.windll.kernel32
            h = k.OpenProcess(_PROCESS_QUERY_LIMITED_INFORMATION, False, int(pid))
            if not h:
                return False
            try:
                # A terminated process whose Popen handle is still open remains queryable and
                # still reports its creation time. GetExitCodeProcess distinguishes a LIVE
                # process (STILL_ACTIVE=259) from an exited one.
                code = wintypes.DWORD()
                if not k.GetExitCodeProcess(h, ctypes.byref(code)):
                    return False
                return int(code.value) == 259
            finally:
                k.CloseHandle(h)
        except Exception:
            return False
    try:
        os.kill(int(pid), 0)
        return True
    except OSError as e:
        return e.errno == errno.EPERM


def my_identity() -> dict:
    pid = os.getpid()
    return {"pid": pid, "creation": _proc_creation(pid), "acquired_at": now_iso()}


def _alive(holder: dict) -> bool:
    """True iff the recorded holder is still the SAME live process.

    Fail-safe by design: we never reclaim a lock whose holder pid is alive and whose
    creation time we cannot DISPROVE (missing / unreadable holder => assume held)."""
    if not holder:
        return False
    if holder.get("unreadable"):
        return True                     # concurrent writer mid-create -> assume held
    pid = holder.get("pid")
    if not isinstance(pid, int) or pid <= 0:
        return False
    if not _pid_exists(pid):
        return False                    # process is gone (or exited with a still-open handle)
    cur = _proc_creation(pid)
    if cur is None:
        # Alive but creation unreadable: we must NOT reclaim it.
        return True
    hc = holder.get("creation")
    if hc is None:
        return True                     # live pid with no recorded creation -> assume holder
    try:
        return int(hc) == int(cur)
    except (TypeError, ValueError):
        return True


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
                txt = f.read().strip()
        except Exception:
            return {"pid": None, "creation": None, "unreadable": True}
        if not txt:
            return {"pid": None, "creation": None, "unreadable": True}
        try:
            obj = json.loads(txt)
            if not isinstance(obj, dict):
                return {"pid": None, "creation": None, "unreadable": True}
            return obj
        except Exception:
            return {"pid": None, "creation": None, "unreadable": True}

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
                # Only reclaim when the holder is provably gone or its PID was recycled.
                try:
                    os.remove(self.path)   # stale holder -> reclaim
                    continue
                except OSError:
                    pass
            if time.time() - t0 > timeout_s:
                raise TimeoutError("GPU lock handoff timeout; holder=%r" % (h,))
            time.sleep(0.1)

    def release(self) -> None:
        """Release ONLY our own identity (matching pid AND creation)."""
        if not self.held:
            return
        h = self.holder()
        mine = my_identity()
        same_pid = bool(h) and h.get("pid") == mine["pid"]
        same_creation = same_pid and (h.get("creation") is None or h.get("creation") == mine["creation"])
        if same_pid and same_creation:
            try:
                os.remove(self.path)
            except FileNotFoundError:
                pass
        self.held = False
