# -*- coding: utf-8 -*-
"""batch54 bounded LOCAL waiter for the model-A (Nabs) ready marker. ZERO LLM / paid cost.

Polls (<=30 s) for <out-root>/Nabs_ready.json until the wall-clock deadline (default
2026-10-04 12:20 Asia/Shanghai). On appearance it verifies the selected checkpoint exists
and its sidecar SHA matches (immutability), then writes ready_event.json atomically and
exits 0. On timeout it writes a deadline-risk event and exits 3. No fit / GPU / network.
"""
from __future__ import annotations
import datetime as dt, hashlib, json, os, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
EVENT = os.path.join(HERE, "modelA_wait_event.json")
LOG = os.path.join(HERE, "modelA_wait.log")


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _creation(pid):
    try:
        import ctypes
        from ctypes import wintypes
        k = ctypes.windll.kernel32
        h = k.OpenProcess(0x1000, False, int(pid))
        if not h:
            return None
        ft = wintypes.FILETIME()
        ok = k.GetProcessTimes(h, ctypes.byref(ft), ctypes.byref(ft), ctypes.byref(ft), ctypes.byref(ft))
        k.CloseHandle(h)
        return (ft.dwHighDateTime << 32) | ft.dwLowDateTime if ok else None
    except Exception:
        return None


def main():
    out_root = sys.argv[1] if len(sys.argv) > 1 else \
        "data/breakthrough_20261002/batch52/corrected/models"
    deadline = sys.argv[2] if len(sys.argv) > 2 else "2026-10-04T12:20:00+08:00"
    poll_s = float(sys.argv[3]) if len(sys.argv) > 3 else 20.0
    dl = dt.datetime.fromisoformat(deadline).timestamp()
    marker = os.path.join(out_root, "Nabs_ready.json")
    me = {"pid": os.getpid(), "creation": _creation(os.getpid()), "started": dt.datetime.now().astimezone().isoformat(),
          "out_root": os.path.abspath(out_root), "marker": os.path.abspath(marker),
          "deadline": deadline, "poll_seconds": poll_s}
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(json.dumps({"event": "waiter_start", **me}) + "\n")
    while True:
        if os.path.exists(marker):
            try:
                m = json.load(open(marker, "r", encoding="utf-8"))
                sel = m.get("selected")
                side = sel + ".sha256.json"
                sha_ok = os.path.exists(sel)
                sel_sha = sha256_file(sel) if sha_ok else None
                side_ok = False
                if os.path.exists(side):
                    sc = json.load(open(side, "r", encoding="utf-8"))
                    side_ok = sc.get("checkpoint_sha256") == sel_sha
                ev = {"event": "modelA_ready", "found_at": dt.datetime.now().astimezone().isoformat(),
                      "waiter": me, "marker": marker, "selected": sel,
                      "selected_sha256": sel_sha, "sidecar_sha_match": side_ok,
                      "marker_contents": m}
                with open(EVENT, "w", encoding="utf-8") as f:
                    json.dump(ev, f, ensure_ascii=False, indent=2)
                with open(LOG, "a", encoding="utf-8") as f:
                    f.write(json.dumps({"event": "modelA_ready", "at": ev["found_at"], "selected": sel}) + "\n")
                return 0
            except Exception as e:
                with open(LOG, "a", encoding="utf-8") as f:
                    f.write(json.dumps({"event": "marker_present_unreadable", "err": repr(e),
                                        "at": dt.datetime.now().astimezone().isoformat()}) + "\n")
        if time.time() >= dl:
            ev = {"event": "deadline_risk_no_marker", "at": dt.datetime.now().astimezone().isoformat(),
                  "waiter": me, "marker": marker}
            with open(EVENT, "w", encoding="utf-8") as f:
                json.dump(ev, f, ensure_ascii=False, indent=2)
            with open(LOG, "a", encoding="utf-8") as f:
                f.write(json.dumps(ev) + "\n")
            return 3
        time.sleep(poll_s)


if __name__ == "__main__":
    sys.exit(main())
