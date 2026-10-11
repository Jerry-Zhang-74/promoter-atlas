# -*- coding: utf-8 -*-
"""batch58 driver: bounded GPU waiter -> throughput profile -> fresh Nabs resource-variant
training (controlled graceful stop) -> genuine resume continuation -> status artifacts.

Safety preconditions for ANY GPU use (protocol): the ORIGINAL healthy run's actual PID+OS
creation must be GONE, BOTH Nabs_ready.json and Njoint_ready.json must exist, and no other
NVIDIA compute process may be competing. Otherwise we report a concrete blocker and DO NOT
touch the GPU. No LLM/token polling; the waiter is a bounded local sleep loop.
"""
from __future__ import annotations
import argparse, json, os, subprocess, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
HELPER = os.path.abspath(os.path.join(HERE, "..", "batch57_model_core_rc2"))
for p in (HERE, HELPER):
    if p not in sys.path:
        sys.path.insert(0, p)

from io_utils import atomic_write_json, now_iso  # noqa: E402
import gpu_lock  # noqa: E402

ROOT = os.path.abspath(os.path.join(HERE, "..", "..", "..", ".."))
COHORT = os.path.join(ROOT, "data", "breakthrough_20261002", "batch52", "corrected", "cohort_v2")
OUT_ROOT = os.path.join(ROOT, "data", "breakthrough_20261002", "batch58", "models")
RUN_DIR = os.path.join(ROOT, "runs", "breakthrough_20261002", "batch58")
REPORTS = os.path.join(ROOT, "reports", "breakthrough_20261002", "batch58")
LEDGER = os.path.join(RUN_DIR, "exec_gate_ledger.jsonl")
READY_DIR = os.path.join(ROOT, "data", "breakthrough_20261002", "batch52", "corrected", "models")
NABS_READY = os.path.join(READY_DIR, "Nabs_ready.json")
NJOINT_READY = os.path.join(READY_DIR, "Njoint_ready.json")
SOURCE_SHA = "1bbe5db71a5f4a92a281d19567b28ca3751f93d62959e6d61bd41188dd550d96"


def _alive(pid, creation):
    if pid is None:
        return False
    if not gpu_lock._pid_exists(int(pid)):
        return False
    cur = gpu_lock._proc_creation(int(pid))
    if creation is None or cur is None:
        return True
    return int(cur) == int(creation)


def _competitors(original_pid=None, own_pids=()):
    """Competing NVIDIA compute processes worth blocking on: any python/pythonw compute
    process that is neither the original healthy run nor one of our own children. Non-python
    desktop/system GPU users (e.g. a browser) are reported but do NOT block the fit."""
    apps = []
    try:
        out = subprocess.check_output(["nvidia-smi", "--query-compute-apps=pid,process_name",
                                       "--format=csv,noheader"], timeout=10).decode().strip()
        for ln in out.splitlines():
            ln = ln.strip()
            if not ln:
                continue
            pid_s, _, name = ln.partition(",")
            try:
                pid = int(pid_s.strip())
            except ValueError:
                pid = None
            apps.append({"pid": pid, "name": name.strip()})
    except Exception:
        pass
    blocking = [x for x in apps if x["pid"] is not None
                and ("python" in x["name"].lower())
                and x["pid"] != original_pid and x["pid"] not in own_pids]
    return {"apps": apps, "blocking": blocking}


def precondition(original_pid, original_creation):
    """Return (ok, detail). OK only when original GPU run is gone, both ready files exist,
    and no NVIDIA compute process is competing."""
    d = {"checked_at": now_iso(), "original_pid": original_pid,
         "original_creation": original_creation}
    d["original_alive"] = _alive(original_pid, original_creation)
    d["nabs_ready"] = os.path.exists(NABS_READY)
    d["njoint_ready"] = os.path.exists(NJOINT_READY)
    d["compute_apps"] = _competitors(original_pid=original_pid,
                                     own_pids=(os.getpid(), os.getppid()))
    blockers = []
    grant = os.path.join(ROOT, "runs", "breakthrough_20261002", "batch52", "GPU_EVALUATION_DONE.json")
    if not os.path.isfile(grant):
        blockers.append("waiting for first-model evaluation GPU handoff: " + grant)
    if d["original_alive"]:
        blockers.append("original healthy run pid=%s creation=%s is STILL alive (must finish "
                        "Nabs then Njoint); refusing to steal the GPU"
                        % (original_pid, original_creation))
    if not d["nabs_ready"]:
        blockers.append("Nabs_ready.json absent at %s" % NABS_READY)
    if not d["njoint_ready"]:
        blockers.append("Njoint_ready.json absent at %s" % NJOINT_READY)
    if d["compute_apps"]["blocking"]:
        blockers.append("competing python compute process(es): %r" % d["compute_apps"]["blocking"])
    d["ok"] = not blockers
    d["blockers"] = blockers
    return d["ok"], d


def cmd_wait(a):
    t0 = time.time()
    last = None
    while time.time() - t0 < a.timeout:
        ok, d = precondition(a.original_pid, a.original_creation)
        last = d
        if ok:
            atomic_write_json(os.path.join(RUN_DIR, "gpu_ready.json"), d)
            print("[waiter] GPU READY:", json.dumps(d), flush=True)
            return 0
        if a.iteration_out:
            atomic_write_json(os.path.join(RUN_DIR, "gpu_blocker.json"), d)
        time.sleep(a.interval)
    if last is not None:
        atomic_write_json(os.path.join(RUN_DIR, "gpu_blocker.json"), last)
    print("[waiter] TIMEOUT after %ss; last=%s" % (a.timeout, json.dumps(last)), flush=True)
    return 2


def _py():
    return sys.executable


def _run(cmd, log_path=None):
    out = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, encoding="utf-8")
    if log_path:
        with open(log_path, "w", encoding="utf-8") as f:
            f.write("CMD: %s\n\nSTDOUT:\n%s\n\nSTDERR:\n%s\n" % (" ".join(cmd), out.stdout, out.stderr))
    return out.returncode, out.stdout, out.stderr


def _vram_mib():
    try:
        out = subprocess.check_output(["nvidia-smi", "--query-gpu=memory.used,memory.free",
                                       "--format=csv,noheader,nounits"], timeout=10).decode().strip()
        best = None
        for ln in out.splitlines():
            u, f = (float(x) for x in ln.split(","))
            if best is None or u > best[0]:
                best = (u, f)
        return best
    except Exception:
        return None


def _run_alive(cmd, log_path=None, poll_s=2.0):
    """Run cmd, sampling ACTUAL NVIDIA memory while it runs (for peak VRAM evidence)."""
    log = open(log_path, "w", encoding="utf-8") if log_path else None
    if log:
        log.write("CMD: %s\n\n" % " ".join(cmd)); log.flush()
    p = subprocess.Popen(cmd, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                         text=True, encoding="utf-8")
    peak_used, min_free = 0.0, None
    samples = 0
    while p.poll() is None:
        m = _vram_mib()
        if m:
            peak_used = max(peak_used, m[0])
            min_free = m[1] if min_free is None else min(min_free, m[1])
            samples += 1
        time.sleep(poll_s)
    out = p.stdout.read() if p.stdout else ""
    if log:
        log.write(out); log.close()
    return p.returncode, out, {"peak_used_mib": peak_used, "min_free_mib": min_free,
                               "samples": samples}


def cmd_all(a):
    ok, d = precondition(a.original_pid, a.original_creation)
    atomic_write_json(os.path.join(RUN_DIR, "precondition.json"), d)
    if not ok:
        print("[b58] BLOCKED:", json.dumps(d, indent=2), flush=True)
        return 3
    from b58_gate import actual_status
    os.makedirs(OUT_ROOT, exist_ok=True)
    model_dir = os.path.join(OUT_ROOT, "Nabs_throughput_variant")
    os.makedirs(model_dir, exist_ok=True)
    lock = gpu_lock.GpuLock(os.path.join(ROOT, "runs", "breakthrough_20261002", "batch52"))
    ident = lock.acquire("b58-throughput")
    status = {"original_run": d, "gpu_lock": ident, "begun_at": now_iso(),
              "gate_before": actual_status(LEDGER), "target_column": "Log2RPM_HEK293"}
    try:
        # ---- (1) profile (reserves b58-profile) ----
        rc, so, _se = _run([_py(), "-X", "utf8", os.path.join(HERE, "b58_probe.py"),
                            "--cohort-dir", a.cohort_dir, "--out-dir", REPORTS,
                            "--ledger", LEDGER, "--accepted-data-sha", a.source_sha,
                            "--i-understand-this-is-a-real-fit"], os.path.join(RUN_DIR, "log_profile.txt"))
        prof_path = os.path.join(REPORTS, "throughput_profile.json")
        status["profile_rc"] = rc
        if not os.path.exists(prof_path):
            status["profile_error"] = ("profile did not produce throughput_profile.json "
                                       "(rc=%s); see log_profile.txt" % rc)
            status["gate_after"] = actual_status(LEDGER)
            print("[b58] PROFILE FAILED rc=%s" % rc, flush=True)
            return 4
        prof = json.load(open(prof_path, encoding="utf-8"))
        sel = prof["selected_batch"]
        status["selected_batch"] = sel
        status["profile_selection"] = prof["selection"]
        status["profile_ok"] = rc == 0 and sel is not None
        if sel is None:
            atomic_write_json(os.path.join(REPORTS, "current_training_status.json"), status)
            return 4
        eff = int(sel)
        # ---- (2) fresh Nabs resource-variant training, controlled graceful stop ----
        # stop INSIDE epoch 0: a fifth of one epoch's optimizer steps
        n_train = _train_count(a.cohort_dir)
        steps_per_epoch = (n_train + eff - 1) // eff
        stop_steps = int(a.stop_steps) if a.stop_steps and a.stop_steps > 0 else max(1, steps_per_epoch // 5)
        status["steps_per_epoch"] = steps_per_epoch
        status["stop_steps"] = stop_steps
        init_sha = prof["init_sha"]
        rc1, so1, vram1 = _run_alive(
            [_py(), "-X", "utf8", os.path.join(HERE, "b58_core.py"),
             "--cohort-dir", a.cohort_dir, "--out-dir", model_dir, "--ledger", LEDGER,
             "--effective-batch", str(eff), "--epoch-cap", "2", "--run-id", "Nabs_throughput_variant",
             "--expected-init-sha", init_sha, "--stop-after-opt-steps", str(stop_steps),
             "--snapshot-interval-s", "60", "--accepted-data-sha", a.source_sha,
             "--result-json", os.path.join(RUN_DIR, "process1_result.json"),
             "--i-understand-this-is-a-real-fit"],
            os.path.join(RUN_DIR, "log_train1.txt"))
        r1 = _read_result(os.path.join(RUN_DIR, "process1_result.json"), so1)
        status["process1_stop_rc"] = rc1
        status["process1_vram"] = vram1
        status["process1"] = r1
        status["train_peak_used_mib"] = max(status.get("train_peak_used_mib", 0), vram1["peak_used_mib"])
        # ---- (3) genuine continuation from the newly saved b58 checkpoint (NEW process) ----
        latest = os.path.join(model_dir, "latest.json")
        rc2, so2, vram2 = _run_alive(
            [_py(), "-X", "utf8", os.path.join(HERE, "b58_core.py"),
             "--cohort-dir", a.cohort_dir, "--out-dir", model_dir, "--ledger", LEDGER,
             "--effective-batch", str(eff), "--epoch-cap", "2", "--run-id", "Nabs_throughput_variant",
             "--expected-init-sha", init_sha, "--resume-latest", latest,
             "--snapshot-interval-s", "60", "--accepted-data-sha", a.source_sha,
             "--result-json", os.path.join(RUN_DIR, "process2_result.json"),
             "--i-understand-this-is-a-real-fit"],
            os.path.join(RUN_DIR, "log_train2.txt"))
        r2 = _read_result(os.path.join(RUN_DIR, "process2_result.json"), so2)
        status["process2_resume_rc"] = rc2
        status["process2_vram"] = vram2
        status["process2"] = r2
        status["train_peak_used_mib"] = max(status.get("train_peak_used_mib", 0), vram2["peak_used_mib"])
        status["gate_after"] = actual_status(LEDGER)
        _finalize_status(status, a, prof, r1, r2)
        return 0 if (rc1 == 0 and rc2 == 0) else 5
    finally:
        lock.release()
        status["ended_at"] = now_iso()
        atomic_write_json(os.path.join(REPORTS, "current_training_status.json"), status)


def _read_result(path, stdout):
    if path and os.path.exists(path):
        try:
            return json.load(open(path, encoding="utf-8"))
        except Exception:
            pass
    return _parse_json_stdout(stdout)


def _parse_json_stdout(s):
    i = s.find('{', s.find('"arm"') - 40 if s.find('"arm"') > 40 else 0)
    if i < 0:
        i = s.find('{')
    if i < 0:
        return None
    try:
        obj, _ = json.JSONDecoder().raw_decode(s[i:])
        return obj
    except Exception:
        return None


def _train_count(cohort_dir):
    c = 0
    p = os.path.join(cohort_dir, "endpoints.tsv")
    with open(p, "r", encoding="utf-8") as f:
        hdr = f.readline().rstrip("\n").split("\t")
        si = hdr.index("split")
        for ln in f:
            parts = ln.rstrip("\n").split("\t")
            if len(parts) > si and parts[si] == "TRAIN":
                c += 1
    return c


def _gen_hashes(model_dir, run_id):
    import glob
    out = []
    for p in sorted(glob.glob(os.path.join(model_dir, run_id + "_gen*.ckpt"))):
        sp = p + ".sha256.json"
        sha = None
        if os.path.exists(sp):
            sha = json.load(open(sp, encoding="utf-8")).get("sha256")
        out.append({"path": os.path.basename(p), "sha256": sha})
    latest = os.path.join(model_dir, "latest.json")
    lp = json.load(open(latest, encoding="utf-8")) if os.path.exists(latest) else None
    return {"generations": out, "latest": lp}


def _finalize_status(status, a, prof, r1, r2):
    n = _train_count(a.cohort_dir)
    model_dir = os.path.join(OUT_ROOT, "Nabs_throughput_variant")
    hist = (r2 or {}).get("history", [])
    full_epochs = [h for h in hist if h.get("coverage_ok")]
    epoch_seconds = [h.get("epoch_seconds") for h in full_epochs if h.get("epoch_seconds")]
    # a COMPLETE epoch time: prefer epoch 1 (fully inside process 2)
    complete_epoch_s = next((h["epoch_seconds"] for h in full_epochs if h["epoch"] == 1),
                            epoch_seconds[-1] if epoch_seconds else None)
    elapsed = ((r1 or {}).get("elapsed_s") or 0) + ((r2 or {}).get("elapsed_s") or 0)
    gs = (r2 or {}).get("global_samples", 0)
    status["n_train"] = n
    status["sample_throughput_samples_per_s"] = (gs / elapsed) if elapsed else None
    status["total_training_elapsed_s"] = elapsed
    status["complete_epoch_seconds"] = complete_epoch_s
    status["honest_eta_note"] = ("all %d requested epochs completed" % (r2 or {}).get("epoch_cap", 2)
                                 if len(full_epochs) >= 2 else
                                 "INCOMPLETE: %d/2 coverage-verified epochs" % len(full_epochs))
    status["honest_eta_seconds_remaining"] = 0 if len(full_epochs) >= 2 else (
        (2 - len(full_epochs)) * complete_epoch_s if complete_epoch_s else None)
    status["peak_actual_vram_mib"] = status.get("train_peak_used_mib")
    status["selected_batch_final"] = (r2 or {}).get("effective_batch")
    status["checkpoint_hashes"] = _gen_hashes(model_dir, "Nabs_throughput_variant")
    stop = (r1 or {}).get("stop_marker") or {}
    rf = (r2 or {}).get("resumed_from") or {}
    evidence = {
        "process1_stopped": bool((r1 or {}).get("stopped")),
        "process1_stop_marker": stop,
        "process2_resumed": bool((r2 or {}).get("resumed")),
        "process2_resumed_from": rf,
        "carried_opt_steps_match": rf.get("carried_opt_steps") == stop.get("opt_steps"),
        "carried_global_samples_match": rf.get("carried_global_samples") ==
        (r1 or {}).get("global_samples"),
        "epochs_coverage_ok": [(h.get("epoch"), h.get("coverage_ok")) for h in hist],
        "global_samples_final": gs,
        "expected_samples_2_epochs": 2 * n,
        "full_pass_coverage": bool(len(full_epochs) >= 2 and gs == 2 * n),
        "restored_device": rf.get("generation") is not None,
    }
    status["resume_evidence"] = evidence
    atomic_write_json(os.path.join(REPORTS, "resume_evidence.json"), evidence)
    atomic_write_json(os.path.join(REPORTS, "coverage_proof.json"),
                      {"n_train": n, "global_samples_final": gs,
                       "expected_2_epochs": 2 * n, "epochs": hist,
                       "full_pass_coverage": evidence["full_pass_coverage"]})
    ready = {
        "batch": "batch58", "version": "v6.39.0-throughput-checkpoint.rc1",
        "variant": "throughput_variant", "target_column": "Log2RPM_HEK293",
        "selected_batch": status.get("selected_batch_final"),
        "sample_throughput_samples_per_s": status.get("sample_throughput_samples_per_s"),
        "peak_actual_vram_mib": status.get("peak_actual_vram_mib"),
        "checkpoint_latest": status["checkpoint_hashes"]["latest"],
        "full_pass_coverage": evidence["full_pass_coverage"],
        "real_resume_verified": evidence["carried_opt_steps_match"] and evidence["process2_resumed"],
        "no_performance_superiority_claim": True,
        "ready_at": now_iso(),
    }
    atomic_write_json(os.path.join(REPORTS, "THROUGHPUT_READY.json"), ready)
    b52 = os.path.join(ROOT, "runs", "breakthrough_20261002", "batch52")
    with open(os.path.join(b52, "THROUGHPUT_READY.txt"), "w", encoding="utf-8") as f:
        f.write("batch58 v6.39.0-throughput-checkpoint.rc1 THROUGHPUT_READY\n")
        f.write("variant=Nabs_throughput_variant (RESOURCE config; NOT accuracy evidence)\n")
        f.write("selected_effective_batch=%s\n" % ready["selected_batch"])
        f.write("sample_throughput=%.1f samples/s\n" % (ready["sample_throughput_samples_per_s"] or 0))
        f.write("peak_actual_vram_mib=%s\n" % ready["peak_actual_vram_mib"])
        f.write("full_pass_coverage=%s  real_resume_verified=%s\n"
                % (ready["full_pass_coverage"], ready["real_resume_verified"]))
        f.write("checkpoint_latest=%s\n" % (ready["checkpoint_latest"] or {}).get("path"))
        f.write("reports=%s\n" % REPORTS)
    return status


def cmd_status(a):
    p = os.path.join(REPORTS, "current_training_status.json")
    if os.path.exists(p):
        print(open(p, encoding="utf-8").read())
    else:
        print(json.dumps({"status": "not_started", "reports": REPORTS}, indent=2))
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description="batch58 driver (wait / all / status)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("wait", "all"):
        s = sub.add_parser(name)
        s.add_argument("--original-pid", type=int, default=54472)
        s.add_argument("--original-creation", type=int, default=None)
        s.add_argument("--cohort-dir", default=COHORT)
        s.add_argument("--source-sha", default=SOURCE_SHA)
        if name == "wait":
            s.add_argument("--timeout", type=float, default=3600.0)
            s.add_argument("--interval", type=float, default=20.0)
            s.add_argument("--iteration-out", action="store_true")
        else:
            s.add_argument("--stop-steps", type=int, default=0,
                           help="0 = auto (a fifth of one epoch's optimizer steps)")
    sub.add_parser("status")
    a = ap.parse_args(argv)
    if a.cmd == "wait":
        return cmd_wait(a)
    if a.cmd == "all":
        return cmd_all(a)
    return cmd_status(a)


if __name__ == "__main__":
    sys.exit(main())
