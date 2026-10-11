# -*- coding: utf-8 -*-
"""batch58 throughput + memory profiler for the true model shape-600, fp32, no AMP.

Measures COMPLETE forward+backward (NO optimizer step) for candidate effective batches
32..2048 with the resident TRAIN cohort codes tensor on the GPU, so the memory numbers
include the real resident cohort. Timing uses torch.cuda.synchronize(); memory uses BOTH
torch (allocated/reserved/peak) and actual NVIDIA samples (nvidia-smi). NO DEV/TEST score is
read; batch selection is resource-only.

Hard rules: this is execution entry (1) of the b58 cap-3 allowance -> it reserves a
CONSUMING gate slot BEFORE the first backward. No optimizer step. Initial weights/RNG are
restored before each candidate for fairness. Bounded <=180 s total.
"""
from __future__ import annotations
import json, os, subprocess, sys, time

import numpy as np
import torch

HELPER = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "batch57_model_core_rc2"))
if HELPER not in sys.path:
    sys.path.insert(0, HELPER)
# corrected cohort_v2 implementation lives in the batch52 data workstream (read-only import)
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
DATACODE = os.path.join(_ROOT, "code", "batch52_data_v2")
if os.path.isdir(DATACODE) and DATACODE not in sys.path:
    sys.path.insert(0, DATACODE)

import losses  # noqa: E402
from checkpoint import capture_rng, restore_rng  # noqa: E402
from io_utils import atomic_write_json, now_iso  # noqa: E402
from model_core import build_model, param_sha256  # noqa: E402
from spec import SEED, config_sha256  # noqa: E402
from train_neural import epoch_order, compute_gc  # noqa: E402
from normalizer import TargetNormalizer  # noqa: E402

CANDIDATES = (32, 64, 128, 256, 512, 1024, 2048)
WARMUP = 1
REPEATS = 3
TIME_BUDGET_S = 180.0
TARGET_USED_GIB = (6.7, 7.1)
MIN_FREE_GIB = 0.7


def _nvidia_mem():
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.total,memory.used,memory.free",
             "--format=csv,noheader,nounits"], timeout=10).decode().strip().splitlines()
        best = None
        for ln in out:
            t, u, f = (float(x) for x in ln.split(","))
            if best is None or u > best[1]:
                best = (t, u, f)
        return {"total_mib": best[0], "used_mib": best[1], "free_mib": best[2]} if best else None
    except Exception:
        return None


def _nvidia_compute_pids():
    try:
        out = subprocess.check_output(["nvidia-smi", "--query-compute-apps=pid,process_name",
                                       "--format=csv,noheader"], timeout=10).decode().strip()
        return [ln.strip() for ln in out.splitlines() if ln.strip()]
    except Exception:
        return []


def _mem_gib(mib):
    return None if mib is None else mib / 1024.0


def run_profile(cohort, out_dir, ledger_path, device="cuda:0", opt_in=False,
                accepted_data_sha=None, candidates=CANDIDATES, time_budget_s=TIME_BUDGET_S,
                reserve_slot=True):
    os.makedirs(out_dir, exist_ok=True)
    token = None
    gate = None
    if reserve_slot:
        if not opt_in:
            raise PermissionError("profile refused: pass --i-understand-this-is-a-real-fit")
        if not accepted_data_sha or accepted_data_sha != cohort.source_sha256:
            raise PermissionError("profile refused: accepted actual-data sha required")
        from b58_gate import open_ledger, reserve
        gate = open_ledger(ledger_path)
        token = reserve(gate, "b58-profile")   # CONSUMING, BEFORE any backward

    t0 = time.time()
    try:
        model = build_model(SEED).to(device)
        init_sha = param_sha256(model)
        tr, wtr = cohort.activity_weights("TRAIN")
        tr_codes = cohort.codes[tr].to(device)          # RESIDENT cohort tensor
        gc_tr = compute_gc(tr_codes.cpu())
        model.pool_stats.fit(cohort.lengths[tr], gc_tr)
        norm = TargetNormalizer.fit(cohort.targets[tr], wtr)
        tr_yz = norm.forward(cohort.targets[tr]).to(device)
        tr_w = wtr.to(device)
        n = int(tr.numel())
        perm = epoch_order(n, 0).tolist()

        base_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        base_rng = capture_rng()
        model.train()

        resident = _nvidia_mem()
        results = []
        for b in candidates:
            if time.time() - t0 > time_budget_s:
                results.append({"batch": b, "skipped": "time_budget_exceeded"})
                break
            model.load_state_dict(base_state)
            restore_rng(base_rng)
            model.zero_grad(set_to_none=True)
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            idx = perm[:b]
            tt = torch.tensor(idx, dtype=torch.long)
            codes_b = tr_codes[tt]
            y_b = tr_yz[tt]
            w_b = tr_w[tt]
            rec = {"batch": b}
            peak_nv_used = 0.0
            peak_nv_free = None
            try:
                for w in range(WARMUP):
                    model.zero_grad(set_to_none=True)
                    loss, _ = losses.total_loss(model(codes_b), y_b, w_b)
                    loss.backward()
                torch.cuda.synchronize()
                times = []
                for r in range(REPEATS):
                    model.zero_grad(set_to_none=True)
                    torch.cuda.synchronize()
                    t = time.perf_counter()
                    loss, _ = losses.total_loss(model(codes_b), y_b, w_b)
                    loss.backward()
                    torch.cuda.synchronize()
                    times.append(time.perf_counter() - t)
                    nv = _nvidia_mem()
                    if nv:
                        peak_nv_used = max(peak_nv_used, nv["used_mib"])
                        peak_nv_free = nv["free_mib"] if peak_nv_free is None else min(peak_nv_free, nv["free_mib"])
                alloc = torch.cuda.memory_allocated() / (1024 ** 3)
                resv = torch.cuda.memory_reserved() / (1024 ** 3)
                peak = torch.cuda.max_memory_allocated() / (1024 ** 3)
                med = float(np.median(times))
                rec.update({"ok": True, "times_s": times, "median_s": med,
                            "samples_per_s": b / med if med > 0 else None,
                            "torch_allocated_gib": alloc, "torch_reserved_gib": resv,
                            "torch_peak_gib": peak,
                            "nvidia_used_gib": _mem_gib(peak_nv_used),
                            "nvidia_free_gib": _mem_gib(peak_nv_free)})
                model.zero_grad(set_to_none=True)
            except torch.cuda.OutOfMemoryError as e:
                rec.update({"ok": False, "oom": True, "error": str(e)[:200]})
                try:
                    model.zero_grad(set_to_none=True)
                except Exception:
                    pass
                torch.cuda.empty_cache()
            results.append(rec)
            model.load_state_dict(base_state)
            restore_rng(base_rng)
            torch.cuda.empty_cache()

        sel = select_batch(results, time_budget_s)
        out = {"probe": "batch58-throughput-profile", "device": device,
               "resident_cohort_gib": (tr_codes.numel() * tr_codes.element_size()) / (1024 ** 3),
               "nvidia_total_mib": None if resident is None else resident["total_mib"],
               "nvidia_used_before_mib": None if resident is None else resident["used_mib"],
               "nvidia_free_before_mib": None if resident is None else resident["free_mib"],
               "nvidia_compute_pids_before": _nvidia_compute_pids(),
               "init_sha": init_sha, "config_sha256": config_sha256(),
               "warmup": WARMUP, "repeats": REPEATS, "candidates": list(candidates),
               "results": results, "selection": sel, "selected_batch": sel["selected_batch"],
               "elapsed_s": time.time() - t0, "generated_at": now_iso(),
               "no_optimizer_step": True, "no_amp": True, "fp32": True,
               "note": "resource-only batch selection; no DEV/TEST score used"}
        atomic_write_json(os.path.join(out_dir, "throughput_profile.json"), out)
        if gate is not None:
            gate.complete(token, identity="b58|profile|init=%s|sel=%s" % (init_sha[:12], sel["selected_batch"]))
        return out
    except BaseException as e:
        if gate is not None and token is not None:
            try:
                gate.fail(token, e)
            except Exception:
                pass
        raise


def select_batch(results, time_budget_s=TIME_BUDGET_S):
    """Fastest SAFE batch. Memory target ~6.7-7.1 GiB used is preferred; free >= 0.7 GiB is a
    hard safety floor. If nothing reaches the memory target, fall back to the fastest safe
    batch and say so. If throughput plateaus (<5% gain vs the previous larger batch), report
    the plateau. NEVER pads usage artificially."""
    ok = [r for r in results if r.get("ok")]
    safe = [r for r in ok if (r.get("nvidia_free_gib") is not None and r["nvidia_free_gib"] >= MIN_FREE_GIB)]
    if not safe:
        return {"selected_batch": None,
                "reason": "no candidate kept >=%.1f GiB free" % MIN_FREE_GIB,
                "safe": [], "plateau": None}
    rate = lambda r: r["batch"] / r["median_s"]
    fastest = max(safe, key=rate)
    in_target = [r for r in safe if r.get("nvidia_used_gib") is not None
                 and TARGET_USED_GIB[0] <= r["nvidia_used_gib"] <= TARGET_USED_GIB[1]]
    safe_sorted = sorted(safe, key=lambda r: r["batch"])
    plateau = None
    for i in range(1, len(safe_sorted)):
        prev, cur = safe_sorted[i - 1], safe_sorted[i]
        gain = (rate(cur) - rate(prev)) / rate(prev)
        if gain < 0.05:
            plateau = {"at_batch": cur["batch"], "gain_vs_prev": gain}
            break
    chosen = fastest
    reason = "maximum samples/second among measured memory-safe candidates; 7GB is a target, never filler"
    return {"selected_batch": chosen["batch"], "selected": chosen, "fastest_safe": fastest["batch"],
            "in_target_batches": [r["batch"] for r in in_target], "reason": reason,
            "plateau": plateau, "safe_batches": [r["batch"] for r in safe],
            "oom_batches": [r["batch"] for r in results if r.get("oom")]}


def main(argv=None):
    import argparse
    from cohort_v2 import load_cohort_v2
    from spec import SOURCE_SHA256
    ap = argparse.ArgumentParser(description="batch58 throughput/memory profiler (no optimizer step)")
    ap.add_argument("--cohort-dir", required=True)
    ap.add_argument("--source-sha", default=SOURCE_SHA256)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--ledger", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--no-reserve", action="store_true", help="artificial CPU/selfcheck only")
    ap.add_argument("--accepted-data-sha", default=None)
    ap.add_argument("--i-understand-this-is-a-real-fit", action="store_true", dest="opt_in")
    a = ap.parse_args(argv)
    cohort = load_cohort_v2(a.cohort_dir, source_sha256=a.source_sha, mmap_codes=True)
    out = run_profile(cohort, a.out_dir, a.ledger, device=a.device, opt_in=a.opt_in,
                      accepted_data_sha=a.accepted_data_sha, reserve_slot=not a.no_reserve)
    print(json.dumps(out, ensure_ascii=False, indent=2, default=str))
    return out


if __name__ == "__main__":
    main()
