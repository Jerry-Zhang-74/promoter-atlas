# -*- coding: utf-8 -*-
"""One-time, throughput-only shared epoch-cap decision (batch52 PARALLEL_PIPELINE).

After the FIRST complete Nabs epoch ONLY, choose the largest common cap from 6/4/2 that
fits cumulative GPU <= 180 min and the first(14:00)/second(16:00) execution targets, using
MEASURED epoch seconds, conservative joint cost = 1.5 x Nabs and a 30 min buffer. DEV/TEST
scores are never inputs. The chosen cap is applied to BOTH arms and logged once BEFORE
Nabs epoch 2. If even 2 does not fit, report the walltime risk (no fabricated completion).
"""
from __future__ import annotations
import json

from io_utils import atomic_write_json, now_iso
from spec import EPOCH_CAP_CANDIDATES

GPU_BUDGET_SECONDS = 180 * 60
BUFFER_SECONDS = 30 * 60
JOINT_FACTOR = 1.5


def choose_epoch_cap(measured_epoch_seconds: float, epoch1_wall_start: float,
                     first_deadline: float, second_deadline: float,
                     remaining_gpu_seconds: float = GPU_BUDGET_SECONDS,
                     candidates=EPOCH_CAP_CANDIDATES) -> dict:
    """epoch1_wall_start: epoch-1 start time (epoch seconds since a common reference);
    deadlines likewise, as comparable floats (epoch seconds). Pure scheduling."""
    best = None
    detail = {}
    for c in sorted(candidates, reverse=True):
        cost_a = c * measured_epoch_seconds
        cost_b = c * measured_epoch_seconds * JOINT_FACTOR
        total = cost_a + cost_b + BUFFER_SECONDS
        fits_gpu = total <= remaining_gpu_seconds
        fits_first = (epoch1_wall_start + cost_a) <= first_deadline
        fits_second = (epoch1_wall_start + cost_a + cost_b) <= second_deadline
        ok = fits_gpu and fits_first and fits_second
        detail[str(c)] = {"cost_a_s": round(cost_a, 1), "cost_b_s": round(cost_b, 1),
                          "total_with_buffer_s": round(total, 1), "fits_gpu": fits_gpu,
                          "fits_first": fits_first, "fits_second": fits_second, "feasible": ok}
        if ok and best is None:
            best = c
    return {"chosen_cap": best, "risk": best is None, "measured_epoch_seconds": measured_epoch_seconds,
            "joint_factor": JOINT_FACTOR, "buffer_seconds": BUFFER_SECONDS,
            "gpu_budget_seconds": GPU_BUDGET_SECONDS, "candidates": detail,
            "note": "throughput/wall-clock only; no DEV/TEST score used"}


def log_epoch_cap(path: str, decision: dict, epoch1_seconds: float) -> str:
    """Write the single cap decision log (before Nabs epoch 2). Atomic."""
    payload = dict(decision)
    payload["logged_at"] = now_iso()
    payload["chosen_after"] = "Nabs epoch 1 complete"
    payload["observed_nabs_epoch1_seconds"] = epoch1_seconds
    return atomic_write_json(path, payload)
