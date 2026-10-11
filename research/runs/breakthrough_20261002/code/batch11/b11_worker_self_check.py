#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Batch11 WORKER SELF-CHECK (NOT independent review).

The worker re-derives its own key numbers as a correctness smoke check and
writes worker_self_check.{json,txt} with an explicit "awaiting fresh review"
statement. The independent fresh reviewer is a SEPARATE session that writes
independent_review.{json,html}. This script must never be named or claim to be
an independent review.
"""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from b11_common import (MODEL_OUT, REPORT_OUT, DATA_OUT, CANDIDATE_VERSION, BATCH,
                        atomic_write_json, atomic_write_text)  # noqa: E402


def load(p):
    with open(p, encoding="utf-8") as fh:
        return json.load(fh)


def main():
    metrics = load(REPORT_OUT / "metrics.json")
    tests = load(REPORT_OUT / "tests_summary.json")
    replay = load(DATA_OUT / "replay.json")
    pri = metrics["primary_conditional_increment"]["contrasts"]

    summary = {
        "artifact_name": "worker_self_check",
        "candidate_version": CANDIDATE_VERSION,
        "batch": BATCH,
        "written_at_local": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "is_independent_review": False,
        "awaiting_fresh_review": True,
        "note": ("This is the WORKER self-check produced by the same implementation "
                 "session. It is NOT an independent review. A separate fresh reviewer "
                 "session will independently verify and write independent_review.json/html."),
        "worker_recomputed_contrasts": {
            k: {"delta": c["delta"], "lo": c["bootstrap"]["lo"], "hi": c["bootstrap"]["hi"],
                "meets_rule": c["meets_rule"]}
            for k, c in pri.items()
        },
        "tests_all_pass": tests["all_pass"],
        "tests_n_pass": tests["n_pass"],
        "replay_all_match": all(replay["batch11_replay_match"].values()),
        "full_batch10_old_prediction_match": all(replay["full_batch10_exact_old_prediction_match"].values()),
        "baseline_batch09_old_prediction_match": all(replay["baseline_batch09_exact_old_prediction_match"].values()),
    }
    atomic_write_json(REPORT_OUT / "worker_self_check.json", summary)

    lines = []
    lines.append("Batch11 worker self-check (%s)" % CANDIDATE_VERSION)
    lines.append("status: awaiting fresh independent review (NOT an independent review)")
    lines.append("four co-primary contrasts (delta / [lo,hi] / meets):")
    for k, c in pri.items():
        lines.append("  %s: %.6f [%.6f, %.6f] meets=%s"
                    % (k, c["delta"], c["bootstrap"]["lo"], c["bootstrap"]["hi"], c["meets_rule"]))
    lines.append("tests all_pass=%s (%d/%d)" % (tests["all_pass"], tests["n_pass"], tests["n_total"]))
    lines.append("replay all match=%s" % summary["replay_all_match"])
    lines.append("written at %s" % summary["written_at_local"])
    atomic_write_text(REPORT_OUT / "worker_self_check.txt", "\n".join(lines) + "\n")

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

