# -*- coding: utf-8 -*-
"""batch52 execution coordinator: R0 -> Nabs -> Njoint on ONE GPU, gated (MAX 4 ledger).

Reuses the reviewed batch53 numeric code (pipeline_coordinator.fit_r0, train_neural.Trainer,
epoch_cap.choose_epoch_cap, exec_gate) but supplies the batch52 data-owner LEAN cohort loader.
Every actual solve/first-gradient reserves a slot in the batch52 ledger BEFORE execution.
Re-reads PARENT_DATA_ACCEPTANCE.txt + PARALLEL_PIPELINE.txt and asserts Log2RPM_HEK293 first.
"""
from __future__ import annotations
import argparse, datetime as dt, json, os, sys, time

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
os.environ.setdefault("PYTHONUTF8", "1")

HERE = os.path.dirname(os.path.abspath(__file__))
HELPER = os.path.abspath(os.path.join(HERE, "..", "..", "runs", "breakthrough_20261002", "code", "batch53_model_core"))
sys.path.insert(0, HERE); sys.path.insert(0, HELPER)

import torch  # noqa: E402
torch.set_num_threads(4)

from io_utils import atomic_write_json  # noqa: E402
from pipeline_coordinator import assert_parent_specs, fit_r0, _deadline_epoch  # noqa: E402
from train_neural import Trainer  # noqa: E402
from epoch_cap import choose_epoch_cap  # noqa: E402
from spec import MAX_EPOCHS, SOURCE_SHA256, config_sha256, TARGET_COLUMN  # noqa: E402
import cohort_v2  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cohort-dir", required=True)
    ap.add_argument("--out-root", required=True)
    ap.add_argument("--ledger", required=True)
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--day", default=None)
    ap.add_argument("--first-hhmm", default="14:00")
    ap.add_argument("--second-hhmm", default="16:00")
    ap.add_argument("--first-model-only", action="store_true")
    ap.add_argument("--i-understand-this-is-a-real-fit", action="store_true", dest="opt_in")
    a = ap.parse_args()

    prov = assert_parent_specs(a.run_dir)
    cohort = cohort_v2.load_cohort_v2(a.cohort_dir, source_sha256=SOURCE_SHA256, mmap_codes=True)
    assert cohort.target_column == TARGET_COLUMN and cohort.source_sha256 == SOURCE_SHA256
    os.makedirs(a.out_root, exist_ok=True)
    acc = SOURCE_SHA256
    day = a.day or dt.datetime.now().strftime("%Y-%m-%d")
    fh, fm = (int(x) for x in a.first_hhmm.split(":"))
    sh, sm = (int(x) for x in a.second_hhmm.split(":"))
    first_dl, second_dl = _deadline_epoch(day, fh, fm), _deadline_epoch(day, sh, sm)
    status = {"parent_provenance": prov, "cohort_datahash": cohort.datahash,
              "target_column": cohort.target_column, "config_sha256": config_sha256(),
              "source_sha256": acc, "day": day, "first_deadline": a.first_hhmm,
              "second_deadline": a.second_hhmm, "arms": {}}
    atomic_write_json(os.path.join(a.out_root, "coordinator_status.json"), status)

    def save():
        atomic_write_json(os.path.join(a.out_root, "coordinator_status.json"), status)

    t0 = time.time()
    r0 = fit_r0(cohort, os.path.join(a.out_root, "R0"), a.ledger, opt_in=a.opt_in, accepted_data_sha=acc)
    status["arms"]["R0"] = r0
    status["R0_elapsed_s"] = round(time.time() - t0, 1)
    save()
    print("[run_all_v2] R0 done", json.dumps(r0), flush=True)

    def decider(epoch_seconds):
        now = time.time()
        remaining = max(0.0, 180 * 60 - epoch_seconds)
        return choose_epoch_cap(epoch_seconds, now, first_dl, second_dl, remaining_gpu_seconds=remaining)

    tr = Trainer("Nabs", cohort, os.path.join(a.out_root, "Nabs"), a.ledger, device=a.device,
                 epoch_cap=MAX_EPOCHS, opt_in=a.opt_in, accepted_data_sha=acc)
    nabs = tr.run(cap_decider=decider, cap_log_path=os.path.join(a.out_root, "epoch_cap_decision.json"))
    status["arms"]["Nabs"] = {k: nabs[k] for k in ("best", "selected", "epoch_cap", "init_sha",
                                                   "attempt_id", "elapsed_s", "peak_cuda_mb")}
    save()
    atomic_write_json(os.path.join(a.out_root, "Nabs_ready.json"),
                      {"arm": "Nabs", "selected": nabs["selected"], "datahash": cohort.datahash,
                       "config_sha256": config_sha256(), "init_sha": nabs["init_sha"],
                       "history": nabs["history"], "best": nabs["best"]})
    print("[run_all_v2] Nabs done", json.dumps(status["arms"]["Nabs"], default=str), flush=True)
    if a.first_model_only:
        return

    tj = Trainer("Njoint", cohort, os.path.join(a.out_root, "Njoint"), a.ledger, device=a.device,
                 epoch_cap=nabs["epoch_cap"], opt_in=a.opt_in, accepted_data_sha=acc)
    njoint = tj.run()
    status["arms"]["Njoint"] = {k: njoint[k] for k in ("best", "selected", "epoch_cap", "init_sha",
                                                       "attempt_id", "elapsed_s", "peak_cuda_mb")}
    save()
    atomic_write_json(os.path.join(a.out_root, "Njoint_ready.json"),
                      {"arm": "Njoint", "selected": njoint["selected"], "datahash": cohort.datahash,
                       "config_sha256": config_sha256(), "init_sha": njoint["init_sha"],
                       "history": njoint["history"], "best": njoint["best"]})
    print("[run_all_v2] Njoint done", json.dumps(status["arms"]["Njoint"], default=str), flush=True)
    status["total_elapsed_s"] = round(time.time() - t0, 1)
    save()


if __name__ == "__main__":
    main()
