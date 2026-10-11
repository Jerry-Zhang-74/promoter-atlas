# -*- coding: utf-8 -*-
"""Batch52 finalization driver (runs AFTER Njoint completes). Sequential, single-GPU.

1) bounded wait for the Njoint exec_gate 'completed' record + Njoint_ready.json
2) create NEW release provenance + sidecar for Njoint_selected.ckpt (no training bytes touched)
3) GPU replay of the frozen selected weights on DEV and TEST for Nabs and Njoint (<=3 min total)
4) CPU final TEST metrics for R0/Nabs/Njoint + fixed baselines + paired group bootstrap
5) figures + Chinese result
NO fit / NO optimizer step / NO new training. Refuses to overwrite immutable outputs.
"""
from __future__ import annotations
import argparse, json, os, subprocess, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
PY = sys.executable
COH = os.path.join(ROOT, "data", "breakthrough_20261002", "batch52", "corrected", "cohort_v2")
MODELS = os.path.join(ROOT, "data", "breakthrough_20261002", "batch52", "corrected", "models")
B52 = os.path.join(ROOT, "runs", "breakthrough_20261002", "batch52")
REP = os.path.join(ROOT, "reports", "breakthrough_20261002", "batch52")
PRED = os.path.join(MODELS, "predictions")
LOG = os.path.join(B52, "logs", "finalize_v2.log")


def log(*a):
    msg = "[%s] %s" % (time.strftime("%H:%M:%S"), " ".join(str(x) for x in a))
    print(msg, flush=True)
    os.makedirs(os.path.dirname(LOG), exist_ok=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(msg + "\n")


def wait_njoint(deadline_s=3600):
    t0 = time.time()
    while time.time() - t0 < deadline_s:
        try:
            ev = [json.loads(l) for l in open(os.path.join(B52, "exec_gate_ledger.jsonl"), encoding="utf-8") if l.strip()]
            done = [e for e in ev if e["kind"] == "completed" and e.get("purpose") == "Njoint"]
            if done and os.path.exists(os.path.join(MODELS, "Njoint_ready.json")) \
                    and os.path.exists(os.path.join(MODELS, "Njoint", "Njoint_selected.ckpt")):
                log("Njoint complete:", done[-1]["ts"])
                return True
        except Exception as e:  # noqa
            log("wait err", repr(e))
        time.sleep(20)
    return False


def wait_gpu_free(max_wait=300):
    """Wait until no python process is using the GPU (torch training exit)."""
    t0 = time.time()
    while time.time() - t0 < max_wait:
        try:
            out = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,used_memory",
                                  "--format=csv,noheader"], capture_output=True, text=True, timeout=30)
            lines = [l for l in out.stdout.strip().splitlines() if l.strip()]
            if not lines:
                log("GPU free (no compute apps)")
                return True
            log("GPU busy:", lines)
        except Exception as e:  # noqa
            log("gpu check err", repr(e))
        time.sleep(15)
    return False


def sh(args):
    log("RUN", " ".join(args))
    p = subprocess.run([PY, "-X", "utf8"] + args, capture_output=True, text=True)
    if p.returncode != 0:
        log("FAILED rc=%d" % p.returncode, p.stdout[-2000:], p.stderr[-2000:])
        raise SystemExit("step failed")
    log("OK", p.stdout.strip()[-600:])
    return p.stdout


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wait-seconds", type=int, default=3600)
    ap.add_argument("--skip-wait", action="store_true")
    a = ap.parse_args()
    os.makedirs(PRED, exist_ok=True); os.makedirs(REP, exist_ok=True)
    if not a.skip_wait:
        if not wait_njoint(a.wait_seconds):
            log("Njoint did not complete within wait window; aborting finalize")
            raise SystemExit(2)
    wait_gpu_free()

    # (2) Njoint release provenance (source epoch = the DEV-selected best epoch)
    try:
        rdy = json.load(open(os.path.join(MODELS, "Njoint_ready.json"), encoding="utf-8"))
        best_ep = int(rdy["best"]["epoch"])
        log("Njoint selected epoch", best_ep, "dev_pair_mae", rdy["best"]["pair_mae"])
        sh([os.path.join(HERE, "release_provenance.py"),
            "--selected", os.path.join(MODELS, "Njoint", "Njoint_selected.ckpt"),
            "--source-epoch-ckpt", os.path.join(MODELS, "Njoint", "Njoint_epoch%02d.ckpt" % best_ep),
            "--arm", "Njoint",
            "--out-sidecar", os.path.join(MODELS, "Njoint", "Njoint_selected.ckpt.sha256.json"),
            "--out-prov", os.path.join(MODELS, "Njoint", "Njoint_selected.release_provenance.json")])
    except SystemExit:
        log("Njoint provenance step failed; continuing (will retry manually)")

    # (3) GPU replays (sequential; DEV then TEST per arm)
    for arm in ("Nabs", "Njoint"):
        for split in ("DEV", "TEST"):
            out = os.path.join(PRED, "%s_%s.tsv" % (arm, split))
            if os.path.exists(out):
                log("skip existing", out); continue
            sh([os.path.join(HERE, "eval_v2.py"), "replay",
                "--checkpoint", os.path.join(MODELS, arm, "%s_selected.ckpt" % arm),
                "--cohort-dir", COH, "--split", split, "--out", out,
                "--ready", out + ".ready.json", "--device", "cuda:0"])

    # (4) final CPU metrics on TEST
    sh([os.path.join(HERE, "eval_v2.py"), "final",
        "--cohort-dir", COH,
        "--r0-json", os.path.join(MODELS, "R0", "R0_predictions.json"),
        "--nabs-preds", os.path.join(PRED, "Nabs_TEST.tsv"),
        "--njoint-preds", os.path.join(PRED, "Njoint_TEST.tsv"),
        "--split", "TEST", "--out", os.path.join(REP, "final_test_metrics.json"),
        "--ready", os.path.join(REP, "final_test_metrics.ready.json")])

    # (5) figures + Chinese result
    sh([os.path.join(HERE, "make_summary_v2.py")])
    log("finalize done")


if __name__ == "__main__":
    main()
