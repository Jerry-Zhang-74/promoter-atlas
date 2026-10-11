# -*- coding: utf-8 -*-
"""batch52 WORKER selfcheck (read-only; NO fit / NO GPU / NO network).

Checks: ledger accounting, artifact presence + SHAs, checkpoint ABI consistency
(config sha / datahash / target Log2RPM_HEK293), cohort counts, and that the shipped
prediction command runs. Worker-only; canonical independent review is batch54's job.
"""
from __future__ import annotations
import argparse, datetime as dt, hashlib, json, os, subprocess, sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
HELPER = os.path.join(ROOT, "runs", "breakthrough_20261002", "code", "batch53_model_core")
sys.path.insert(0, HELPER)
from spec import config_sha256, SOURCE_SHA256, TARGET_COLUMN  # noqa: E402

B52 = os.path.join(ROOT, "runs", "breakthrough_20261002", "batch52")
COH = os.path.join(ROOT, "data", "breakthrough_20261002", "batch52", "corrected", "cohort_v2")
MODELS = os.path.join(ROOT, "data", "breakthrough_20261002", "batch52", "corrected", "models")


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def main():
    R = {"generated_at": dt.datetime.now().astimezone().isoformat(), "checks": {}, "fails": []}

    def chk(name, ok, detail=None):
        R["checks"][name] = {"ok": bool(ok), "detail": detail}
        if not ok:
            R["fails"].append(name)

    # 1. ledger
    led = [json.loads(l) for l in open(os.path.join(B52, "exec_gate_ledger.jsonl"), encoding="utf-8") if l.strip()]
    res = [e for e in led if e["kind"] == "reserved"]
    comp = {e["exec_id"] for e in led if e["kind"] == "completed"}
    fails = [e for e in led if e["kind"] == "failed"]
    consumed = sum(1 for e in res if e.get("consumes"))
    chk("ledger_reserves_4", len(res) == 4, {"n_reserved": len(res), "n_completed": len(comp), "n_failed": len(fails)})
    chk("ledger_completed_all", all(e["exec_id"] in comp for e in res),
        {"reserved": [e["purpose"] for e in res], "completed_ids": sorted(comp)})
    chk("ledger_consumed_eq_4", consumed == 4, {"consumed": consumed})

    # 2. artifacts
    arts = {
        "R0_readout": os.path.join(MODELS, "R0", "R0_readout.json"),
        "R0_preds": os.path.join(MODELS, "R0", "R0_predictions.json"),
        "Nabs_selected": os.path.join(MODELS, "Nabs", "Nabs_selected.ckpt"),
        "Nabs_ready": os.path.join(MODELS, "Nabs_ready.json"),
        "Njoint_selected": os.path.join(MODELS, "Njoint", "Njoint_selected.ckpt"),
        "Njoint_ready": os.path.join(MODELS, "Njoint_ready.json"),
        "coordinator_status": os.path.join(MODELS, "coordinator_status.json"),
        "epoch_cap_decision": os.path.join(MODELS, "epoch_cap_decision.json"),
    }
    present = {k: os.path.exists(v) for k, v in arts.items()}
    chk("all_artifacts_present", all(present.values()), present)
    R["artifact_sha256"] = {k: (sha256_file(v) if os.path.exists(v) else None) for k, v in arts.items()}

    # 3. checkpoint ABI consistency
    import torch
    abi = {}
    for arm in ("Nabs", "Njoint"):
        p = arts["%s_selected" % arm]
        if not os.path.exists(p):
            continue
        o = torch.load(p, map_location="cpu", weights_only=False)
        abi[arm] = {"format": o.get("format"), "config_sha256": o.get("config_sha256"),
                    "datahash": o.get("datahash"), "target": o["config"].get("target_column"),
                    "epoch": o.get("epoch"), "attempt_id": o.get("gate_attempt_id"),
                    "has_optimizer": o.get("optimizer") is not None, "has_rng": "rng" in o}
    dhs = {a: v["datahash"] for a, v in abi.items()}
    chk("ckpt_config_sha_match", all(v["config_sha256"] == config_sha256() for v in abi.values()) and len(abi) == 2,
        {"expected": config_sha256(), "got": {a: v["config_sha256"] for a, v in abi.items()}})
    chk("ckpt_targets_Log2RPM", all(v["target"] == TARGET_COLUMN for v in abi.values()), abi)
    chk("ckpt_same_datahash", len(set(dhs.values())) == 1 and len(dhs) == 2, dhs)
    R["checkpoint_abi"] = abi

    # 4. cohort counts (manifest + endpoints)
    man = json.load(open(os.path.join(COH, "manifest.json"), encoding="utf-8"))
    chk("manifest_source_sha", man.get("source_sha256", "").lower() == SOURCE_SHA256,
        {"manifest": man.get("source_sha256"), "expected": SOURCE_SHA256})
    hdr, cnt = None, {}
    with open(os.path.join(COH, "endpoints.tsv"), encoding="utf-8") as f:
        hdr = f.readline().rstrip("\n").split("\t")
        si, ti = hdr.index("split"), hdr.index(TARGET_COLUMN)
        for ln in f:
            if ln.strip():
                a = ln.rstrip("\n").split("\t")
                cnt[a[si]] = cnt.get(a[si], 0) + 1
    R["cohort"] = {"columns": hdr, "endpoints_by_split": cnt, "n_total": sum(cnt.values())}
    chk("cohort_has_target_column", TARGET_COLUMN in hdr and "Log2Norm_HEK293" not in hdr, hdr)
    br = json.load(open(os.path.join(COH, "build_report.json"), encoding="utf-8"))
    R["build_report"] = {k: br[k] for k in ("n_endpoints", "endpoint_counts", "group_counts",
                                            "pair_counts", "datahash")}
    chk("cohort_counts_consistent", br["endpoint_counts"] == cnt,
        {"build_report": br["endpoint_counts"], "observed": cnt})
    chk("cohort_datahash_matches_build", man.get("datahash") == br.get("datahash"),
        {"manifest": man.get("datahash"), "build_report": br.get("datahash")})
    chk("split_min_groups", br["group_counts"]["TRAIN"] >= 100 and br["group_counts"]["DEV"] >= 20
        and br["group_counts"]["TEST"] >= 30, br["group_counts"])

    # 5. shipped prediction command runs (CPU)
    try:
        out = subprocess.run([sys.executable, "-X", "utf8", os.path.join(HERE, "predict_v2.py"),
                              "--checkpoint", arts["Nabs_selected"], "--example", "--device", "cpu"],
                             capture_output=True, text=True, timeout=180)
        ok = out.returncode == 0 and "predicted_Log2RPM_HEK293" in out.stdout
        chk("prediction_command_runs", ok, {"rc": out.returncode, "stdout": out.stdout.strip()[:300],
                                            "stderr": out.stderr.strip()[-300:]})
    except Exception as e:  # noqa
        chk("prediction_command_runs", False, repr(e))

    R["ok"] = not R["fails"]
    print(json.dumps(R, ensure_ascii=False, indent=2, default=str))
    return 0 if R["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
