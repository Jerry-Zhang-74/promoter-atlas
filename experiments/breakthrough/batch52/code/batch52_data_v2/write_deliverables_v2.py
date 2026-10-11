# -*- coding: utf-8 -*-
"""Generate the batch52 delivery documents from the frozen JSON artifacts."""
from __future__ import annotations
import datetime as dt, json, os

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
REP = os.path.join(ROOT, "reports", "breakthrough_20261002", "batch52")
RUNS = os.path.join(ROOT, "runs", "breakthrough_20261002", "batch52")
MODELS = os.path.join(ROOT, "data", "breakthrough_20261002", "batch52", "corrected", "models")
CARDS = os.path.join(ROOT, "models", "cards")
COH = os.path.join(ROOT, "data", "breakthrough_20261002", "batch52", "corrected", "cohort_v2")
VERSION = "v6.38.0-full-hek-nested-multitask.rc1"
CARD_NAME = "promoter_atlas_v6_38_0_full_hek_nested_multitask_rc1_card.txt"


def J(p):
    return json.load(open(p, encoding="utf-8")) if os.path.exists(p) else None


def build_report(now, binfo, nabs, njoint, final, cap, status, aprov, nprov):
    L = []
    A = L.append
    A("# batch52 " + VERSION + " -- worker delivery report")
    A("")
    A("Generated: " + now + " (Asia/Shanghai). Worker selfcheck only; the canonical independent")
    A("review is batch54's, run in a separate session.")
    A("")
    A("## 1. Deliverable")
    A("Three fitted estimators on the corrected full HEK293 cohort:")
    A("- R0 composition ridge (342 features: mono/di/tri/tetra one-hot + GC frac + log length, alpha 100) -- closed-form, one solve.")
    A("- Nabs random-init sequence CNN, absolute-activity loss (masked, group-weighted SmoothL1).")
    A("- Njoint the SAME CNN init/architecture/optimizer/seed plus the nested-difference loss (0.5 x group-weighted SmoothL1 on short-long).")
    A("")
    A("Selected DEV-best checkpoints (usable, loadable, prediction-verified):")
    for arm, r in (("Nabs", nabs), ("Njoint", njoint)):
        if r:
            A("- " + arm + ": data/breakthrough_20261002/batch52/corrected/models/" + arm + "/"
              + arm + "_selected.ckpt (best epoch " + str(r["best"]["epoch"])
              + ", DEV group-mean pair MAE %.4f)" % r["best"]["pair_mae"])
    A("")
    A("Prediction entrypoint (CPU or cuda:0, no training):")
    A("  python code/batch52_data_v2/predict_v2.py --checkpoint CKPT --seq ACGT...")
    A("  --seq-file f.txt or --example ; sequences >600 bp are rejected (no windowing).")
    A("")
    A("## 2. Corrected target + data")
    A("- Target = author-documented Log2RPM_HEK293 (released values used directly; NOT Log2Norm_HEK293, no recomputation, no pseudocount guess).")
    A("- Source BED GSE301246 v2_MH4 fold0, 103144961 B, sha256 1bbe5db71a5f4a92a281d19567b28ca3751f93d62959e6d61bd41188dd550d96.")
    if binfo:
        A("- Cohort data/breakthrough_20261002/batch52/corrected/cohort_v2: %d endpoints; TRAIN %d / DEV %d / TEST %d."
          % (binfo["n_endpoints"], binfo["endpoint_counts"]["TRAIN"], binfo["endpoint_counts"]["DEV"], binfo["endpoint_counts"]["TEST"]))
        A("- Union groups " + json.dumps(binfo["group_counts"]) + " (all >= 100/20/30). Nested pairs " + json.dumps(binfo["pair_counts"]) + ".")
        A("- Exclusions: missing sequence %d, missing target %d, conflicting labels %d; join coverage %.3f."
          % (binfo["excluded_missing_seq"], binfo["excluded_target"], binfo["conflicting_target_labels"], binfo["join_coverage"]))
        A("- datahash " + binfo["datahash"] + ".")
    A("- Reference: Ensembl GRCh37 (= author hg19 contract), sha256 0a43b56dec40debae976d6e70cac68ea6ed874f9fb7c8c814363702ff1d47865, integrity-checked.")
    A("  It is a reference PROXY for donor HG02601, not an actual donor/clone sequence; no local 293FT claim.")
    A("- Split: one SHA256('b52-full-hek-v1|'+ID) mod 100, frozen before labels were rebuilt; TEST labels used only in the single final evaluation.")
    A("")
    A("## 3. Training effort (all eligible TRAIN rows, complete fold0)")
    if cap:
        A("- Epoch cap chosen once after Nabs epoch 0 from measured %.1f s/epoch: cap = %s"
          % (cap.get("measured_epoch_seconds", float("nan")), cap.get("chosen_cap")))
    for arm in ("R0", "Nabs", "Njoint"):
        if status and arm in status.get("arms", {}):
            v = status["arms"][arm]
            A("- " + arm + ": " + json.dumps({k: v[k] for k in v if k in
              ("elapsed_s", "epoch_cap", "peak_cuda_mb", "init_sha", "attempt_id")}, default=str))
    A("- Fit budget: exec_gate ledger MAX 4; consumed 4/4 (1 historical ungated wrong-target R0 + corrected R0 + Nabs + Njoint). NO reserve left.")
    A("- Batch <=32 (pair microbatch 16); CPU 4 threads; one RTX 5060 cuda:0; GPU training well inside 180 min.")
    A("")
    A("## 4. Held-out (TEST) results")
    if final:
        A("| arm | rho | group-equal MAE | RMSE | top-10pct prec | pair groupMAE(delta) | macro sign acc | pair n/groups |")
        A("|---|---|---|---|---|---|---|---|")
        for arm in ("R0", "Nabs", "Njoint"):
            if arm in final["arms"]:
                ac = final["arms"][arm]["activity"]; pr = final["arms"][arm].get("pair", {})
                A("| %s | %.4f | %.4f | %.4f | %.4f | %s | %s | %s/%s |" % (
                    arm, ac["rho"], ac["group_mae"], ac["rmse"], ac["top10_precision"],
                    ("%.4f" % pr["group_mean_mae"]) if pr else "-",
                    ("%.4f" % pr["macro_sign_accuracy"]) if pr else "-",
                    pr.get("n_pairs", "-") if pr else "-", pr.get("n_groups", "-") if pr else "-"))
        A("")
        A("- Fixed pair baselines: " + json.dumps(final["fixed_baselines"], ensure_ascii=False))
        A("- Paired group bootstrap (>=30 TEST groups, 2000 resamples): " + json.dumps(final["bootstrap_Nabs_vs_Njoint"], ensure_ascii=False))
        A("- Predeclared investigation target: " + json.dumps(final["investigation_target"], ensure_ascii=False))
        A("- Verdict (worker): **" + str(final["verdict"]) + "**")
    else:
        A("(final TEST metrics not yet written at report-generation time)")
    A("")
    A("## 5. Provenance / SHA")
    for arm, p in (("Nabs", aprov), ("Njoint", nprov)):
        if p:
            A("- " + arm + " selected sha256 " + p["selected_sha256"] + " (model state identical to source epoch "
              + str(p["selected_epoch"]) + ": " + str(p["model_state_identical"]) + ")")
    A("- config_sha256 (ABI) 12488fae6f07c9a61abcee1dd1bf127bdc5cbd7ea2955a22c6fa3b5936dbf157.")
    A("- Reference SHA, cohort SHA, checkpoint SHAs and the ledger are in WORKER_SELFCHECK.json and the .sha256.json sidecars.")
    A("")
    A("## 6. Honest limitations")
    A("- Reference-encoded sequences are a hg19 PROXY for the donor; the source is HEK293, NOT the local 293FT reporter assay; no local 293FT validation is claimed.")
    A("- Adjacent same-end nested fragments are OBSERVATIONAL, not causal.")
    A("- Near-homology candidate independence is a sequence-similarity argument, not confirmed 90/90 alignments.")
    A("- All results are retrospective; TEST was opened exactly once with frozen DEV-selected weights.")
    A("- R0/Nabs/Njoint are exploratory; NO production promotion. Production pointer unchanged (v6.3 main / v6.4 cellline pilot).")
    return L


def write_card(now, binfo, nabs, njoint, final):
    C = []
    B = C.append
    B("model_card: promoter_atlas batch52 " + VERSION)
    B("created: " + now)
    B("task: HEK293 MPRA promoter-fragment activity regression, target = Log2RPM_HEK293 (author units)")
    B("data: GSE301246 v2_MH4 fold0 HEK293; " + (str(binfo["n_endpoints"]) if binfo else "?") + " endpoints")
    B("reference: Ensembl GRCh37 (author hg19 contract) sha256 0a43b56dec40debae976d6e70cac68ea6ed874f9fb7c8c814363702ff1d47865; PROXY for donor HG02601")
    B("arms: R0 (composition ridge 342 feat, alpha 100); Nabs (random-init CNN, absolute loss); Njoint (same CNN + 0.5 nested-difference loss)")
    B("architecture: one-hot ACGT -> zero-pad 600 + valid mask; Conv1d(4,64,k9,same)+GELU; 3 residual blocks Conv1d(64,64,k5,same) dilations 1/2/4 + GELU + dropout 0.1; masked mean AND masked max + standardized loglength + GC; linear 130->64 GELU dropout 0.1 -> scalar")
    B("optimizer: AdamW lr 0.001 wd 0.0001, grad clip 1.0, seed 20261004, batch 32 (pair microbatch 16), <=6 epochs, DEV-selected best")
    B("selection metric: DEV group-mean pair MAE (earliest epoch on ties)")
    B("shipped checkpoints:")
    for arm, r in (("Nabs", nabs), ("Njoint", njoint)):
        if r:
            B("  " + arm + "_selected.ckpt  best_epoch=" + str(r["best"]["epoch"])
              + "  dev_pair_mae=%.6f" % r["best"]["pair_mae"])
    if final:
        B("held-out TEST (single frozen evaluation):")
        for arm in ("R0", "Nabs", "Njoint"):
            if arm in final["arms"]:
                ac = final["arms"][arm]["activity"]; pr = final["arms"][arm].get("pair", {})
                B("  %s rho=%.4f groupMAE=%.4f top10=%.4f pairGroupMAE=%s macroSign=%s"
                  % (arm, ac["rho"], ac["group_mae"], ac["top10_precision"],
                     pr.get("group_mean_mae"), pr.get("macro_sign_accuracy")))
        B("verdict: " + str(final["verdict"]))
    B("usage: python code/batch52_data_v2/predict_v2.py --checkpoint Nabs_selected.ckpt --seq ACGT... [--seq-file f.txt | --example]")
    B("limits: reference-proxy not donor clone; HEK293 not local 293FT; observational nested pairs; NOT promoted (production pointer unchanged)")
    os.makedirs(CARDS, exist_ok=True)
    p = os.path.join(CARDS, CARD_NAME)
    open(p, "w", encoding="utf-8").write("\n".join(C) + "\n")
    return p


def write_schedule(now):
    sched = {
        "created_at": now,
        "gpu": "one RTX 5060 Laptop cuda:0",
        "policy": "single GPU; training owner only; other agents need an explicit granted slot",
        "sequence_after_Njoint": [
            "1. batch52 worker: held-out replays (Nabs/Njoint DEV+TEST) + final CPU metrics [this process]",
            "2. GPU released: batch54 independent evaluator short verified slot (<=60 s per model)",
            "3. GPU released: batch58 throughput profiling / resource-variant training picks it up",
        ],
        "shared_lock_hint": "acquire runs/breakthrough_20261002/batch52/gpu_slot/gpu_exclusive.lock (PID+creation) before any forward",
    }
    p = os.path.join(RUNS, "GPU_SLOT_SCHEDULE.json")
    json.dump(sched, open(p, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    return p


def main():
    now = dt.datetime.now().astimezone().isoformat()
    binfo = J(os.path.join(COH, "build_report.json"))
    nabs = J(os.path.join(MODELS, "Nabs_ready.json"))
    njoint = J(os.path.join(MODELS, "Njoint_ready.json"))
    final = J(os.path.join(REP, "final_test_metrics.json"))
    cap = J(os.path.join(MODELS, "epoch_cap_decision.json"))
    status = J(os.path.join(MODELS, "coordinator_status.json"))
    aprov = J(os.path.join(MODELS, "Nabs", "Nabs_selected.release_provenance.json"))
    nprov = J(os.path.join(MODELS, "Njoint", "Njoint_selected.release_provenance.json"))
    L = build_report(now, binfo, nabs, njoint, final, cap, status, aprov, nprov)
    rpt = os.path.join(REP, "REPORT.md")
    open(rpt, "w", encoding="utf-8").write("\n".join(L) + "\n")
    card = write_card(now, binfo, nabs, njoint, final)
    sched = write_schedule(now)
    print(json.dumps({"report": rpt, "card": card, "schedule": sched,
                      "final_present": bool(final)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
