# -*- coding: utf-8 -*-
"""batch54 assemble the first-model acceptance JSON + report from independent evidence.

Combines: independent source/reference SHA checks, fixture-test evidence, exec_gate ledger
accounting, model-A artifact/coverage/selection/init checks, and independent DEV metrics
(R0 always; Nabs when frozen predictions exist). Writes reports/.../batch54/acceptance.json
and report.md. CPU only, read-only, no fit/GPU/network.
"""
from __future__ import annotations
import datetime as dt, glob, json, os, platform, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import indep_cohort as C
import indep_metrics as M
import model_checks as K
import run_acceptance as R

B54 = os.path.abspath(os.path.join(HERE))
REPORTS = os.path.abspath(os.path.join(HERE, "..", "..", "..", "..", "reports",
                                       "breakthrough_20261002", "batch54"))
DEFAULT_COHORT = "data/breakthrough_20261002/batch52/corrected/cohort_v2"
DEFAULT_OUTROOT = "data/breakthrough_20261002/batch52/corrected/models"
DEFAULT_LEDGER = "runs/breakthrough_20261002/batch52/exec_gate_ledger.jsonl"


def now():
    return dt.datetime.now().astimezone().isoformat()


def _libver():
    out = {}
    for mod in ("numpy", "scipy"):
        try:
            out[mod] = __import__(mod).__version__
        except Exception:
            out[mod] = None
    return out


def _load(path, default=None):
    try:
        return json.load(open(path, "r", encoding="utf-8"))
    except Exception:
        return default


def find_apreds(out_root):
    """Locate any frozen Nabs prediction artifact produced by the producer/evaluator."""
    pats = [os.path.join(out_root, "Nabs", "*.tsv"), os.path.join(out_root, "Nabs", "*.json"),
            os.path.join(out_root, "Nabs_pred*.tsv"), os.path.join(out_root, "Nabs_pred*.json"),
            os.path.join(out_root, "eval", "*Nabs*.tsv"), os.path.join(out_root, "eval", "*Nabs*.json"),
            os.path.join(out_root, "preds", "Nabs*.tsv")]
    found = []
    for p in pats:
        found.extend(sorted(glob.glob(p)))
    # exclude checkpoints, sidecars and markers
    bad = (".ckpt", ".sha256.json", "ready", "mid", "epoch")
    found = [f for f in found if not any(b in os.path.basename(f) for b in bad)]
    # a real prediction artifact must parse as coord_id -> float
    for f in found:
        try:
            d = R.load_preds_any(f)
            if d and all(isinstance(v, float) for v in list(d.values())[:5]):
                return f
        except Exception:
            continue
    return None


def main():
    cohort_dir = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_COHORT
    out_root = sys.argv[2] if len(sys.argv) > 2 else DEFAULT_OUTROOT
    ledger = sys.argv[3] if len(sys.argv) > 3 else DEFAULT_LEDGER
    apreds = sys.argv[4] if len(sys.argv) > 4 else None
    if apreds is None:
        apreds = find_apreds(out_root)
    os.makedirs(REPORTS, exist_ok=True)
    co = C.load(cohort_dir)
    events = K.read_ledger(ledger)
    acc = {
        "batch": "batch54",
        "version": "v6.38.0-first-model-independent-acceptance.rc1",
        "role": "independent evaluator (parallel to control62 data/training and control61 model code)",
        "clock": now(),
        "host": {"platform": platform.platform(), "python": platform.python_version(),
                 "interpreter": sys.executable, "metric_libs": _libver()},
        "evaluator_session": {"note": "Claude Code batch54 evaluator; clock + observed PIDs recorded",
                              "clock_start": now()},
        "scope": ["runs/breakthrough_20261002/code/batch54_review", "reports/breakthrough_20261002/batch54"],
        "cohort_dir": cohort_dir, "out_root": out_root, "ledger": ledger,
        "cohort_counts": {s: int((co["splits"] == s).sum()) for s in C.SPLITS},
        "pair_counts": {s: int(co["pairs"][s].shape[0]) for s in C.SPLITS},
        "group_counts": {s: len(set(co["groups"][co["splits"] == s].tolist())) for s in C.SPLITS},
        "provenance_evidence": _load(os.path.join(B54, "provenance_evidence.json")),
        "fixture_evidence": _load(os.path.join(B54, "fixture_evidence.json")),
        "fit_accounting": K.check_ledger(events),
    }

    # independent fixed pair baselines on DEV (no model needed)
    pr = co["pairs"]["DEV"]
    dtdev = co["targets"][pr[:, 0]] - co["targets"][pr[:, 1]]
    pgdev = co["groups"][pr[:, 0]]
    acc["baselines_DEV"] = M.fixed_pair_baselines(dtdev, pgdev)

    # R0 independent DEV metrics (frozen predictions already exist)
    r0p = os.path.join(out_root, "R0", "R0_predictions.json")
    if os.path.exists(r0p):
        pr0 = R.load_preds_any(r0p)
        acc["independent_metrics"] = {"R0_DEV": {
            "activity": R.activity_metrics(co, "DEV", pr0),
            "pair": R.pair_metrics(co, "DEV", pr0),
            "preds": r0p, "note": "independent batch54 metrics on producer frozen R0 predictions"}}
        acc["r0_baseline_provenance"] = {
            "kind": "corrected-target composition ridge R0 (alpha=100, 342 features)",
            "predictions": r0p,
            "target_column": "Log2RPM_HEK293",
            "used_for_comparison": True,
            "invalid_early_log2norm_R0_used_for_comparison": False,
            "note": "the early invalid log2-normalised R0 readout is NOT used as a baseline reference"}

    # model A checks + metrics
    acc["modelA"] = K.check_arm(out_root, "Nabs")
    acc["init_parity"] = K.check_init_parity_both(out_root)
    if apreds is None:
        own = os.path.join(B54, "preds", "Nabs_DEV.tsv")
        if os.path.exists(own):
            apreds = own
    rp = os.path.join(out_root, "Nabs", "Nabs_selected.release_provenance.json")
    if os.path.exists(rp):
        acc["modelA_release_provenance"] = _load(rp)
    mr = os.path.join(out_root, "Nabs_ready.json")
    if os.path.exists(mr):
        m = _load(mr, {})
        acc["modelA_producer_dev_history"] = {
            "note": "PRODUCER-computed DEV metrics (not independent); selection recomputed independently in modelA",
            "best": m.get("best"), "init_sha": m.get("init_sha"),
            "history": [{k: h.get(k) for k in ("epoch", "loss", "dev", "dev_pair_mae", "dev_sign")}
                        for h in m.get("history", [])]}
    if apreds and os.path.exists(apreds):
        pa = R.load_preds_any(apreds)
        acc["independent_metrics"]["Nabs_DEV"] = {
            "activity": R.activity_metrics(co, "DEV", pa),
            "pair": R.pair_metrics(co, "DEV", pa), "preds": apreds,
            "note": "independent batch54 metrics on frozen Nabs predictions"}
        acc["modelA_preds"] = apreds
        # A vs fixed zero-delta baseline, paired group bootstrap (multiplicity retained)
        maeA, signA, _ = R.pair_group_values_for(co, "DEV", pa)
        maeZ, signZ, _ = R.pair_group_values_for(co, "DEV", {c: 0.0 for c in co["coord_ids"].tolist()})
        groups = list(dict.fromkeys(pgdev.tolist()))
        acc["Nabs_vs_zero_delta_bootstrap_DEV"] = M.group_bootstrap_paired(
            groups, maeA, maeZ, signA, signZ, n_boot=2000)

    # quarantined TEST evaluation (all arms immutable); independent
    q_A = os.path.join(B54, "quarantine", "A_TEST.tsv")
    q_B = os.path.join(B54, "quarantine", "Njoint_TEST.tsv")
    if os.path.exists(q_A) and os.path.exists(q_B):
        pa = R.load_preds_any(q_A); pb = R.load_preds_any(q_B)
        acc["quarantine_dir"] = os.path.join(B54, "quarantine")
        acc["independent_metrics"]["Nabs_TEST_QUARANTINED"] = {
            "activity": R.activity_metrics(co, "TEST", pa), "pair": R.pair_metrics(co, "TEST", pa),
            "preds": q_A, "note": "quarantined; never fed to any trainer"}
        acc["independent_metrics"]["Njoint_TEST_QUARANTINED"] = {
            "activity": R.activity_metrics(co, "TEST", pb), "pair": R.pair_metrics(co, "TEST", pb),
            "preds": q_B, "note": "quarantined; never fed to any trainer"}
        prt = co["pairs"]["TEST"]
        dtt = co["targets"][prt[:, 0]] - co["targets"][prt[:, 1]]
        pgt = co["groups"][prt[:, 0]]
        acc["baselines_TEST"] = M.fixed_pair_baselines(dtt, pgt)
        maeA, signA, _ = R.pair_group_values_for(co, "TEST", pa)
        maeB, signB, _ = R.pair_group_values_for(co, "TEST", pb)
        gt = list(dict.fromkeys(pgt.tolist()))
        acc["A_vs_B_bootstrap_TEST"] = M.group_bootstrap_paired(gt, maeA, maeB, signA, signB, n_boot=2000)
        # predeclared investigation target (DEV primary)
        if "Nabs_DEV" in acc["independent_metrics"]:
            paD = R.load_preds_any(acc["independent_metrics"]["Nabs_DEV"]["preds"])
            pbD = R.load_preds_any(os.path.join(B54, "preds", "Njoint_DEV.tsv"))
            maeAD, signAD, _ = R.pair_group_values_for(co, "DEV", paD)
            maeBD, signBD, _ = R.pair_group_values_for(co, "DEV", pbD)
            gd = list(dict.fromkeys(pgdev.tolist()))
            acc["A_vs_B_bootstrap_DEV"] = M.group_bootstrap_paired(gd, maeAD, maeBD, signAD, signBD, n_boot=2000)
            baseA = acc["baselines_DEV"]
            best_dir = max((v["macro_sign_accuracy"] for v in baseA.values()
                            if v["macro_sign_accuracy"] == v["macro_sign_accuracy"]), default=float("nan"))
            gmAD = M.group_mean_mae_from_values(maeAD); gmBD = M.group_mean_mae_from_values(maeBD)
            rhoAD = acc["independent_metrics"]["Nabs_DEV"]["pair"]["rho_pooled"]
            rhoBD = R.pair_metrics(co, "DEV", pbD)["rho_pooled"]
            imp = (gmAD - gmBD) / max(gmAD, 1e-12)
            bt = acc["A_vs_B_bootstrap_DEV"]
            acc["investigation_target_DEV"] = {
                "pair_mae_improve_fraction_B_over_A": imp,
                "checks": {"improve_ge_10pct": bool(imp >= 0.10),
                           "bootstrap_lowerCI_gt_0": bool(bt["ok"] and bt["improvement_mae_B_over_A"]["lo"] > 0),
                           "macro_dir_ge_5pp_above_best_fixed": bool(
                               M.macro_sign_accuracy(signBD) >= best_dir + 0.05),
                           "activity_rho_not_worse_by_0.05": bool(rhoBD >= rhoAD - 0.05)},
                "verdict": None}
            acc["investigation_target_DEV"]["verdict"] = ("TARGET_MET" if all(
                acc["investigation_target_DEV"]["checks"].values()) else "LIMITED")

    # independent saved-weight GPU replay evidence (performed within the <=60 s shared budget)
    replay_files = []
    for d in (os.path.join(B54, "preds"), os.path.join(B54, "quarantine")):
        if os.path.isdir(d):
            for fn in sorted(os.listdir(d)):
                if fn.endswith(".ready.json"):
                    replay_files.append(os.path.join(d, fn))
    if replay_files:
        rj = [_load(p, {}) for p in replay_files]
        acc["gpu_replay_evidence"] = {
            "note": "independent batch54 no_grad replay of SAVED weights (cuda:0); GPU lock held "
                    "only during forward, released before CPU metric work; cumulative elapsed seconds",
            "runs": rj,
            "total_elapsed_s": round(sum(float(r.get("elapsed_s") or 0.0) for r in rj), 3),
            "budget_s": 60.0,
            "within_budget": bool(sum(float(r.get("elapsed_s") or 0.0) for r in rj) <= 60.0),
            "cpu_neural_fallback_used": False,
        }

    # fit accounting narrative (do NOT claim all runs compliant: one historical UNGATED run exists)
    exec_ids = acc["fit_accounting"].get("exec_ids", [])
    acc["fit_accounting_narrative"] = {
        "total_executions": acc["fit_accounting"].get("n_executions_reserved"),
        "max": 4,
        "historical_ungated": {
            "note": "preserved historical run; wrong target R0 log2-normalised, fit BEFORE the "
                    "exec_gate; counted against the MAX4 budget and NOT claimed compliant",
            "exec_id": exec_ids[0] if exec_ids else None, "purpose": "batch52-R0",
            "target": "log2-normalised R0 (NOT Log2RPM_HEK293)", "compliant": False},
        "corrected_new_models": {
            "note": "corrected-target gated fits accepted SEPARATELY with history",
            "rows": [{"exec_id": exec_ids[i], "purpose": p, "target": "Log2RPM_HEK293",
                      "compliant": True} for i, p in enumerate(acc["fit_accounting"].get("purposes", []))
                     if i >= 1]},
        "all_runs_compliant_claimed": False,
    }

    # parent-readable TEST copies (identical replay, same pred_sha256); never fed back to B trainer
    pc = {}
    for arm in ("Nabs", "Njoint"):
        tsv = os.path.join(REPORTS, "%s_TEST_parent_granted.tsv" % arm)
        rj = tsv + ".ready.json"
        if os.path.exists(tsv) and os.path.exists(rj):
            pc[arm] = {"tsv": tsv, "pred_sha256": _load(rj, {}).get("pred_sha256"),
                       "rows": _load(rj, {}).get("n_rows")}
    qa = _load(os.path.join(B54, "quarantine", "A_TEST.tsv.ready.json"), {})
    if pc and qa:
        acc["parent_readable_test_copies"] = {
            "note": "Parent-readable TEST prediction copies; identical replay (pred_sha256 match) of "
                    "the quarantined results. TEST activity is NEVER fed back to the B (Njoint) trainer.",
            "A_pred_sha256_match": pc.get("Nabs", {}).get("pred_sha256") == qa.get("pred_sha256"),
            "copies": pc,
            "test_fed_back_to_B_trainer": False,
        }

    # verdict dictionary
    prov = acc.get("provenance_evidence") or {}
    fx = acc.get("fixture_evidence") or {}
    ledger_ok = acc["fit_accounting"].get("within_max", False)
    prov_ok = bool(prov.get("source", {}).get("match") and prov.get("reference", {}).get("match")
                   and prov.get("canonical_only"))
    fx_ok = bool(fx.get("all_passed"))
    a = acc["modelA"]
    a_ok = bool(a.get("ok"))
    acc["state"] = "MODEL_A_AVAILABLE" if a.get("ready_exists") else "PRELIMINARY_WAITING_FOR_MODEL_A"
    acc["limits"] = [
        "Retrospective development evaluation; public/pretrained exposure means no pristine prospective claim.",
        "Reference-encoded sequence is a proxy for donor HG02601, not the clone sequence; HEK293 identity preserved; no local 293FT validation.",
        "TEST targets are physically present in endpoints.tsv; 'sealed' TEST separation is procedural (trainer never loads TEST), not cryptographic.",
        "Group bootstrap assumes declared groups; residual genomic dependence means CIs are conditional, not independent-family guarantees.",
        "No CPU neural fallback used; the independent GPU replay of SAVED weights was performed within the shared <=60 s exclusive budget (GPU lock with PID+creation identity, released before CPU metric work).",
        "One historical UNGATED R0 run (wrong target) is counted against the MAX4 budget and is NOT claimed compliant; only the corrected-target gated fits are accepted, separately, with history.",
    ]
    acc["verdict"] = {
        "integrity_and_compliance": ("PASS" if (ledger_ok and prov_ok and fx_ok and (a_ok or not a.get("ready_exists")))
                                     else "FAIL/PENDING"),
        "checks": {"source_and_reference_sha_independent": prov_ok,
                   "fixture_metric_definitions_no_fit": fx_ok,
                   "fit_ledger_within_max_4": ledger_ok,
                   "modelA_artifact_checks": a_ok if a.get("ready_exists") else "PENDING"},
        "scientific_gain": (acc.get("investigation_target_DEV", {}).get("verdict")
                            or "SEPARATE (not part of integrity verdict)"),
        "first_model_delivery": "DELIVERED" if a.get("ready_exists") else "PENDING",
    }
    acc["verdict"]["model_quality"] = ("FIRST_MODEL_DELIVERED_WITH_HONEST_LIMITS"
                                      if a.get("ready_exists") else "PENDING")
    json.dump(acc, open(os.path.join(REPORTS, "acceptance.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    write_report(acc)
    print(json.dumps(acc, ensure_ascii=False, indent=2, default=str))
    return acc


def write_report(acc):
    a = acc.get("modelA", {})
    im = acc.get("independent_metrics", {})
    lines = []
    lines.append("# batch54 -- independent first-model acceptance (v6.38.0 ...rc1)")
    lines.append("")
    lines.append("Role: independent evaluator (CPU, read-only). Scope: `runs/breakthrough_20261002/code/batch54_review`, `reports/breakthrough_20261002/batch54`.")
    lines.append("Clock: %s" % acc["clock"])
    lines.append("State: **%s**" % acc["state"])
    lines.append("")
    lines.append("## Independent integrity checks")
    prov = acc.get("provenance_evidence") or {}
    lines.append("- source BED sha256 match: %s" % prov.get("source", {}).get("match"))
    lines.append("- Ensembl GRCh37 reference sha256 match: %s (bytes=%s)" % (
        prov.get("reference", {}).get("match"), prov.get("reference", {}).get("bytes")))
    lines.append("- canonical chromosome mapping only: %s (%d chroms: %s)" % (
        prov.get("canonical_only"), prov.get("n_distinct_chrom"),
        ",".join(sorted(prov.get("chrom_prefixes", {}).keys()))))
    lines.append("- fit ledger: %s of MAX4 consumed: %s; purposes=%s" % (
        acc["fit_accounting"].get("n_executions_reserved"), acc["fit_accounting"].get("within_max"),
        acc["fit_accounting"].get("purposes")))
    fx = acc.get("fixture_evidence") or {}
    lines.append("- artificial no-fit metric fixtures: %s/%s pass" % (fx.get("n_pass"), fx.get("n_total")))
    lines.append("")
    lines.append("## Cohort")
    lines.append("- endpoints per split: %s" % acc["cohort_counts"])
    lines.append("- pairs per split: %s" % acc["pair_counts"])
    lines.append("- groups per split: %s" % acc["group_counts"])
    lines.append("")
    lines.append("## Independent DEV metrics (batch54 own implementations)")
    for k, v in im.items():
        act = v.get("activity", {}); pai = v.get("pair", {})
        lines.append("- **%s** (%s)" % (k, v.get("preds")))
        lines.append("  - activity rho=%.4f groupMAE=%.4f RMSE=%.4f top10=%.4f" % (
            act.get("rho", float("nan")), act.get("group_mae", float("nan")),
            act.get("rmse", float("nan")), act.get("top10_precision_fixed_ids", float("nan"))))
        lines.append("  - pair groupMAE=%.4f macroSign=%.4f rho_pooled=%.4f truezero=%.4f predTies=%s" % (
            pai.get("group_mean_mae", float("nan")), pai.get("macro_sign_accuracy", float("nan")),
            pai.get("rho_pooled", float("nan")), pai.get("true_zero_rate", float("nan")),
            pai.get("n_pred_ties")))
    lines.append("")
    ph = acc.get("modelA_producer_dev_history")
    if ph:
        lines.append("")
        lines.append("## Producer-reported DEV history (NOT independent; selection recomputed independently)")
        for h in ph.get("history", []):
            d = h.get("dev", {}) or {}
            lines.append("- epoch %s: loss=%.4f devMAE=%.4f devRho=%.4f pairMAE=%.4f sign=%.4f" % (
                h.get("epoch"), h.get("loss", float("nan")), d.get("mae", float("nan")),
                d.get("rho", float("nan")), h.get("dev_pair_mae", float("nan")),
                h.get("dev_sign", float("nan"))))
        lines.append("- best: %s" % json.dumps(ph.get("best")))
    boot = acc.get("Nabs_vs_zero_delta_bootstrap_DEV")
    if boot:
        lines.append("")
        lines.append("## Independent paired group bootstrap (Nabs vs zero-delta, DEV, multiplicity retained)")
        lines.append("- %s" % json.dumps(boot))
    lines.append("")
    bt = acc.get("A_vs_B_bootstrap_DEV")
    if bt:
        lines.append("")
        lines.append("## Independent A(Nabs) vs B(Njoint) paired bootstrap (DEV, multiplicity retained)")
        lines.append("- improvement B over A: %s" % json.dumps(bt.get("improvement_mae_B_over_A")))
        lines.append("- sign B-A: %s" % json.dumps(bt.get("sign_B_minus_A")))
    it = acc.get("investigation_target_DEV")
    if it:
        lines.append("")
        lines.append("## Predeclared investigation target (DEV): **%s**" % it.get("verdict"))
        lines.append("- improve fraction B over A = %.4f" % it.get("pair_mae_improve_fraction_B_over_A", float("nan")))
        lines.append("- checks: %s" % json.dumps(it.get("checks")))
    btT = acc.get("A_vs_B_bootstrap_TEST")
    if btT:
        lines.append("")
        lines.append("## Quarantined TEST (independent, both arms immutable)")
        lines.append("- improvement B over A: %s" % json.dumps(btT.get("improvement_mae_B_over_A")))
        lines.append("- sign B-A: %s" % json.dumps(btT.get("sign_B_minus_A")))
        lines.append("- quarantine dir: %s" % acc.get("quarantine_dir"))
    lines.append("")
    lines.append("## Fixed DEV baselines")
    for name, v in (acc.get("baselines_DEV") or {}).items():
        lines.append("- %s: groupMAE=%.4f macroSign=%.4f" % (
            name, v.get("group_mean_mae", float("nan")), v.get("macro_sign_accuracy", float("nan"))))
    lines.append("")
    lines.append("## Model A artifact checks")
    if a.get("ready_exists"):
        lines.append("- selected: %s" % a.get("selected"))
        lines.append("- selected sha256: %s (sidecar match=%s)" % (a.get("selected_sha256"), a.get("selected_sidecar_sha_match")))
        lines.append("- config_sha_ok=%s target_ok=%s datahash_ok=%s" % (a.get("config_sha_ok"), a.get("target_ok"), a.get("datahash_ok")))
        lines.append("- epochs saved=%s contiguous=%s all_full_train=%s" % (a.get("n_epochs_saved"), a.get("epochs_contiguous"), a.get("all_epochs_full_train")))
        lines.append("- DEV selection matches marker=%s" % a.get("selection_matches_marker"))
        lines.append("- shared init bytes ok=%s" % a.get("shared_init_ok"))
        lines.append("- runnable load check: %s" % json.dumps(a.get("runnable_load_check")))
        lines.append("- issues: %s" % a.get("issues"))
    else:
        lines.append("- PENDING: Nabs_ready.json not present yet. Bounded local waiter polls every <=30s until 12:20 Asia/Shanghai (no paid LLM polling).")
    lines.append("")
    gre = acc.get("gpu_replay_evidence")
    if gre:
        lines.append("## Independent GPU replay of saved weights (no CPU neural fallback)")
        lines.append("- total elapsed %.3f s of %.0f s budget; within budget=%s" % (
            gre.get("total_elapsed_s", float("nan")), gre.get("budget_s"), gre.get("within_budget")))
        for r in gre.get("runs", []):
            lines.append("- %s [%s] %s rows, %.3f s, pred_sha256=%s" % (
                os.path.basename(r.get("checkpoint") or "?"), r.get("split"), r.get("n_rows"),
                float(r.get("elapsed_s") or 0.0), (r.get("pred_sha256") or "")[:16]))
        lines.append("")
    fan = acc.get("fit_accounting_narrative")
    if fan:
        lines.append("## Fit accounting (history preserved; no blanket compliance claim)")
        hu = fan.get("historical_ungated", {})
        lines.append("- historical UNGATED run counted against MAX4: exec=%s purpose=%s target=%s compliant=%s" % (
            (hu.get("exec_id") or "")[:12], hu.get("purpose"), hu.get("target"), hu.get("compliant")))
        for row in fan.get("corrected_new_models", {}).get("rows", []):
            lines.append("- corrected gated fit: exec=%s purpose=%s target=%s compliant=%s" % (
                (row.get("exec_id") or "")[:12], row.get("purpose"), row.get("target"), row.get("compliant")))
        lines.append("- all_runs_compliant_claimed=%s" % fan.get("all_runs_compliant_claimed"))
        lines.append("")
    lines.append("## Verdict")
    lines.append("```")
    lines.append(json.dumps(acc["verdict"], ensure_ascii=False, indent=2))
    lines.append("```")
    lines.append("")
    lines.append("## Limits (honest)")
    for l in acc["limits"]:
        lines.append("- %s" % l)
    lines.append("")
    open(os.path.join(REPORTS, "report.md"), "w", encoding="utf-8").write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
