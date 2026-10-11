# -*- coding: utf-8 -*-
"""BATCH29 reporting: model card, status.json, selfcheck.json, registry+changelog append-only, next_proposal."""
import json, hashlib, os, datetime
from pathlib import Path

ROOT = Path("E:/promoter_atlas")
REP = ROOT / "reports/breakthrough_20261002/batch29"
BT = ROOT / "reports/breakthrough_20261002"
RUNC = ROOT / "runs/breakthrough_20261002/batch29"
CARDS = ROOT / "models/cards"
CARDS.mkdir(parents=True, exist_ok=True)
VER = "v6.20.1-sureproxy.rc1"

def rd(p, d=None):
    try: return json.load(open(p, encoding="utf-8"))
    except Exception: return d

def sha_bytes(b): return hashlib.sha256(b).hexdigest()
def now(): return datetime.datetime.now().astimezone().replace(microsecond=0).isoformat()

passA = rd(REP / "passA.json"); split = rd(REP / "split.json"); tf = rd(REP / "target_freeze.json")
tm = rd(REP / "test_metrics.json"); rt = rd(REP / "retrieval.json"); sr = rd(REP / "source_resolution.json")
man = rd(REP / "selection_manifest.json"); sens = tm["target_sensitivity"]
sel = tm["selected_primary"]

card = ["MODEL CARD  %s  (batch29)" % VER, "=" * 60,
 "Scope: prediction of a per-fragment RNA/plasmid PROXY for K562 autonomous promoter activity",
 "  from native genomic fragment sequence (200-2000 bp, hg19/GRCh37).",
 "THIS IS NOT measured GFP activity and NOT validated HEK293T activity. The 14 local wet",
 "constructs are a different assay/context and were NOT used for fitting or calibration.",
 "",
 "TARGET (parent-approved NEW proxy, declared approximation):",
 "  D_i = 1 + P_i , P_i = plasmid library PCRrep1 + PCRrep2 (iPCR column EXCLUDED).",
 "  a_ir = (r_ir / N_Rr) / (D_i / N_D) ; y_i = mean_r log2(1 + a_ir)",
 "  N_D=%d N_R1=%d N_R2=%d (FULL valid deposited table)." % (passA["library_totals_fulltable"]["N_D_1plusP"], passA["library_totals_fulltable"]["N_R1"], passA["library_totals_fulltable"]["N_R2"]),
 "  One fixed pseudocount per unique measured interval; original barcode multiplicity unknown.",
 "",
 "DATA: GSE78709 SuRE K562 counts (SHA256 8868fa641c2c89bef0c8aa18b6b553030cc7199ea4be4f12c2cd1f3ecd458b85),",
 "  eligibility P_i>=1, canonical chr, 200-2000 bp, zero-RNA rows retained. Selected %d inserts by lowest sha256 (label-free)." % passA["n_selected"],
 "  Sequences: Ensembl GRCh37 REST; assay-oriented = forward for '+', RC once for '-' (author-verified).",
 "SPLIT: %s ; gates_all=%s" % (split["split_rule"], split["gate_all"]),
 "MODELS: E1 ridge(342feat) ; E2/E3 compact LegNet(64ch,110145 params) seeds 20261101/20261102 ; E4 shuffled-label control.",
 "  Trained from scratch on this target only; no old K562 features/weights.",
 "SELECTION: validation group-balanced MAE, ties -> lower seed; primary = %s" % sel,
 "",
 "TEST METRICS (ALL test rows incl. zero-RNA):",
 "  baseline(train-mean) rho=%s MAE=%.5f gbMAE=%.5f" % (tm['baseline_train_mean']['spearman'], tm['baseline_train_mean']['mae'], tm['baseline_train_mean']['group_balanced_mae']),
]
for m in ["E1_ridge", "E2_cnn_true_seed20261101", "E3_cnn_true_seed20261102", "E4_cnn_shuffle_seed20261101"]:
    if m in tm["models"]:
        r = tm["models"][m]["all_test"]
        card.append("  %-32s rho=%s MAE=%.5f gbMAE=%.5f R2=%s topdec_prec=%s" %
                    (m, r["spearman"], r["mae"], r["group_balanced_mae"], r["r2"], r["top_decile"]["precision"]))
card += ["", "CRITERIA (%s): %s  all_met=%s" % (sel, json.dumps(tm["criteria"]), tm["criteria_all_met"]),
 "", "LIMITATIONS:",
 " - unknown barcode aggregation per interval (one-pseudocount approximation);",
 " - low DNA depth (P median 1); label reliability uncertain; RNA detection in selected cohort low;",
 " - reference hg19 sequence, donor variants unresolved;",
 " - retrospective internal chromosome-holdout, NOT prospective/external validation; residual near-homology;",
 " - positive performance here does NOT establish calibrated expression or HEK293T utility.",
 "Author-source note: the authors' own PARM preprocessing normalizes RNA by the iPCR column;",
 " batch29 nevertheless uses the parent-approved D=1+P proxy and reports this separately.",
]
(CARDS / "promoter_atlas_v6_20_1_sureproxy_rc1_card.txt").write_text("\n".join(card) + "\n", encoding="utf-8")
print("card written")


# ---------------- selfcheck ----------------
def sha_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""): h.update(b)
    return h.hexdigest()

checks = []
def chk(name, ok, detail): checks.append({"check": name, "ok": ok, "detail": detail})

chk("source_sha_matches", sha_file(ROOT / "data/breakthrough_20261002/batch27/source/GSE78709_SuRE-counts-K562_B45_B55_LP170105.txt.gz") == "8868fa641c2c89bef0c8aa18b6b553030cc7199ea4be4f12c2cd1f3ecd458b85",
    "frozen SuRE table unchanged")
chk("denominator_is_approved_proxy", passA["denominator_definition"].startswith("D_i = 1 + P_i"), passA["denominator_definition"])
chk("ipcr_excluded_from_target", "iPCR EXCLUDED" in passA["denominator_definition"], "iPCR not in D")
chk("zero_rna_retained", passA["eligibility_rule"]["rna_thresholded"] is False and passA["eligibility_rule"]["zero_rna_rows_retained"] is True, "no RNA filter")
chk("selection_label_free", "label-free" in passA["selection_rule"], "lowest sha256")
chk("selected_10000", passA["n_selected"] == 10000, "n=%d" % passA["n_selected"])
chk("retrieval_db", rt is not None, "n_usable_assay=%s" % (rt.get("n_usable_assay_oriented") if rt else None))
chk("hit_10000", rt and rt.get("n_usable_assay_oriented", 0) >= 10000, "usable=%s" % (rt.get("n_usable_assay_oriented") if rt else None))
chk("split_gates", split["gate_all"], json.dumps(split["gates"]))
chk("target_variance_positive", tf["train_target_variance_positive"], "train std>0")
gate_led = RUNC / "exec_gate" / "ledger.jsonl"
n_fits = sum(1 for l in open(gate_led, encoding="utf-8") if l.strip() and json.loads(l).get("kind") == "reserved" and json.loads(l).get("consumes"))
chk("exec_gate_within_budget", n_fits <= 4, "reserved consuming=%d budget=4" % n_fits)
chk("selection_before_test", os.path.exists(REP / "selection_manifest.json"), "selection manifest frozen pre-test scoring")
selfcheck = {"batch": "batch29", "version": VER, "written_at": now(), "n_checks": len(checks),
             "all_ok": all(c["ok"] for c in checks), "checks": checks}
json.dump(selfcheck, open(REP / "selfcheck.json", "w", encoding="utf-8"), indent=2)
print("selfcheck all_ok=%s" % selfcheck["all_ok"])


# ---------------- artifacts + status ----------------
arts = {}
for p in sorted(list(REP.glob("*.json")) + list((ROOT / "figures/batch29").glob("*.png"))):
    if p.name in ("status.json",): continue
    arts[str(p.relative_to(ROOT)).replace("\\", "/")] = sha_file(p)

reg = rd(BT / "version_registry.json")
sha_before = sha_file(BT / "version_registry.json")
entry = {
 "record_type": "model_candidate", "batch": "batch29", "candidate_version": VER,
 "internal_id": "b29_legnet_sureproxy",
 "track": "K562 autonomous promoter per-fragment RNA/plasmid PROXY (declared approximation)",
 "status": "experimental candidate; NOT promoted; production pointer unchanged (v6.3 main / v6.4 cellline pilot)",
 "parent_decision": "2026-10-03 ~19:45 Asia/Shanghai (batch29 approval of D=1+P proxy)",
 "architecture": "E1 ridge(342); E2/E3/E4 batch02 trackB_grouped_cv LegNetInspired(1,ch=64), scratch",
 "n_parameters_trainable": 110145,
 "input": "one-hot ACGT full native 200-2000bp assay-oriented, equal-exact-length batches, no padding",
 "target": "y_i=mean_r log2(1+a_ir), a_ir=(r_ir/N_Rr)/(D_i/N_D), D_i=1+P_i; iPCR excluded",
 "data_version": "d_batch29_sureproxy_10000", "split_version": "s_batch29_chrblock",
 "run_id": "batch29_worker", "seeds": [20261101, 20261102, 20261103],
 "selection_primary": sel, "criteria": tm["criteria"], "criteria_all_met": tm["criteria_all_met"],
 "fit_budget_compliance": {"declared_max": 4, "observed": n_fits, "ok": n_fits <= 4},
 "creation_time": now(),
 "decision": "worker_complete_awaiting_independent_review; NO promotion; NO HEK293T/prospective claim"}
reg.setdefault("candidates", []).append(entry)
reg.setdefault("batch_appends", []).append(entry)
reg["last_appended_at"] = now(); reg["last_appended_by"] = "batch29 worker (append-only)"
reg["registry_sha256_before_batch29_append"] = sha_before
tmp = BT / "version_registry.json.tmp"
json.dump(reg, open(tmp, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
os.replace(tmp, BT / "version_registry.json")

s0 = tm["models"][sel]["all_test"]
cl = ["", "=== batch29 v6.20.1-sureproxy.rc1  (%s) ===" % now(),
 "PARENT-APPROVED NEW proxy target: D_i = 1 + P_i (plasmid PCRrep1+rep2), iPCR EXCLUDED; y_i = mean_r log2(1 + (r_ir/N_Rr)/(D_i/N_D)).",
 "Eligibility P_i>=1, canonical chr, 200-2000bp, zero-RNA rows RETAINED. 37,282,554 eligible; 10,000 selected by lowest sha256 (label-free).",
 "Sequences: Ensembl GRCh37; assay orientation = forward for '+', RC once for '-' (verified against author repo get_onehot_from_normalized_data.py).",
 "NEW MATERIAL SOURCE FINDING: the van Steensel author PARM preprocessing normalizes RNA by the iPCR column (replicate_rpm/ipcr_rpm); batch29 retains the approved D=1+P proxy and reports this separately.",
 "Four fits behind batch27 exec_gate (budget 4): E1 ridge, E2/E3 LegNet seeds 20261101/20261102, E4 shuffled-label control.",
 "TEST (all test rows incl zero-RNA, n=%d): %s rho=%s MAE=%.5f gbMAE=%.5f; E1 rho=%s; criteria_all_met=%s." %
 (s0["n"], sel, s0["spearman"], s0["mae"], s0["group_balanced_mae"], tm["models"]["E1_ridge"]["all_test"]["spearman"], tm["criteria_all_met"]),
 "Retrospective internal chromosome-holdout only; NOT prospective/external; NOT HEK293T. Production pointer unchanged (v6.3 main / v6.4 cellline pilot). All prior entries preserved."]
with open(BT / "CHANGELOG.txt", "a", encoding="utf-8", newline="\n") as f:
    f.write("\n".join(cl) + "\n")

status = {"batch": "batch29", "version": VER, "phase": "worker_complete_awaiting_independent_review",
 "state": "fits_done_metrics_computed", "experiment_complete": False, "ongoing_task_active": True,
 "protocol": "runs/breakthrough_20261002/batch29/PROTOCOL.txt",
 "review_protocol": "runs/breakthrough_20261002/batch29/REVIEW_PROTOCOL.txt",
 "control_dir": "runs/breakthrough_20261002/control_36", "updated_at_local": now(),
 "execution": {"estimator_executions": n_fits, "budget": 4, "gpu": "RTX 5060 cuda:0", "attempts": [a for a in ["E1_ridge", "E2_cnn_true_seed20261101", "E3_cnn_true_seed20261102", "E4_cnn_shuffle_seed20261101"] if (RUNC / "attempts" / a / "result.json").is_file()]},
 "headline": {"selected_primary": sel, "test": tm["models"][sel]["all_test"],
              "E1_ridge": tm["models"]["E1_ridge"]["all_test"], "criteria": tm["criteria"], "criteria_all_met": tm["criteria_all_met"],
              "baseline_train_mean": tm["baseline_train_mean"], "paired_bootstrap": tm["paired_group_bootstrap_2000"],
              "source_finding": "author PARM code uses iPCR column as normalizer; approved proxy retained"},
 "data": {"n_eligible": passA["n_eligible"], "n_selected": passA["n_selected"], "n_usable_sequence": (rt or {}).get("n_usable_assay_oriented"),
          "split_counts": split["counts"], "gates": split["gates"]},
 "artifacts": arts, "selfcheck": {"all_ok": selfcheck["all_ok"], "path": "reports/breakthrough_20261002/batch29/selfcheck.json"},
 "scope_limitations": "declared proxy approximation; low DNA depth; K562 per-fragment; retrospective internal holdout; not HEK293T; no promotion"}
json.dump(status, open(REP / "status.json", "w", encoding="utf-8"), indent=2)
print("status + registry + changelog written; selfcheck_all_ok=%s" % selfcheck["all_ok"])


# ---------------- next proposal ----------------
prop = {
 "proposal_id": "batch30_sureproxy_followup", "written_at": now(),
 "status": "PROPOSED ONLY - worker self-check; parent decides after fresh independent review",
 "context": "batch29 produced a first bounded K562 per-fragment RNA/plasmid PROXY predictor (declared approximation D=1+P). Source search additionally found the authors' own code normalizes RNA by the iPCR column.",
 "observations": {"selected_primary": sel, "test_all": tm["models"][sel]["all_test"], "criteria_all_met": tm["criteria_all_met"]},
 "options": [
  {"id": "O1", "name": "iPCR-normalized target (author-consistent), declared NEW proxy",
   "rationale": "author PARM code uses replicate_rpm/ipcr_rpm; a bounded re-run with D=iPCR (or D=iPCR+plasmid) would test whether the approved D=1+P proxy materially limited performance.",
   "bounds": "requires explicit parent approval; <=4 fits; reuse frozen cohort/split; report as a DIFFERENT declared target, not a replacement of batch29."},
  {"id": "O2", "name": "independent new-promoter sequence prediction with local wet compatibility",
   "rationale": "user priority is new-sequence performance; evaluate batch29 predictions against the 14 local constructs as clearly-scoped retrospective transfer diagnostics (NOT calibration/validation), and report domain limits.",
   "bounds": "no fitting on 14 constructs; descriptive only."},
  {"id": "O3", "name": "increase DNA depth / barcode-aware target if author iPCR semantics fully resolved",
   "rationale": "reduce label-reliability uncertainty.", "bounds": "needs new data semantics; not required now."}],
 "recommended_finite_next": "Fresh independent review of batch29 first. Then IF the parent approves, O1 (author-consistent iPCR-normalized target) as a bounded 4-fit comparison on the SAME frozen cohort/split, reported as a declared different target; in parallel O2 descriptive transfer diagnostics of the batch29 model on the 14 local constructs.",
 "must_not": ["no promotion", "no HEK293T/prospective claim", "no fitting on the 14 wet constructs", "no credentials/email"]}
json.dump(prop, open(REP / "next_proposal.json", "w", encoding="utf-8"), indent=2)
print("next_proposal written")
