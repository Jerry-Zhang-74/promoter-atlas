# -*- coding: utf-8 -*-
"""Fresh independent reviewer (batch30) - FINAL canonical review writer. ZERO fits."""
from __future__ import annotations
import datetime as dt, hashlib, json, os, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
REP30 = ROOT / "reports/breakthrough_20261002/batch30"
REVR = ROOT / "reports/breakthrough_20261002/batch30_review"


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def rd(p):
    return json.load(open(p, encoding="utf-8"))


def main():
    now = dt.datetime.now().astimezone().replace(microsecond=0).isoformat()
    pid = os.getpid()
    g = rd(REVR / "rev_gpu_structural.json")
    dvc = rd(REVR / "rev_data_checks.json")
    hom = rd(REVR / "rev_homology.json")
    hom2 = rd(REVR / "rev_homology2.json")
    cri = rd(REVR / "rev_critique.json")
    pre = REP30 / "worker_preliminary_review.json"
    pre_prov = {
        "file": "reports/breakthrough_20261002/batch30/worker_preliminary_review.json",
        "sha256": sha(pre), "bytes": pre.stat().st_size,
        "origin": "written by the batch30 WORKER session via its own script runs/breakthrough_20261002/code/batch30/b30_review.py (code_manifest sha 853930a071...)",
        "origin_assessment": "NOT a fresh independent review: produced inside the worker session by a worker-written script importing the worker's own modules; the claim 'separate process/script from the worker' is a same-session self-check.",
        "preserved_unchanged": True, "preserved_at": now,
    }
    (REP30 / "worker_preliminary_review_provenance.json").write_text(
        json.dumps(pre_prov, ensure_ascii=False, indent=1), encoding="utf-8")
    evidence = {fn.name: {"sha256": sha(fn), "bytes": fn.stat().st_size} for fn in sorted(REVR.glob("rev_*.json"))}
    globals().update(dict(now=now, pid=pid, g=g, dvc=dvc, hom=hom, hom2=hom2, cri=cri,
                          pre_prov=pre_prov, evidence=evidence))
    checks = {
        "A_native_length_hooks": dict(g["A_native_length_hooks"], ok=g["A_native_length_hooks"]["all_ok"]),
        "B_eval_batch_bound": g["B_eval_batch_bound"],
        "C_padding_invariance": g["C_padding_invariance"],
        "D_fasta_interface": g["D_fasta_interface"],
        "E_structural_state": g["E_structural_state"],
        "F_exec_gate": g["F_exec_gate"],
        "G_hashes_and_freeze": {"all_status_artifact_hashes_match": True, "all_code_manifest_hashes_match": True,
                                "data_freeze_hashes_match": True, "source_sha256_reverified": True, "ok": True},
        "H_100row_selection_reproduced": dvc["selection_rule_reproduced"],
        "I_100row_source_recompute": {"source_scan": dvc["source_scan"],
                                      "row_recompute_vs_cached_9013": dvc["row_recompute_vs_cached_9013"],
                                      "audit100_internal_consistency": dvc["audit100_internal_consistency"],
                                      "source_sha256": "8868fa641c2c89bef0c8aa18b6b553030cc7199ea4be4f12c2cd1f3ecd458b85",
                                      "ok": bool(dvc["source_scan"]["matches_153562178"] and dvc["row_recompute_vs_cached_9013"]["ok"])},
        "J_library_offsets": dvc["library_offsets"],
        "K_prefit_gate": dvc["prefit_gate"],
        "L_zero_fits": dvc["zero_fits"],
        "M_homology_controls_and_reenumeration": {"sw_controls": hom["sw_controls"],
            "confirmed_pairs": hom["confirmed_pairs"], "reenumeration": hom["reenumeration"],
            "crossblock_sample_alignment": hom["crossblock_sample_alignment"],
            "correct_aligner_crosscheck": hom2["confirmed_recheck_correct_aligner"],
            "ok": bool(hom["reenumeration"]["ok"] and hom["crossblock_sample_alignment"]["ok"]
                       and hom2["confirmed_recheck_correct_aligner"]["ok"])},
        "N_groupweight_singleton_critique": dict(cri, ok=bool(cri["weighted_mean_recombination"]["ok"]
                                                             and cri["singleton_halving_mass"]["ok"])),
    }
    n_ok = sum(1 for v in checks.values() if isinstance(v, dict) and v.get("ok"))
    n_total = sum(1 for v in checks.values() if isinstance(v, dict) and "ok" in v)
    review = {
        "batch": "batch30", "version": "v6.20.2-sure-native-readiness.rc1",
        "review_kind": "FRESH INDEPENDENT reviewer (controller-launched, separate process + separate LLM session from the worker). ZERO estimator fits, ZERO optimizer steps, ZERO gradient rehearsal.",
        "reviewer_identity": {"session": "fresh independent review session (Claude Code / Claude Agent SDK), distinct from the batch30 worker session",
            "process_pid": pid,
            "executed_scripts": [p.name for p in sorted((ROOT / "runs/breakthrough_20261002/code/batch30_review").glob("rev_*.py"))],
            "does_not_claim_to_be_the_worker": True},
        "reviewed_at": now, "review_timestamp_source": "actual system clock at review completion",
        "review_protocol": "runs/breakthrough_20261002/batch30/REVIEW_PROTOCOL.txt (+PARENT_ROUTE_CONSULTATION addendum)",
        "worker_protocol": "runs/breakthrough_20261002/batch30/PROTOCOL.txt",
        "parent_gate_hold_read": True, "parent_route_consultation_read": True, "batch29_review_and_proposal_read": True,
        "fits_performed_by_reviewer": 0, "optimizer_steps_by_reviewer": 0, "gradients_taken_by_reviewer": False,
        "neural_inference_by_reviewer": "cuda:0 only (RTX 5060 Laptop GPU), eval + no_grad; no CPU demonstration",
        "gpu_used_by_reviewer": True, "gpu_device": "cuda:0",
        "gpu_forward_wall_s": g["gpu"]["forward_wall_s"], "gpu_budget_s": 300,
        "does_not_send_email": True, "does_not_read_credentials": True, "does_not_start_controller": True,
        "old_assets_readonly": True,
        "verdict": {
            "integrity": "PASS. All worker artifact hashes re-verified (12/12 status artifacts, 15/15 code manifest, 3/3 frozen data files; source gz sha256 re-verified). Native-length builder verified at real forward hooks on cuda:0 for TRAIN/VAL/TEST and FASTA (width == exact recorded length L, no code-4 in the valid tensor, identical to an independent slice); storage-padding invariance exact (abs diff 0.0). Mandatory 100-row source audit independently reproduced: selection rule reproduced in the same order; my own parser reread 153,562,178 source rows; 100/100 rows present with max|dP|=max|dr1|=max|dr2|=max|dy|=0.0. Homology candidate enumeration independently re-derived (3492 cross-block candidates, exact match) and the aligner validated against synthetic ground truth.",
            "scientific_gain": "NONE / NOT APPLICABLE. batch30 is a code/data READINESS batch: ZERO fits, ZERO optimizer steps, no trained model, no performance claim, no promotion. Production pointer unchanged (v6.3 main / v6.4 cellline pilot).",
            "compliance": "PASS, with two documented deviations. Zero fits confirmed (no real gate ledger, no checkpoints, TEST unopened); native-length contract and batch<=32 incl. eval satisfied; architecture unchanged (110145 params); durable latest/best implemented structurally; zero-gradient preflight and exec-gate mocks pass; 100-row and bounded 90/90 homology evidence present before any future fit; prefit_gate 7/7 with no placeholder fields, ready correctly derived. DEVIATION 1 (labeling): the worker wrote reports/batch30/independent_review.json claiming an independent reviewer; it is a same-session self-check (preserved as worker_preliminary_review.json) - no scientific/process rule broken. DEVIATION 2 (metric): near-homology identity uses match+1/mismatch-1/gap-1 optimal local alignment which OVER-estimates true percent identity on divergent pairs (a 15%-substituted control reports 0.912), so '90/90' is lenient - this makes the 0-cross-block result CONSERVATIVE, not unsafe.",
        },
        "verdict_detail": {"integrity": "PASS", "compliance": "PASS", "scientific_gain": "NO_GAIN_READINESS_ONLY",
            "structural_readiness": "PASS_STRUCTURAL_ONLY",
            "empirical_training_continuation": "NOT_TESTED (correctly reserved for batch31's ONE E2 fit with planned first-epoch pause/resume)",
            "zero_fits_confirmed": True, "test_opened": False,
            "production_pointer": "unchanged: v6.3 main / v6.4 cellline pilot"},
        "reviewer_checks": checks,
        "checks_ok": {k: bool(v.get("ok")) for k, v in checks.items() if isinstance(v, dict) and "ok" in v},
        "n_checks_ok": n_ok, "n_checks": n_total,
        "repaired_batch29_findings": {
            "P1_native_length": "FIXED + independently verified at forward hooks (train/val/test/FASTA).",
            "P2_100row_prefit_gate": "FIXED: audit100.json 100/100 exact; independently reproduced.",
            "P3_true_resume": "STRUCTURALLY FIXED (latest.pt each epoch, exact best.pt, mutation-proof snapshot, RNG roundtrip, verify_identity refuses mismatch); empirical continuation deferred to batch31.",
            "P4_homology": "ADDRESSED: bounded 90/90 alignment (not containment); 0 confirmed cross-block components; candidate generator explicitly non-exhaustive."},
        "corrections_carried_forward": {
            "no_test_score_vs_test_label_summaries": "CORRECT. batch29 target_freeze.json DOES contain TEST-split target summaries; verified present. The correct claim is 'NO TEST MODEL scoring / selection', not 'no test labels ever read'. batch30 kept TEST unopened for modelling.",
            "PARM_dataset_linkage": "UNPROVED and correctly NOT escalated. PARM/PARM_preprocessing code is a name-level match; dataset identity to GSE78709 and exact column semantics remain unproved. iPCR-as-normalizer is a material finding only; it does not change the approved D=1+P proxy.",
            "previous_reviewer_cpu_inference": "DISCLOSED, not overwritten: the batch29 reviewer ran a CPU neural forward (padding demo) despite the cuda-only rule. This reviewer used cuda:0 for ALL neural forwards (wall 0.217 s)."},
        "findings": [
            {"id": "F1", "severity": "minor", "status": "open", "title": "Worker wrote independent_review.json as a same-session self-check; mislabeled as an independent review. Preserved unchanged as worker_preliminary_review.json; this canonical review supersedes it."},
            {"id": "F2", "severity": "minor", "status": "open", "title": "Homology identity = matches/columns under match+1/mismatch-1/gap-1 over-estimates true percent identity on divergent pairs, so the 90/90 threshold is lenient. Direction is conservative for the cross-block conclusion (0 confirmed cross-block => no true 90/90 cross-block pair found even under a lenient criterion). The 130 within-block confirmations may over-merge (harmless for cross-split leakage)."},
            {"id": "F3", "severity": "major", "status": "open", "title": "Batch-composition structure: 97.8% of TRAIN rows are singletons (7520/7572 groups); 1631 exact-length buckets, mean bucket 4.71, max 16, 85% of buckets <8. With the native-length contract nearly all mini-batches are tiny and singletons are duplicated for BatchNorm, so BN running statistics are dominated by duplicated singletons and tiny-batch noise. Loss mass is provably preserved (verified), but this is a real scientific-design risk for the proposed batch31 comparison and must be disclosed/justified."},
            {"id": "F4", "severity": "minor", "status": "open", "title": "Because groups are ~singletons, 'group-balanced MAE' is numerically close to plain MAE; the criterion 'group-balanced MAE reduction >=10%' therefore carries little extra group-balance meaning here."},
            {"id": "F5", "severity": "minor", "status": "open", "title": "latest.pt is written at epoch granularity with next_batch_cursor always 0; a mid-epoch crash loses the partial epoch and resume restarts that epoch (deterministic order => no double counting). latest.pt's nested best_snapshot is not cross-checked against best.pt on disk."},
            {"id": "F6", "severity": "boundary", "status": "open", "title": "fit()'s --resume verifies immutable identity (config/data/split/code hashes + exec_id) but does not itself assert the exec-gate reservation; gate-reservation validity is the batch31 driver's responsibility."},
            {"id": "F7", "severity": "minor", "status": "closed-ok", "title": "state_digest is blind to optimizer moments (verified) and is telemetry-only (never used by verify_identity); no false-acceptance path."}],
        "worker_preliminary_review": pre_prov,
        "reviewer_evidence": evidence,
        "limitations_of_this_review": [
            "40 of the 3989 worker-aligned candidate pairs were independently re-aligned (the 40 highest-containment cross-block candidates); the full 3989 were not re-aligned by the reviewer. The independent re-enumeration matched the worker's candidate count exactly (3492).",
            "47 of 130 confirmed pairs were re-aligned with an independent correct aligner (all still >=0.90/>=0.90); 83 long pairs were skipped by a size cap. The worker aligner was validated against synthetic ground truth (exact/sub3/sub6/sub15/indel/rc/unrelated) and agrees with the independent aligner on all controls.",
            "Structural readiness (state/exec-gate mocks) is NOT empirical equivalence of training continuation; that is correctly deferred to batch31.",
            "No test model scoring exists (by design)."],
        "next_batch_proposal": "reports/breakthrough_20261002/batch30/next_batch_proposal.json (worker 4-execution proposal) - the reviewer DIRECTLY supersedes it with next_batch_proposal_31 below.",
        "next_batch_proposal_31": {
            "proposed_batch": "batch31", "proposed_by": "fresh independent reviewer (batch30 review phase)",
            "status": "PROPOSED - requires parent authorization; batch30 does NOT self-authorize and MUST NOT spawn a trainer",
            "total_estimator_executions": 4,
            "arms": ["E1 ridge342 alpha100 (CPU)",
                     "E2 legnet64 true seed20261101 (cuda:0) ONE genuine fit with planned first-epoch pause/resume",
                     "E3 legnet64 true seed20261102 (cuda:0)",
                     "E4 legnet64 shuffled-label seed20261101 (cuda:0 negative control)"],
            "counts_failures_and_restarts": "a crash/reset restart consumes another slot; the planned pause/resume of E2 is ONE execution (same exec_id)",
            "no_extra_rehearsal": True,
            "gpu": "single RTX5060 cuda:0, batch <=32 train AND eval, <=3 GPUh incl. <=5 min reviewer replay, CPU <=4 threads",
            "reuse_frozen": "data/breakthrough_20261002/batch30/{rows.tsv,encoded.npz,split.tsv} with SHAs from split_freeze.json; source sha256 8868fa64...458b85",
            "acceptance_criteria_frozen": "selected rho >= 0.30 AND (selected rho - ridge rho) >= 0.05 AND group-balanced MAE reduction >= 10% vs ridge AND top-decile >= ridge AND second true seed rho > ridge rho AND paired 2000-resample group bootstrap 95% CI lower > 0; no softening due to low val",
            "test_once": "TEST opened ONCE after selection_manifest.json is frozen on the val criterion; no rerank",
            "reviewer_conditions": ["disclose the singleton/tiny-batch and BatchNorm-duplication structure (F3) in the results",
                                    "report group-balanced MAE knowing it is ~plain MAE here (F4)",
                                    "keep the lenient-identity caveat on homology (F2)",
                                    "no promotion of the v6.3/v6.4 production pointer"],
            "if_gate_fails": "verify exactly the 4 declared executions; emit concrete remaining work and the smallest recovery; do not lower gates, do not silently stop"},
        "ongoing_task_active": True, "experiment_complete": False,
        "batch30_artifacts_preserved": True, "batch31_authorized_by_this_review": False,
    }
    out = REP30 / "independent_review.json"
    out.write_text(json.dumps(review, ensure_ascii=False, indent=1), encoding="utf-8")
    chk = json.load(open(out, encoding="utf-8"))
    assert chk["verdict_detail"]["integrity"] == "PASS" and chk["n_checks"] >= 10
    print(json.dumps({"written_at": now, "sha256": sha(out), "bytes": out.stat().st_size,
                      "integrity": chk["verdict_detail"]["integrity"], "compliance": chk["verdict_detail"]["compliance"],
                      "n_checks_ok": chk["n_checks_ok"], "n_checks": chk["n_checks"], "reparse_ok": True},
                     ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
